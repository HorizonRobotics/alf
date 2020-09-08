# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay buffer."""

from absl import logging
import gin
import math
import numpy as np
import torch
import torch.nn as nn

import alf
from alf import data_structures as ds
from alf.data_structures import namedtuple
from alf.nest.utils import convert_device
from alf.utils.common import warning_once
from alf.utils.data_buffer import atomic, RingBuffer
from alf.utils import checkpoint_utils

from .segment_tree import SumSegmentTree, MaxSegmentTree, MinSegmentTree

BatchInfo = namedtuple(
    "BatchInfo", ["env_ids", "positions", "importance_weights"],
    default_value=())


@gin.configurable
class ReplayBuffer(RingBuffer):
    """Replay buffer with RingBuffer as implementation.

    Terminology: consistent with RingBuffer, we use ``pos`` to refer to the
    always increasing position of an element in the infinitly long buffer,
    and ``idx`` as the actual index of the element in the underlying store
    (``_buffer``).  That means ``idx == pos % _max_length`` is always true,
    and one should use ``_buffer[idx]`` to retrieve the stored data.
    """

    ONE_MINUS = np.float32(1) - np.finfo(np.float32).eps

    def __init__(self,
                 data_spec,
                 num_environments,
                 max_length=1024,
                 num_earliest_frames_ignored=0,
                 prioritized_sampling=False,
                 initial_priority=1.0,
                 recent_data_steps=1,
                 recent_data_ratio=0.,
                 with_replacement=False,
                 device="cpu",
                 allow_multiprocess=False,
                 keep_episodic_info=None,
                 step_type_field="step_type",
                 postprocess_exp_fn=None,
                 enable_checkpoint=False,
                 name="ReplayBuffer"):
        """
        Args:
            data_spec (alf.TensorSpec): spec of an entry nest in the buffer.
            num_environments (int): total number of parallel environments
                stored in the buffer.
            max_length (int): maximum number of time steps stored in buffer.
            num_earliest_frames_ignored (int): ignore the earlist so many frames
                from the buffer when sampling. This is typically required when
                FrameStacker is used. ``keep_episodic_info`` will be set to True
                if ``num_earliest_frames_ignored`` > 0 as ``FrameStacker`` need
                episode information.
            prioritized_sampling (bool): Use prioritized sampling if this is True.
            initial_priority (float): initial priority used for new experiences.
                The actual initial priority used for new experience is the maximum
                of this value and the current maximum priority of all experiences.
            recent_data_steps (int): the most recent so many steps of data is
                considered as recent data for ``get_batch()``. Note that this
                quantity is per environment.
            recent_data_ratio (float): ``recent_data_ratio * batch_size`` samples
                in the batch are sampled from recent data for ``get_batch()``.
            with_replacement (bool): If False, sample without replacement whenever
                poissible for ``get_batch()``. If True, a batch may contains
                duplicated samples.
            device (string): "cpu" or "cuda" where tensors are created.
            allow_multiprocess (bool): whether multiprocessing is supported.
            keep_episodic_info (bool): index episode start and ending positions.
                If None, its value will be set to True if ``num_earliest_frames_ignored``>0
            step_type_field (string): path to the step_type field in exp nest.
                This and the following fields are for hindsight relabeling.
            postprocess_exp_fn (callable): function to postprocess experience.
                Args:
                    ``buffer`` (``ReplayBuffer``): the replay buffer object.
                    ``batch`` (nest): nested sampled experience of shape
                        ``[batch_size, batch_length, ...]``.
                    ``batch_info`` (BatchInfo): sample information for the batch.
                These three arguments are required.  Other optional arguments
                    could be populated by gin.
                Returns:
                    updated ``(batch, batch_info)``.
            enable_checkpoint (bool): whether checkpointing this replay buffer.
            name (string): name of the replay buffer object.
        """
        super().__init__(
            data_spec,
            num_environments,
            max_length=max_length,
            device=device,
            allow_multiprocess=allow_multiprocess,
            name=name)
        self._num_earliest_frames_ignored = num_earliest_frames_ignored
        if num_earliest_frames_ignored > 0:
            if keep_episodic_info is None:
                keep_episodic_info = True
        if keep_episodic_info is None:
            keep_episodic_info = False
        self._step_type_field = step_type_field
        self._prioritized_sampling = prioritized_sampling
        if prioritized_sampling:
            self._mini_batch_length = 1
            tree_size = self._max_length * num_environments
            self._sum_tree = SumSegmentTree(tree_size, device=device)
            self._max_tree = MaxSegmentTree(tree_size, device=device)
            self._initial_priority = torch.tensor(
                initial_priority, dtype=torch.float32, device=device)
        self._postprocess_exp_fn = postprocess_exp_fn
        self._keep_episodic_info = keep_episodic_info
        self._recent_data_steps = recent_data_steps
        self._recent_data_ratio = recent_data_ratio
        self._with_replacement = with_replacement
        if self._keep_episodic_info:
            # _indexed_pos records for each timestep of experience in the
            # buffer the raw position of the first step of the episode in
            # the buffer (without modulo _max_length).  If it's a ``FIRST``
            # step, it records the position of the last step in the buffer,
            # which could either be a ``LAST`` or a ``MID`` step, or the
            # ``FIRST`` step itself if no more data is stored.
            self.register_buffer(
                "_indexed_pos",
                torch.zeros((self._num_envs, self._max_length),
                            dtype=torch.int64,
                            device=device))
            # If a ``FIRST`` step is overwritten, the remaining steps in the
            # replay buffer become headless, and the corresponding _indexed_pos
            # will be backed up in this tensor.  Only one ``FIRST`` step for
            # each environment needs to be backed up -- the most recently
            # overwritten.
            self.register_buffer(
                "_headless_indexed_pos",
                torch.zeros(self._num_envs, dtype=torch.int64, device=device))
        checkpoint_utils.enable_checkpoint(self, enable_checkpoint)

    @property
    def initial_priority(self):
        """The initial priority used for newly added experiences.

        We use a large value for initial priority so that a new experience can
        be used for training sooner. We make it at least 1.0 so that it can never
        be very small.
        """
        return convert_device(
            torch.max(self._max_tree.summary(), self._initial_priority))

    def _initialize_priority(self, env_ids):
        last_pos = self._current_pos[env_ids] - self._mini_batch_length
        indices = self._env_id_idx_to_index(env_ids, self.circular(last_pos))
        values = torch.full(indices.shape, self.initial_priority)

        if self._mini_batch_length > 1:
            last_pos = self._current_pos[env_ids] - 1
            indices = torch.cat([
                indices,
                self._env_id_idx_to_index(env_ids, self.circular(last_pos))
            ])
            values = torch.cat(
                [values,
                 torch.zeros_like(last_pos, dtype=torch.float32)])

        self._update_segment_tree(indices, values)

    def _update_segment_tree(self, indices, values):
        self._sum_tree[indices] = values
        self._max_tree[indices] = values

    def _env_id_idx_to_index(self, env_ids, idx):
        """Convert env_id, idx in batched buffer to indices in SegmentTree."""
        return env_ids * self._max_length + idx

    def _index_to_env_id_idx(self, indices):
        """Convert indices used by SegmentTree to (env_id, idx)."""
        env_ids = indices / self._max_length
        return env_ids, indices % self._max_length

    def _change_mini_batch_length(self, mini_batch_length):
        env_ids = torch.arange(self._num_envs, dtype=torch.int64)
        if mini_batch_length > self._mini_batch_length:
            for i in range(self._mini_batch_length, mini_batch_length):
                last_idx = self.circular(self._current_pos - i)
                indices = self._env_id_idx_to_index(env_ids, last_idx)
                valid = torch.where(self._current_size >= i)[0]
                indices = indices[valid]
                values = torch.zeros_like(indices, dtype=torch.float32)
                self._update_segment_tree(indices, values)
        elif mini_batch_length < self._mini_batch_length:
            for i in range(mini_batch_length, self._mini_batch_length):
                last_idx = self.circular(self._current_pos - i)
                indices = self._env_id_idx_to_index(env_ids, last_idx)
                valid = torch.where(self._current_size >= i)[0]
                indices = indices[valid]
                values = torch.full(indices.shape, self.initial_priority)
                self._update_segment_tree(indices, values)
        self._mini_batch_length = mini_batch_length

    @torch.no_grad()
    def update_priority(self, env_ids, positions, priorities):
        """Update the priorities for the given experiences.

        Args:
            env_ids (Tensor): 1-D int64 Tensor.
            positions (Tensor): 1-D int64 Tensor with same shape as ``env_ids``.
                These positions should be obtained from the BatchInfo returned
                by ``get_batch()``.
            priorities (Tensor): 1-D float Tensor with same shape as ``env_ids``.
                The elements are the new priorities corresponds to experiences
                indicated by ``(env_ids, positions)``
        """
        # If positions are outdated, we don't update their priorities.
        valid, = torch.where(positions >= self._current_pos[env_ids] -
                             self._current_size[env_ids])
        indices = self._env_id_idx_to_index(env_ids[valid],
                                            self.circular(positions[valid]))
        self._update_segment_tree(indices, priorities[valid])

    @atomic
    @torch.no_grad()
    def add_batch(self, batch, env_ids=None, blocking=False):
        """Add a batch of entries to buffer updating indices as needed.

        We build an index of episode beginning indices for each element
        in the buffer.  The beginning point stores where episode end is.

        Args:
            batch (Tensor): of shape ``[batch_size] + tensor_spec.shape``
            env_ids (Tensor): If ``None``, ``batch_size`` must be
                ``num_environments``. If not ``None``, its shape should be
                ``[batch_size]``. We assume there are no duplicate ids in
                ``env_id``. ``batch[i]`` is generated by environment
                ``env_ids[i]``.
            blocking (bool): If ``True``, blocks if there is no free slot to add
                data.  If ``False``, enqueue can overwrite oldest data.
        """
        with alf.device(self._device):
            env_ids = self.check_convert_env_ids(env_ids)
            if self._keep_episodic_info:
                assert not blocking, (
                    "HER replay buffer doesn't wait for dequeue to free up " +
                    "space, but instead just overwrites.")
                batch = convert_device(batch)
                # 1. save episode beginning data that will be overwritten
                overwriting_pos = self._current_pos[env_ids]
                buffer_step_types = alf.nest.get_field(self._buffer,
                                                       self._step_type_field)
                first, = torch.where(
                    (buffer_step_types[(env_ids, self.circular(overwriting_pos)
                                        )] == ds.StepType.FIRST) *
                    (self._current_size[env_ids] == self._max_length))
                first_env_ids = env_ids[first]
                first_step_idx = self.circular(overwriting_pos[first])
                self._headless_indexed_pos[first_env_ids] = self._indexed_pos[(
                    first_env_ids, first_step_idx)]

            # 2. enqueue batch
            self.enqueue(batch, env_ids, blocking=blocking)
            if self._prioritized_sampling:
                self._initialize_priority(env_ids)
                if self._num_earliest_frames_ignored > 0:
                    # Make sure the priortized sampling ignores the earliest
                    # frames by setting their priorities to 0.
                    current_pos = self._current_pos[env_ids]
                    pos = current_pos - self._current_size[env_ids]
                    pos = pos + self._num_earliest_frames_ignored - 1
                    pos = torch.min(pos, current_pos - 1)
                    self.update_priority(
                        env_ids, pos, torch.zeros_like(
                            pos, dtype=torch.float32))

            if self._keep_episodic_info:
                # 3. Update associated episode end indices
                # 3.1. find ending steps in batch (incl. MID and LAST steps)
                step_types = alf.nest.get_field(batch, self._step_type_field)
                non_first, = torch.where(step_types != ds.StepType.FIRST)
                # 3.2. update episode ending positions
                self._store_episode_end_pos(non_first, overwriting_pos,
                                            env_ids)
                # 3.3. initialize episode beginning positions to itself
                epi_first, = torch.where(step_types == ds.StepType.FIRST)
                self._indexed_pos[(env_ids[epi_first],
                                   self.circular(overwriting_pos[epi_first])
                                   )] = overwriting_pos[epi_first]

    @atomic
    @torch.no_grad()
    def get_batch(self, batch_size, batch_length):
        """Randomly get ``batch_size`` trajectories from the buffer.

        It could hindsight relabel the experience via postprocess_exp_fn.

        Note: The environments where the sampels are from are ordered in the
            returned batch.

        Args:
            batch_size (int): get so many trajectories
            batch_length (int): the length of each trajectory
        Returns:
            tuple:
                - nested Tensors: The samples. Its shapes are [batch_size, batch_length, ...]
                - BatchInfo: Information about the batch. Its shapes are [batch_size].
                    - env_ids: environment id for each sequence
                    - positions: starting position in the replay buffer for each sequence.
                    - importance_weights: priority divided by the average of all
                        non-zero priorities in the buffer.
        """
        with alf.device(self._device):
            recent_batch_size = 0
            if self._recent_data_ratio > 0:
                d = batch_length - 1 + self._num_earliest_frames_ignored
                avg_size = self.total_size / float(self._num_envs) - d
                if (avg_size * self._recent_data_ratio >
                        self._recent_data_steps):
                    # If this condition is False, regular sampling without considering
                    # recent data will get enough samples from recent data. So
                    # we don't need to have a separate step just for sampling from
                    # the recent data.
                    recent_batch_size = math.ceil(
                        batch_size * self._recent_data_ratio)

            normal_batch_size = batch_size - recent_batch_size
            if self._prioritized_sampling:
                info = self._prioritized_sample(normal_batch_size,
                                                batch_length)
            else:
                info = self._uniform_sample(normal_batch_size, batch_length)

            if recent_batch_size > 0:
                # Note that _uniform_sample() get samples duplicated with those
                # from _recent_sample()
                recent_info = self._recent_sample(recent_batch_size,
                                                  batch_length)
                info = alf.nest.map_structure(lambda *x: torch.cat(x),
                                              recent_info, info)

            start_pos = info.positions
            env_ids = info.env_ids

            idx = start_pos.reshape(-1, 1)  # [B, 1]
            idx = self.circular(
                idx + torch.arange(batch_length).unsqueeze(0))  # [B, T]
            out_env_ids = env_ids.reshape(-1, 1).expand(
                batch_size, batch_length)  # [B, T]
            result = alf.nest.map_structure(lambda b: b[(out_env_ids, idx)],
                                            self._buffer)

            if alf.summary.should_record_summaries():
                alf.summary.scalar(
                    "replayer/" + self._name + ".original_reward_mean",
                    torch.mean(result.reward[:-1]))

            if self._postprocess_exp_fn:
                result, info = self._postprocess_exp_fn(self, result, info)

        if alf.get_default_device() == self._device:
            return result, info
        else:
            return convert_device(result), convert_device(info)

    def _recent_sample(self, batch_size, batch_length):
        return self._sample(batch_size, batch_length, self._recent_data_steps)

    def _uniform_sample(self, batch_size, batch_length):
        min_size = self._current_size.min() - self._num_earliest_frames_ignored
        assert min_size >= batch_length, (
            "Not all environments have enough data. The smallest data "
            "size is: %s Try storing more data before calling get_batch" %
            min_size)
        return self._sample(batch_size, batch_length)

    def _sample(self,
                batch_size,
                batch_length,
                sample_from_recent_n_data_steps=None):
        batch_size_per_env = batch_size // self._num_envs
        remaining = batch_size % self._num_envs
        env_ids = torch.arange(self._num_envs).repeat(batch_size_per_env)
        if remaining > 0:
            remaining_eids = torch.randperm(self._num_envs)[:remaining]
            env_ids = torch.cat([env_ids, remaining_eids], dim=0)

        r = torch.rand(*env_ids.shape)
        if not self._with_replacement:
            # For sampling without replacement,  we want r's for the same env to
            # be roughly evenly spaced.
            num_samples_per_env = batch_size_per_env * torch.ones(
                (self._num_envs, ), dtype=torch.int64)
            # Each i means that the corresponding sample is the i-th sample from
            # that environment.
            i = torch.arange(batch_size_per_env).unsqueeze(-1).expand(
                -1, self._num_envs).reshape(-1)
            if remaining > 0:
                num_samples_per_env[remaining_eids] += 1
                remaining_i = batch_size_per_env * torch.ones(
                    (remaining, ), dtype=torch.int64)
                i = torch.cat([i, remaining_i])

            r = (i + r) / num_samples_per_env[env_ids]
            # Because of limited floating point precision (e.g. (3 + ONE_MINUS) / 4 == 1.0),
            # we need to make sure r is smaller than 1.
            r = torch.clamp(r, max=self.ONE_MINUS)

        d = batch_length - 1 + self._num_earliest_frames_ignored
        num_positions = self._current_size - d
        if sample_from_recent_n_data_steps is not None:
            num_positions = torch.clamp(
                num_positions, max=sample_from_recent_n_data_steps)
        pos = (r * num_positions[env_ids]).to(torch.int64)
        pos += (self._current_pos - num_positions - batch_length + 1)[env_ids]
        info = BatchInfo(env_ids=env_ids, positions=pos)
        return info

    def _prioritized_sample(self, batch_size, batch_length):
        if batch_length != self._mini_batch_length:
            if self._mini_batch_length > 1:
                warning_once(
                    "It is not advisable to use different batch_length "
                    "for different calls to get_batch(). Previous batch_length=%d "
                    "new batch_length=%d" % (self._mini_batch_length,
                                             batch_length))
            self._change_mini_batch_length(batch_length)

        total_weight = self._sum_tree.summary()
        assert total_weight > 0, (
            "There is no data in the "
            "buffer or the data of all the environments are shorter than "
            "batch_length=%s" % batch_length)

        r = torch.rand((batch_size, ))
        if not self._with_replacement:
            r = (
                r + torch.arange(batch_size, dtype=torch.float32)) / batch_size
        r = r * total_weight
        indices = self._sum_tree.find_sum_bound(r)
        env_ids, idx = self._index_to_env_id_idx(indices)
        info = BatchInfo(env_ids=env_ids, positions=self._pad(idx, env_ids))
        avg_weight = self._sum_tree.nnz / total_weight
        info = info._replace(
            importance_weights=self._sum_tree[indices] * avg_weight)

        return info

    def _pad(self, x, env_ids):
        """Make ``x`` (any index) absolute positions in the RingBuffer.

        This is the reverse of ``circular()``, and is useful when trying to
        compute the distance from one index to the other.  This is done by
        adding multiples of _max_length back to the index.

        NOTE, this operation depends on the _current_pos of the RingBuffer,
        and can generate different results if new data are added to the buffer.

        position = idx + n L
        current_pos - L <= idx + n L < current_pos
        current_pos - idx - 1 - L < n L <= current_pos - idx - 1
        n = (current_pos - idx - 1) / L
        """
        return ((self._current_pos[env_ids] - x - 1) /
                self._max_length) * self._max_length + x

    def _store_episode_end_pos(self, non_first, pos, env_ids):
        """Update _indexed_pos and _headless_indexed_pos for episode end pos.

        Args:
            non_first_idx (tensor): index of the added batch of exp, which are
                not FIRST steps.  We need to update last step pos for all
                these env_ids[non_first_idx].
            pos (tensor): position of the stored batch.
            env_ids (tensor): env_ids of the stored batch.
        """
        _env_ids = env_ids[non_first]
        _pos = pos[non_first]
        # Because this is a non-first step, the previous step's first step
        # is the same as this stored step's first step.  Look it up.
        prev_pos = _pos - 1
        prev_idx = self.circular(prev_pos)
        prev_first = self._indexed_pos[(_env_ids, prev_idx)]
        prev_first_idx = self.circular(prev_first)
        # Record pos of ``FIRST`` step into the current _indexed_pos
        self._indexed_pos[(_env_ids, self.circular(_pos))] = prev_first

        # Store episode end into the ``FIRST`` step of the episode.
        has_head_cond = prev_first > _pos - self._max_length
        has_head, = torch.where(has_head_cond)
        self._indexed_pos[(_env_ids[has_head],
                           prev_first_idx[has_head])] = _pos[has_head]
        # For a headless episode whose ``FIRST`` step was overwritten by new
        # data, the current step has to belong to the same episode as all the
        # other steps in the buffer, i.e. episode is longer than max_length of
        # the buffer.  This means prev_first <= _pos - max_length.
        headless, = torch.where(torch.logical_not(has_head_cond))
        self._headless_indexed_pos[_env_ids[headless]] = _pos[headless]

    def _get_episode_end_pos(self, idx, env_ids):
        """Use _indexed_pos and _headless_indexed_pos to get last step pos.
        """
        # Look up ``FIRST`` step position, then look up the stored last pos
        first_step_pos = self._indexed_pos[(env_ids, idx)]
        first_step_idx = self.circular(first_step_pos)
        result = self._indexed_pos[(env_ids, first_step_idx)]
        # If current step is FIRST, skip second lookup.
        buffer_step_types = alf.nest.get_field(self._buffer,
                                               self._step_type_field)
        is_first = buffer_step_types[(env_ids, idx)] == ds.StepType.FIRST
        result[is_first] = first_step_pos[is_first]
        # If the current timestep is "headless", i.e. whose ``FIRST`` step
        # has been overwritten by new data added into the RingBuffer,
        # retrieve pos from _headless_indexed_pos.
        #
        # In this case, the _current_pos will be more than _max_length over
        # the position of the episode's first_step from _indexed_pos.
        # (The case where current step is a ``FIRST`` step can be safely
        # ignored because the first_step_pos points to episode end which
        # is guarranteed to be existing.)
        headless = self._current_pos[
            env_ids] > first_step_pos + self._max_length
        headless_env_ids = env_ids.expand_as(idx)[headless]
        result[headless] = self._headless_indexed_pos[headless_env_ids]
        return result

    def steps_to_episode_end(self, pos, env_ids):
        """Get the distance to the closest episode end in future.

        Args:
            pos (tensor): shape ``L``, positions of the current timesteps in
                the replay buffer.
            env_ids (tensor): shape ``L``
        Returns:
            tensor of shape ``L``.
        """
        last_pos = self._get_episode_end_pos(self.circular(pos), env_ids)
        return last_pos - pos

    def get_episode_begin_position(self, pos, env_ids):
        """
        Note that the episode begin may not still be in the replay buffer.
        """
        indices = (env_ids, self.circular(pos))
        first_step_pos = self._indexed_pos[indices]
        buffer_step_types = alf.nest.get_field(self._buffer,
                                               self._step_type_field)
        is_first = buffer_step_types[indices] == ds.StepType.FIRST
        first_step_pos[is_first] = pos[is_first]
        return first_step_pos

    @atomic
    def gather_all(self):
        """Returns all the items in the buffer.

        Returns:
            Tensors of shape [B, T, ...], B=num_environments, T=current_size
        Raises:
            AssertionError: if the current_size is not same for all the
            environments.
        """
        size = self._current_size.min()
        max_size = self._current_size.max()
        assert size == max_size, (
            "Not all environments have the same size. min_size: %s "
            "max_size: %s" % (size, max_size))
        if size < self._max_length:
            pos = self._current_pos.min()
            max_pos = self._current_pos.max()
            assert pos == max_pos, (
                "Not all environments have the same ending position. "
                "min_pos: %s max_pos: %s" % (pos, max_pos))
            assert size == pos, (
                "When buffer not full, ending position of the data in the "
                "buffer current_pos coincides with current_size")

        # NOTE: this is not the proper way to gather all from a ring
        # buffer whose data can start from the middle, so this is limited
        # to the case where clear() is the only way to remove data from
        # the buffer.
        if size == self._max_length:
            result = self._buffer
        else:
            # Assumes that non-full buffer always stores data starting from 0
            result = alf.nest.map_structure(lambda buf: buf[:, :size, ...],
                                            self._buffer)
        return convert_device(result)

    def dequeue(self, env_ids=None):
        raise NotImplementedError(
            "gather needs to be modified to support" +
            " dequeue. Also need to update episode beginning indices if any" +
            " gets removed.")

    def get_field(self, field_name, env_ids, positions):
        """Get stored data of field from the replay buffer by ``env_ids`` and ``positions``.

        Args:
            field_name (str): indicate the path to the field with '.' separating
                the field name at different level
            env_ids (Tensor): 1-D int64 Tensor.
            positions (Tensor): 1-D int64 Tensor with same shape as ``env_ids``.
                These positions should be obtained from the BatchInfo returned
                by ``get_batch()``.
        Returns:
            Tensor: with the same shape as broadcasted shape of env_ids and positions
        """
        current_pos = self._current_pos[env_ids]
        assert torch.all(positions < current_pos), "Invalid positions"
        assert torch.all(
            positions >= current_pos - self._max_length), "Invalid positions"
        field = alf.nest.get_field(self._buffer, field_name)
        indices = (env_ids, self.circular(positions))
        result = alf.nest.map_structure(lambda x: x[indices], field)
        return convert_device(result)

    @property
    def total_size(self):
        """Total size from all environments."""
        return convert_device(self._current_size.sum())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
        if not self._step_type_field:
            return
        env_ids = torch.arange(self._num_envs)
        positions = self._current_pos - 1
        valid = positions >= 0
        env_ids = env_ids[valid]
        positions = positions[valid]
        step_type = alf.nest.get_field(self._buffer, self._step_type_field)
        step_type[env_ids, self.circular(positions)] = int(ds.StepType.LAST)


@gin.configurable
def l2_dist_close_reward_fn(achieved_goal, goal, threshold=.05, device="cpu"):
    if goal.dim() == 2:  # when goals are 1-dimentional
        assert achieved_goal.dim() == goal.dim()
        achieved_goal = achieved_goal.unsqueeze(2)
        goal = goal.unsqueeze(2)
    return torch.where(
        torch.norm(achieved_goal - goal, dim=2) < threshold,
        torch.zeros(1, dtype=torch.float32, device=device),
        -torch.ones(1, dtype=torch.float32, device=device))


@gin.configurable
def hindsight_relabel_fn(buffer,
                         result,
                         info,
                         her_proportion,
                         achieved_goal_field="observation.achieved_goal",
                         desired_goal_field="observation.desired_goal",
                         reward_fn=l2_dist_close_reward_fn):
    """Randomly get `batch_size` hindsight relabeled trajectories.

    Note: The environments where the sampels are from are ordered in the
        returned batch.

    Args:
        buffer (ReplayBuffer): for access to future achieved goals.
        result (nest): of tensors of the sampled exp
        info (BatchInfo): of the sampled result
        her_proportion (float): proportion of hindsight relabeled experience.
        achieved_goal_field (str): path to the achieved_goal field in
            exp nest.
        desired_goal_field (str): path to the desired_goal field in the
            exp nest.
        reward_fn (Callable): function to recompute reward based on
            achieve_goal and desired_goal.  Default gives reward 0 when
            L2 distance less than 0.05 and -1 otherwise, same as is done in
            suite_robotics environments.
    Returns:
        tuple:
            - nested Tensors: The samples. Its shapes are [batch_size, batch_length, ...]
            - BatchInfo: Information about the batch. Its shapes are [batch_size].
                - env_ids: environment id for each sequence
                - positions: starting position in the replay buffer for each sequence.
                - importance_weights: priority divided by the average of all
                    non-zero priorities in the buffer.
    """
    if her_proportion == 0:
        return result

    env_ids = info.env_ids
    start_pos = info.positions
    shape = alf.nest.get_nest_shape(result)
    batch_size, batch_length = shape[:2]
    # TODO: add support for batch_length > 2.
    assert batch_length == 2, shape

    # relabel only these sampled indices
    her_cond = torch.rand(batch_size) < her_proportion
    (her_indices, ) = torch.where(her_cond)
    (non_her_indices, ) = torch.where(torch.logical_not(her_cond))

    last_step_pos = start_pos[her_indices] + batch_length - 1
    last_env_ids = env_ids[her_indices]
    # Get x, y indices of LAST steps
    dist = buffer.steps_to_episode_end(last_step_pos, last_env_ids)
    if alf.summary.should_record_summaries():
        alf.summary.scalar(
            "replayer/" + buffer._name + ".mean_steps_to_episode_end",
            torch.mean(dist.type(torch.float32)))

    # get random future state
    future_idx = buffer.circular(last_step_pos + (torch.rand(*dist.shape) *
                                                  (dist + 1)).to(torch.int64))
    achieved_goals = alf.nest.get_field(buffer._buffer, achieved_goal_field)
    future_ag = achieved_goals[(last_env_ids, future_idx)].unsqueeze(1)

    # relabel desired goal
    result_desired_goal = alf.nest.get_field(result, desired_goal_field)
    relabed_goal = result_desired_goal.clone()
    her_batch_index_tuple = (her_indices.unsqueeze(1),
                             torch.arange(batch_length).unsqueeze(0))
    relabed_goal[her_batch_index_tuple] = future_ag

    # recompute rewards
    result_ag = alf.nest.get_field(result, achieved_goal_field)
    relabeled_rewards = reward_fn(
        result_ag, relabed_goal, device=buffer._device)
    if alf.summary.should_record_summaries():
        alf.summary.scalar(
            "replayer/" + buffer._name + ".reward_mean_before_relabel",
            torch.mean(result.reward[her_indices][:-1]))
        alf.summary.scalar(
            "replayer/" + buffer._name + ".reward_mean_after_relabel",
            torch.mean(relabeled_rewards[her_indices][:-1]))
    # assert reward function is the same as used by the environment.
    if not torch.allclose(relabeled_rewards[non_her_indices],
                          result.reward[non_her_indices]):
        msg = ("hindsight_relabel_fn:\nrelabeled_reward\n{}\n!=\n" +
               "env_reward\n{}\nag:\n{}\ndg:\n{}\nenv_ids:\n{}\nstart_pos:\n{}"
               ).format(relabeled_rewards[non_her_indices],
                        result.reward[non_her_indices],
                        result_ag[non_her_indices],
                        result_desired_goal[non_her_indices],
                        env_ids[non_her_indices], start_pos[non_her_indices])
        logging.warning(msg)
        # assert False, msg
        relabeled_rewards[non_her_indices] = result.reward[non_her_indices]

    result = alf.nest.transform_nest(
        result, desired_goal_field, lambda _: relabed_goal)
    result = result._replace(reward=relabeled_rewards)
    return result, info
