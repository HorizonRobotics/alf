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

import gin
import torch
import torch.nn as nn

import alf
from alf import data_structures as ds
from alf.data_structures import namedtuple
from alf.nest.utils import convert_device
from alf.utils.common import warning_once
from alf.utils.data_buffer import atomic, RingBuffer

from .segment_tree import SumSegmentTree, MaxSegmentTree, MinSegmentTree

BatchInfo = namedtuple(
    "BatchInfo", ["env_ids", "index", "importance_weights"], default_value=())


@gin.configurable
class ReplayBuffer(RingBuffer):
    """Replay buffer with RingBuffer as implementation.

    Terminology: consistent with RingBuffer, we use ``pos`` to refer to the
    always increasing position of an element in the infinitly long buffer,
    and ``idx`` as the actual index of the element in the underlying store
    (``_buffer``).  That means ``idx == pos % _max_length`` is always true,
    and one should use ``_buffer[idx]`` to retrieve the stored data.
    """

    def __init__(self,
                 data_spec,
                 num_environments,
                 max_length=1024,
                 prioritized_sampling=False,
                 initial_priority=1.0,
                 device="cpu",
                 allow_multiprocess=False,
                 her_k=0,
                 step_type_field="step_type",
                 achieved_goal_field="observation.achieved_goal",
                 desired_goal_field="observation.desired_goal",
                 reward_field="reward",
                 reward_fn=None,
                 name="ReplayBuffer"):
        """
        Args:
            data_spec (alf.TensorSpec): spec of an entry nest in the buffer.
            num_environments (int): total number of parallel environments
                stored in the buffer.
            max_length (int): maximum number of time steps stored in buffer.
            prioritized_sampling (bool): Use prioritized sampling if this is True.
            initial_priority (float): initial priority used for new experiences.
                The actual initial priority used for new experience is the maximum
                of this value and the current maximum priority of all experiences.
            device (string): "cpu" or "cuda" where tensors are created.
            allow_multiprocess (bool): whether multiprocessing is supported.
            her_k (float): proportion of hindsight relabeled experience.
            step_type_field (string): path to the step_type field in exp nest.
                This and the following fields are for hindsight relabeling.
            achieved_goal_field (string): path to the achieved_goal field in
                exp nest.
            desired_goal_field (string): path to the desired_goal field in the
                exp nest.
            reward_field (string): path to the reward field in the exp nest.
                are for hindsight experience replay.
            reward_fn (callable): function to recompute reward based on
                achieve_goal and desired_goal.  Default gives reward 0 when
                L2 distance less than 0.05 and -1 otherwise, same as is done in
                suite_robotics environments.
            name (string): name of the replay buffer object.
        """
        super().__init__(
            data_spec,
            num_environments,
            max_length=max_length,
            device=device,
            allow_multiprocess=allow_multiprocess,
            name=name)
        self._step_type_field = step_type_field
        self._achieved_goal_field = achieved_goal_field
        self._desired_goal_field = desired_goal_field
        self._reward_field = reward_field
        self._her_k = her_k
        self._prioritized_sampling = prioritized_sampling
        if prioritized_sampling:
            self._mini_batch_length = 1
            tree_size = self._max_length * num_environments
            self._sum_tree = SumSegmentTree(tree_size, device=device)
            self._max_tree = MaxSegmentTree(tree_size, device=device)
            self._initial_priority = torch.tensor(
                initial_priority, dtype=torch.float32, device=device)
        if self._her_k > 0:
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
        if reward_fn:
            self._reward_fn = reward_fn
        else:

            def default_reward_fn(achieved_goal, goal):
                if goal.dim() == 2:  # when goals are 1-dimentional
                    assert achieved_goal.dim() == goal.dim()
                    achieved_goal = achieved_goal.unsqueeze(2)
                    goal = goal.unsqueeze(2)
                return torch.where(
                    torch.norm(achieved_goal - goal, dim=2) < .05,
                    torch.zeros(1), -torch.ones(1))

            self._reward_fn = default_reward_fn

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

    # This function needs to be called in the same atomic transaction as
    # ``prioritized_sample()`` for the idx sampled to be still valid
    # in multiprocessing/asynchronous cases.
    def update_priority(self, env_ids, idx, priorities):
        """Update the priorities for the given experiences.

        Args:
            env_ids (Tensor): 1-D int64 Tensor.
            idx (Tensor): 1-D int64 Tensor with same shape as ``env_ids``.
                This idx should be obtained the BatchInfo returned by
                ``get_batch()``
        """
        indices = self._env_id_idx_to_index(env_ids, idx)
        self._update_segment_tree(indices, priorities)

    @atomic
    def add_batch(self, batch, env_ids=None, blocking=False):
        """adds a batch of entries to buffer updating indices as needed.

        We build an index of episode beginning indices for each element
        in the buffer.  The beginning point stores where episode end is.
        
        """
        with alf.device(self._device):
            env_ids = self.check_convert_env_ids(env_ids)
            if self._her_k > 0:
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

            if self._her_k > 0:
                # 3. Update associated episode end indices
                # 3.1. find ending steps in batch (incl. MID and LAST steps)
                step_types = alf.nest.get_field(batch, self._step_type_field)
                non_first, = torch.where(step_types != ds.StepType.FIRST)
                # 3.2. update episode ending positions
                self.store_episode_end_pos(non_first, overwriting_pos, env_ids)
                # 3.3. initialize episode beginning positions to itself
                epi_first, = torch.where(step_types == ds.StepType.FIRST)
                self._indexed_pos[(env_ids[epi_first],
                                   self.circular(overwriting_pos[epi_first])
                                   )] = overwriting_pos[epi_first]

    @atomic
    def get_batch(self, batch_size, batch_length):
        """Randomly get ``batch_size`` trajectories from the buffer.

        It hindsight relabels the experience when ReplayBuffer.her_k > 0.

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
                    - importance_weights: importance weight divided by the average of
                        all non-zero importance weights in the buffer.
        """
        with alf.device(self._device):
            if self._prioritized_sampling:
                env_ids, idx = self._prioritized_sample(
                    batch_size, batch_length)
            else:
                env_ids, idx = self._uniform_sample(batch_size, batch_length)

            info = BatchInfo(env_ids=env_ids, index=idx)

            idx = idx.reshape(-1, 1)  # [B, 1]
            idx = self.circular(
                idx + torch.arange(batch_length).unsqueeze(0))  # [B, T]
            env_ids = env_ids.reshape(-1, 1).expand(batch_size,
                                                    batch_length)  # [B, T]
            result = alf.nest.map_structure(lambda b: b[(env_ids, idx)],
                                            self._buffer)

            if self._prioritized_sampling:
                indices = self._env_id_idx_to_index(info.env_ids, info.index)
                avg_weight = self._sum_tree.nnz / self._sum_tree.summary()
                info = info._replace(
                    importance_weights=self._sum_tree[indices] * avg_weight)

            if self._her_k > 0:
                result = self._hindsight_relabel(batch_size, batch_length,
                                                 result, env_ids, idx)

        return convert_device(result), convert_device(info)

    def _uniform_sample(self, batch_size, batch_length):
        min_size = self._current_size.min()
        assert min_size >= batch_length, (
            "Not all environments have enough data. The smallest data "
            "size is: %s Try storing more data before calling get_batch" %
            min_size)

        batch_size_per_env = batch_size // self._num_envs
        remaining = batch_size % self._num_envs
        if batch_size_per_env > 0:
            env_ids = torch.arange(self._num_envs).repeat(batch_size_per_env)
        else:
            env_ids = torch.zeros(0, dtype=torch.int64)
        if remaining > 0:
            eids = torch.randperm(self._num_envs)[:remaining]
            env_ids = torch.cat([env_ids, eids], dim=0)

        r = torch.rand(*env_ids.shape)

        num_positions = self._current_size - batch_length + 1
        num_positions = num_positions[env_ids]
        pos = (r * num_positions).to(torch.int64)
        pos += (self._current_pos - self._current_size)[env_ids]
        return env_ids, self.circular(pos)

    def _prioritized_sample(self, batch_size, batch_length):
        if batch_length != self._mini_batch_length:
            if self._mini_batch_length > 1:
                warning_once(
                    "It is not advisable to use different batch_length"
                    "for different calls to get_batch(). Previous batch_length=%d "
                    "new batch_length=%d" % (self._mini_batch_length,
                                             batch_length))
            self._change_mini_batch_length(batch_length)

        assert self._sum_tree.summary() > 0, (
            "There is no data in the "
            "buffer or the data of all the environments are shorter than "
            "batch_length=%s" % batch_length)

        r = torch.rand((batch_size, ))
        r = (r + torch.arange(batch_size, dtype=torch.float32)) / batch_size
        r = r * self._sum_tree.summary()
        indices = self._sum_tree.find_sum_bound(r)
        env_ids, idx = self._index_to_env_id_idx(indices)
        return env_ids, idx

    def _pad(self, x, env_ids):
        """Make ``x`` (any index) absolute positions in the RingBuffer.

        This is the reverse of ``circular()``, and is useful when trying to
        compute the distance from one index to the other.  This is done by
        adding multiples of _max_length back to the index.

        NOTE, this operation depends on the _current_pos of the RingBuffer,
        and can generate different results if new data are added to the buffer.
        """
        multiples = (
            self._current_pos[env_ids] / self._max_length) * self._max_length
        idx = self.circular(x)
        return torch.where(
            idx < self.circular(self._current_pos[env_ids] -
                                self._current_size[env_ids]), idx + multiples,
            idx + multiples - self._max_length)

    def store_episode_end_pos(self, non_first, pos, env_ids):
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
        prev_pos = _pos - 1
        prev_idx = self.circular(prev_pos)
        prev_first = self._indexed_pos[(_env_ids, prev_idx)]
        prev_first_idx = self.circular(prev_first)
        # record pos of ``FIRST`` step into the current position of _index
        self._indexed_pos[(_env_ids, self.circular(_pos))] = prev_first

        # update episode end for the ``FIRST`` step of the episode
        has_head_cond = prev_first <= prev_pos
        has_head, = torch.where(has_head_cond)
        self._indexed_pos[(_env_ids[has_head],
                           prev_first_idx[has_head])] = _pos[has_head]
        headless, = torch.where(torch.logical_not(has_head_cond))
        self._headless_indexed_pos[_env_ids[headless]] = _pos[headless]

    def get_episode_end_pos(self, idx, env_ids):
        """Use _indexed_pos and _headless_indexed_pos to get last step pos.

        NOTE, the positions returned are the stored positions, are
        possibly outdated, and need to be padded with self._pad().
        """
        # Look up ``FIRST`` step position, then look up the stored last pos
        first_step_pos = self._indexed_pos[(env_ids, idx)]
        first_step_idx = self.circular(first_step_pos)
        result = self._indexed_pos[(env_ids, first_step_idx)].clone()
        # If current step is FIRST, skip second lookup.
        buffer_step_types = alf.nest.get_field(self._buffer,
                                               self._step_type_field)
        is_first_cond = buffer_step_types[(env_ids, idx)] == ds.StepType.FIRST
        is_first, = torch.where(is_first_cond)
        result[env_ids[is_first]] = first_step_idx[is_first]
        # Special handling for headless timesteps whose ``FIRST`` steps
        # were recently overwritten in the RingBuffer.
        new_pos = self._pad(idx, env_ids)
        new_first_step_pos = self._pad(first_step_pos, env_ids)
        headless, = torch.where(
            (new_first_step_pos > new_pos) * torch.logical_not(is_first_cond))
        headless_env_ids = env_ids[headless]
        result[headless] = self._headless_indexed_pos[headless_env_ids]
        return result

    def distance_to_episode_end(self, idx, env_ids):
        """Get the distance to the closest episode end in future.

        Args:
            idx (tensor): shape ``L``, index of the current timesteps in
                the replay buffer.
            env_ids (tensor): shape ``L``
        Returns:
            tensor of shape ``L``.
        """
        pos = self._pad(idx, env_ids)
        last_pos = self.get_episode_end_pos(idx, env_ids)
        # RingBuffer probably changed between when the data was stored and now.
        # We need to pad the retrieved episode ending positions again, which
        # involves making last_pos circular and then adding back multiples of
        # _max_length.
        current_last_pos = self._pad(last_pos, env_ids)
        return current_last_pos - pos

    def _hindsight_relabel(self, batch_size, batch_length, result, env_ids,
                           idx):
        """Randomly get `batch_size` hindsight relabeled trajectories.

        Note: The environments where the sampels are from are ordered in the
            returned batch.

        Args:
            batch_size (int): get so many trajectories
            batch_length (int): the length of each trajectory
            result (nest): of tensors of the sampled exp
            env_ids (tensor): env_id indices of the sampled data, of shape
                [batch_size, batch_length]
            idx (tensor): indices of the sampled data, of shape
                [batch_size, batch_length]
        Returns:
            nested Tensors. The shapes are [batch_size, batch_length, ...]
        """
        if self._her_k == 0:
            return result

        # TODO: add support for batch_length > 2.
        assert batch_length == 2

        # relabel only these sampled indices
        (her_indices, ) = torch.where(torch.rand(batch_size) < self._her_k)

        batch_last_step_idx = idx[:, -1][her_indices]
        env_ids = env_ids[:, -1]
        # Get x, y indices of LAST steps
        dist = self.distance_to_episode_end(batch_last_step_idx,
                                            env_ids[her_indices])

        # get random future state
        future_idx = self.circular(batch_last_step_idx + (
            torch.rand(*dist.shape) * (dist + 1)).to(torch.int64))
        achieved_goals = alf.nest.get_field(self._buffer,
                                            self._achieved_goal_field)
        future_ag = achieved_goals[(env_ids[her_indices],
                                    future_idx)].unsqueeze(1)

        # relabel desired goal
        result_desired_goal = alf.nest.get_field(
            result, self._desired_goal_field).clone()
        result_desired_goal[(
            her_indices.unsqueeze(1),
            torch.arange(batch_length).unsqueeze(0))] = future_ag

        # recompute rewards
        result_reward = alf.nest.get_field(result, self._reward_field).clone()
        result_ag = alf.nest.get_field(result, self._achieved_goal_field)
        relabeled_rewards = self._reward_fn(
            result_ag[(her_indices.unsqueeze(1),
                       torch.arange(batch_length).unsqueeze(0))],
            result_desired_goal[(her_indices.unsqueeze(1),
                                 torch.arange(batch_length).unsqueeze(0))])
        result_reward[(
            her_indices.unsqueeze(1),
            torch.arange(batch_length).unsqueeze(0))] = relabeled_rewards

        result = alf.nest.utils.transform_nest(
            result, self._desired_goal_field, lambda _: result_desired_goal)
        result = alf.nest.utils.transform_nest(
            result, self._reward_field, lambda _: result_reward)
        return result

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

    @property
    def total_size(self):
        """Total size from all environments."""
        return convert_device(self._current_size.sum())
