# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""PredictiveRepresentationLearner."""

import gin
import torch

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, TimeStep, Experience, LossInfo, namedtuple
from alf.experience_replayers.replay_buffer import BatchInfo, ReplayBuffer
from alf.nest import nest
from alf.nest.utils import convert_device
from alf.networks import Network, LSTMEncodingNetwork
from alf.utils import dist_utils, spec_utils, tensor_utils
from alf.utils.normalizers import AdaptiveNormalizer
from alf.utils.summary_utils import safe_mean_hist_summary, safe_mean_summary

PredictiveRepresentationLearnerInfo = namedtuple(
    'PredictiveRepresentationLearnerInfo',
    [
        # actual actions taken in the next unroll_steps + 1 steps
        # [B, unroll_steps + 1, ...]
        'action',

        # The flag to indicate whether to include this target into loss
        # [B, unroll_steps + 1]
        'mask',

        # nest for targets
        # [B, unroll_steps + 1, ...]
        'target'
    ])


@gin.configurable
class SimpleDecoder(Algorithm):
    """A simple decoder with elementwise loss between the target and the predicted value.

    It is used to predict the target value from the given representation. Its
    loss can be used to train the representation.
    """

    def __init__(self,
                 input_tensor_spec,
                 target_field,
                 decoder_net_ctor,
                 loss=torch.nn.SmoothL1Loss(reduction='none'),
                 loss_weight=1.0,
                 summarize_each_dimension=False,
                 optimizer=None,
                 normalize_target=False,
                 debug_summaries=False,
                 name="SimpleDecoder"):
        """
        Args:
            input_tensor_spec (TensorSpec): describing the input tensor.
            target_field (str): name of the field in the experience to be used
                as the decoding target.
            decoder_net_ctor (Callable): called as ``decoder_net_ctor(input_tensor_spec=input_tensor_spec)``
                to construct an instance of ``Network`` for decoding. The network
                should take the latent representation as input and output the
                predicted value of the target.
            loss (Callable): loss function with signature ``loss(y_pred, y_true)``.
                Note that it should not reduce to a scalar. It should at least
                keep the batch dimension in the returned loss.
            loss_weight (float): weight for the loss.
            optimizer (Optimzer|None): if provided, it will be used to optimize
                the parameter of decoder_net
            normalize_target (bool): whether to normalize target.
                Note that the effect of this is to change the loss. The predicted
                value itself is not normalized.
            debug_summaries (bool): whether to generate debug summaries
            name (str): name of this instance
        """
        super().__init__(
            optimizer=optimizer, debug_summaries=debug_summaries, name=name)
        self._decoder_net = decoder_net_ctor(
            input_tensor_spec=input_tensor_spec)
        assert self._decoder_net.state_spec == (
        ), "RNN decoder is not suppported"
        self._summarize_each_dimension = summarize_each_dimension
        self._target_field = target_field
        self._loss = loss
        self._loss_weight = loss_weight
        if normalize_target:
            self._target_normalizer = AdaptiveNormalizer(
                self._decoder_net.output_spec,
                auto_update=False,
                name=name + ".target_normalizer")
        else:
            self._target_normalizer = None

    def get_target_fields(self):
        return self._target_field

    def train_step(self, repr, state=()):
        predicted_target = self._decoder_net(repr)[0]
        return AlgStep(
            output=predicted_target, state=state, info=predicted_target)

    def predict_step(self, repr, state=()):
        predicted_target = self._decoder_net(repr)[0]
        return AlgStep(
            output=predicted_target, state=state, info=predicted_target)

    def calc_loss(self, target, predicted, mask=None):
        """Calculate the loss between ``target`` and ``predicted``.

        Args:
            target (Tensor): target to be predicted. Its shape is [T, B, ...]
            predicted (Tensor): predicted target. Its shape is [T, B, ...]
            mask (bool Tensor): indicating which target should be predicted.
                Its shape is [T, B].
        Returns:
            LossInfo
        """
        if self._target_normalizer:
            self._target_normalizer.update(target)
            target = self._target_normalizer.normalize(target)
            predicted = self._target_normalizer.normalize(predicted)

        loss = self._loss(predicted, target)
        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):

                def _summarize1(pred, tgt, loss, mask, suffix):
                    alf.summary.scalar(
                        "explained_variance" + suffix,
                        tensor_utils.explained_variance(pred, tgt, mask))
                    safe_mean_hist_summary('predict' + suffix, pred, mask)
                    safe_mean_hist_summary('target' + suffix, tgt, mask)
                    safe_mean_summary("loss" + suffix, loss, mask)

                def _summarize(pred, tgt, loss, mask, suffix):
                    _summarize1(pred[0], tgt[0], loss[0], mask[0],
                                suffix + "/current")
                    if pred.shape[0] > 1:
                        _summarize1(pred[1:], tgt[1:], loss[1:], mask[1:],
                                    suffix + "/future")

                if loss.ndim == 2:
                    _summarize(predicted, target, loss, mask, '')
                elif not self._summarize_each_dimension:
                    m = mask
                    if m is not None:
                        m = m.unsqueeze(-1).expand_as(predicted)
                    _summarize(predicted, target, loss, m, '')
                else:
                    for i in range(predicted.shape[2]):
                        suffix = '/' + str(i)
                        _summarize(predicted[..., i], target[..., i],
                                   loss[..., i], mask, suffix)

        if loss.ndim == 3:
            loss = loss.mean(dim=2)

        if mask is not None:
            loss = loss * mask

        return LossInfo(loss=loss * self._loss_weight, extra=loss)


@gin.configurable
class PredictiveRepresentationLearner(Algorithm):
    """Learn representation based on the prediction of future values.

    ``PredictiveRepresentationLearner`` contains 3 ``Module``s:

    * encoding_net: it is a ``Network`` that encodes the raw observation to a
      latent vector.
    * dynamics_net: it is a ``Network`` that generates the future latent states
      from the current latent state.
    * decoder: it is an ``Algorithm`` that decode the target values from the
      latent state and calcuate the loss.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 num_unroll_steps,
                 decoder_ctor,
                 encoding_net_ctor,
                 dynamics_net_ctor,
                 encoding_optimizer=None,
                 dynamics_optimizer=None,
                 debug_summaries=False,
                 name="PredictiveRepresentationLearner"):
        """
        Args:
            observation_spec (nested TensorSpec): describing the observation.
            action_spec (nested BoundedTensorSpec): describing the action.
            num_unroll_steps (int): the number of future steps to predict.
            decoder_ctor (Callable): called as ``decoder_ctor(observation)`` to
                construct the decoder algorithm. It should follow the ``Algorithm``
                interface. In addition to the interface of ``Algorithm``, it should
                also implement a member function ``get_target_fields()``, which
                returns a nest of the names of target fields. See ``SimpleDecoder``
                for an example of decoder.
            encoding_net_ctor (Callable): called as ``encoding_net_ctor(observation_spec)``
                to construct the encoding ``Network``. The network takes raw observation
                as input and output the latent representation. encoding_net can
                be an RNN.
            dynamics_net_ctor (Callable): called as ``dynamics_net_ctor(action_spec)``
                to construct the dynamics ``Network``. It must be an RNN. The
                constructed network takes action as input and outputs the future
                latent representation. If the state_spec of the dynamics net is
                exactly same as the state_spec of the encoding net, the current
                state of the encoding net will be used as the initial state of the
                dynamics net. Otherwise, a linear projection will be used to
                convert the current latent represenation to the initial state for
                the dynamics net.
            encoding_optimizer (Optimizer|None): if provided, will be used to optimize
                the parameter for the encoding net.
            dynamics_optimizer (Optimizer|None): if provided, will be used to optimize
                the parameter for the dynamics net.
            debug_summaries (bool): whether to generate debug summaries
            name (str): name of this instance.
        """
        encoding_net = encoding_net_ctor(observation_spec)
        super().__init__(
            train_state_spec=encoding_net.state_spec,
            debug_summaries=debug_summaries,
            name=name)

        self._encoding_net = encoding_net
        if encoding_optimizer is not None:
            self.add_optimizer(encoding_optimizer, [self._encoding_net])
        repr_spec = self._encoding_net.output_spec
        self._decoder = decoder_ctor(
            repr_spec, debug_summaries=debug_summaries, name=name + ".decoder")
        assert len(alf.nest.flatten(self._decoder.train_state_spec)) == 0, (
            "RNN decoder is not suported")
        self._num_unroll_steps = num_unroll_steps
        self._target_fields = self._decoder.get_target_fields()
        self._output_spec = repr_spec

        if num_unroll_steps > 0:
            self._dynamics_net = dynamics_net_ctor(action_spec)
            self._dynamics_state_dims = alf.nest.map_structure(
                lambda spec: spec.numel,
                alf.nest.flatten(self._dynamics_net.state_spec))
            assert sum(
                self._dynamics_state_dims) > 0, ("dynamics_net should be RNN")
            compatible_state = True

            try:
                alf.nest.assert_same_structure(self._dynamics_net.state_spec,
                                               self._encoding_net.state_spec)
                compatible_state = all(
                    alf.nest.flatten(
                        alf.nest.map_structure(lambda s1, s2: s1 == s2,
                                               self._dynamics_net.state_spec,
                                               self._encoding_net.state_spec)))
            except Exception:
                compatible_state = False

            self._latent_to_dstate_fc = None
            modules = [self._dynamics_net]
            if not compatible_state:
                self._latent_to_dstate_fc = alf.layers.FC(
                    repr_spec.numel, sum(self._dynamics_state_dims))
                modules.append(self._latent_to_dstate_fc)

            if dynamics_optimizer is not None:
                self.add_optimizer(dynamics_optimizer, modules)

    @property
    def output_spec(self):
        return self._output_spec

    def predict_step(self, time_step: TimeStep, state):
        latent, state = self._encoding_net(time_step.observation, state)
        return AlgStep(output=latent, state=state)

    def rollout_step(self, time_step: TimeStep, state):
        latent, state = self._encoding_net(time_step.observation, state)
        return AlgStep(output=latent, state=state)

    def predict_multi_step(self, init_latent, actions, state=None):
        """Perform multi-step predictions based on the initial latent
            representation and actions sequences.
        Args:
            init_latent (Tensor): the latent representation for the initial
                step of the prediction
            actions (Tensor): [B, unroll_steps, action_dim]
            state:
        Returns:
            prediction (Tensor): predicted target of shape
                [B, unroll_steps + 1, d], where d is the dimension of
                the predicted target.
        """
        num_unroll_steps = actions.shape[1]

        assert num_unroll_steps > 0

        sim_latent = self._multi_step_latent_rollout(
            init_latent, num_unroll_steps, actions, state)

        prediction = self._decoder.predict_step(sim_latent).info
        return prediction

    def _multi_step_latent_rollout(self, init_latent, num_unroll_steps,
                                   actions, state):
        """Perform multi-step latent rollout based on the initial latent
            representation and action sequences.
        Args:
            init_latent (Tensor): the latent representation for the initial
                step of the prediction
            actions (Tensor): [B, unroll_steps, action_dim]
            state:
        Returns:
            sim_latent (Tensor): a tensor of the shape [(unroll_steps+1)*B, ...],
                obtained by concataning all the latent states during rollout,
                including the input initial latent represenataion
        """

        sim_latents = [init_latent]

        if num_unroll_steps > 0:
            if self._latent_to_dstate_fc is not None:
                dstate = self._latent_to_dstate_fc(init_latent)
                dstate = dstate.split(self._dynamics_state_dims, dim=1)
                dstate = alf.nest.pack_sequence_as(
                    self._dynamics_net.state_spec, dstate)
            else:
                dstate = state

        for i in range(self._num_unroll_steps):
            sim_latent, dstate = self._dynamics_net(actions[:, i, ...], dstate)
            sim_latents.append(sim_latent)

        sim_latent = torch.cat(sim_latents, dim=0)
        return sim_latent

    def train_step(self, exp: TimeStep, state):
        # [B, num_unroll_steps + 1]
        info = exp.rollout_info
        batch_size = exp.step_type.shape[0]
        latent, state = self._encoding_net(exp.observation, state)

        sim_latent = self._multi_step_latent_rollout(
            latent, self._num_unroll_steps, info.action, state)

        # [num_unroll_steps + 1)*B, ...]
        train_info = self._decoder.train_step(sim_latent).info
        train_info_spec = dist_utils.extract_spec(train_info)
        train_info = dist_utils.distributions_to_params(train_info)
        train_info = alf.nest.map_structure(
            lambda x: x.reshape(self._num_unroll_steps + 1, batch_size, *x.
                                shape[1:]), train_info)
        # [num_unroll_steps + 1, B, ...]
        train_info = dist_utils.params_to_distributions(
            train_info, train_info_spec)
        target = alf.nest.map_structure(lambda x: x.transpose(0, 1),
                                        info.target)
        loss_info = self._decoder.calc_loss(target, train_info, info.mask.t())
        loss_info = alf.nest.map_structure(lambda x: x.mean(dim=0), loss_info)

        return AlgStep(output=latent, state=state, info=loss_info)

    @torch.no_grad()
    def preprocess_experience(self, experience: Experience):
        """Fill experience.rollout_info with PredictiveRepresentationLearnerInfo

        Note that the shape of experience is [B, T, ...]
        """
        assert experience.batch_info != ()
        batch_info: BatchInfo = experience.batch_info
        replay_buffer: ReplayBuffer = experience.replay_buffer
        mini_batch_length = experience.step_type.shape[1]

        with alf.device(replay_buffer.device):
            # [B, 1]
            positions = convert_device(batch_info.positions).unsqueeze(-1)
            # [B, 1]
            env_ids = convert_device(batch_info.env_ids).unsqueeze(-1)

            # [B, T]
            positions = positions + torch.arange(mini_batch_length)

            # [B, T]
            steps_to_episode_end = replay_buffer.steps_to_episode_end(
                positions, env_ids)
            # [B, T]
            episode_end_positions = positions + steps_to_episode_end

            # [B, T, unroll_steps+1]
            positions = positions.unsqueeze(-1) + torch.arange(
                self._num_unroll_steps + 1)
            # [B, 1, 1]
            env_ids = env_ids.unsqueeze(-1)
            # [B, T, 1]
            episode_end_positions = episode_end_positions.unsqueeze(-1)

            # [B, T, unroll_steps+1]
            mask = positions <= episode_end_positions

            # [B, T, unroll_steps+1]
            positions = torch.min(positions, episode_end_positions)

            # [B, T, unroll_steps+1, ...]
            target = replay_buffer.get_field(self._target_fields, env_ids,
                                             positions)

            # [B, T, unroll_steps+1]
            action = replay_buffer.get_field('action', env_ids, positions)

            rollout_info = PredictiveRepresentationLearnerInfo(
                action=action, mask=mask, target=target)

        rollout_info = convert_device(rollout_info)

        return experience._replace(rollout_info=rollout_info)
