# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Base class for off policy algorithms."""

import abc
from collections import namedtuple
from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import StepType
from tf_agents.specs.distribution_spec import DistributionSpec
from tf_agents.specs.distribution_spec import nested_distributions_from_specs

from alf.algorithms.rl_algorithm import ActionTimeStep, RLAlgorithm
from alf.algorithms.rl_algorithm import make_training_info
from alf.experience_replayers.experience_replay import OnetimeExperienceReplayer
from alf.experience_replayers.experience_replay import SyncUniformExperienceReplayer
from alf.utils import common

Experience = namedtuple("Experience", [
    'step_type', 'reward', 'discount', 'observation', 'prev_action', 'action',
    'info', 'action_distribution', 'state'
])


def make_experience(time_step: ActionTimeStep, policy_step: PolicyStep,
                    action_distribution, state):
    """Make an instance of Experience from ActionTimeStep and PolicyStep."""
    return Experience(
        step_type=time_step.step_type,
        reward=time_step.reward,
        discount=time_step.discount,
        observation=time_step.observation,
        prev_action=time_step.prev_action,
        action=policy_step.action,
        info=policy_step.info,
        action_distribution=action_distribution,
        state=state)


class OffPolicyAlgorithm(RLAlgorithm):
    """
       OffPolicyAlgorithm works with alf.drivers.off_policy_driver to do training

       User needs to implement rollout() and train_step().

       rollout() is called to generate actions for every environment step.

       train_step() is called to generate necessary information for training.

       The following is the pseudo code to illustrate how OffPolicyAlgorithm is used
       with OffPolicyDriver:

       ```python
        # (1) collect stage
        for _ in range(steps_per_collection):
            # collect experience and store to replay buffer
            policy_step = rollout(time_step, policy_step.state)
            experience = make_experience(time_step, policy_step)
            store experience to replay buffer
            action = sample action from policy_step.action
            time_step = env.step(action)

        # (2) train stage
        for _ in range(training_per_collection):
            # sample experiences and perform training
            experiences = sample batch from replay_buffer
            with tf.GradientTape() as tape:
                batched_training_info = []
                for experience in experiences:
                    policy_step = train_step(experience, state)
                    train_info = make_training_info(info, ...)
                    write train_info to batched_training_info
                train_complete(tape, batched_training_info,...)
    ```
    """
    @property
    def exp_replayer(self):
        """Return experience replayer."""
        return self._exp_replayer

    def predict(self, time_step: ActionTimeStep, state=None):
        """Default implementation of predict.

        Subclass may override.
        """
        policy_step = self._rollout_partial_state(time_step, state)
        return policy_step._replace(info=())

    def rollout(self,
                time_step: ActionTimeStep,
                state=None,
                with_experience=False):
        """Base implementation of rollout for OffPolicyAlgorithm.

        Calls _rollout_full_state or _rollout_partial_state based on
        use_rollout_state.

        Subclass may override.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec
            with_experience (bool): a boolean flag indicating whether the current
                rollout is with sampled experiences or not. By default this flag
                is ignored. See ActorCriticAlgorithm's rollout for an example of
                usage to avoid computing intrinsic rewards if
                `with_experience=True`.
        Returns:
            policy_step (PolicyStep):
              action (nested tf.distribution): should be consistent with
                `action_distribution_spec`
              state (nested Tensor): should be consistent with `train_state_spec`
              info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OnPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        if self._use_rollout_state and self._is_rnn:
            return self._rollout_full_state(time_step, state)
        else:
            return self._rollout_partial_state(time_step, state)

    def _rollout_partial_state(self, time_step: ActionTimeStep, state=None):
        """Rollout without the full state for train_step().

        It is used for non-RNN model or RNN model without computating all states
        in train_state_spec. In the returned PolicyStep.state, you can use an
        empty tuple as a placeholder for those states that are not necessary for
        rollout.

        User needs to override this if _rollout_full_state() is not implemented.
        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec
        Returns:
            policy_step (PolicyStep):
              action (nested tf.distribution): should be consistent with
                `action_distribution_spec`
              state (nested Tensor): should be consistent with `train_state_spec`.
              info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OnPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        return self._rollout_full_state(time_step, state)

    def _rollout_full_state(self, time_step: ActionTimeStep, state=None):
        """Rollout with full state for train_step().

        If you want to use the rollout state for off-policy training (by setting
        TrainerConfig.use_rollout=True), you need to implement this function.
        You need to compute all the states for the returned PolicyStep.state.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec
        Returns:
            policy_step (PolicyStep):
              action (nested tf.distribution): should be consistent with
                `action_distribution_spec`
              state (nested Tensor): should be consistent with `train_state_spec`.
              info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OnPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        raise NotImplementedError("_rollout_full_state is not implemented")

    @abc.abstractmethod
    def train_step(self, experience: Experience, state):
        """Perform one step of training computation.

        Args:
            experience (Experience):
            state (nested Tensor): should be consistent with train_state_spec

        Returns (PolicyStep):
            action (nested tf.distribution): should be consistent with
                `action_distribution_spec`
            state (nested Tensor): should be consistent with `train_state_spec`
            info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OffPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        pass

    def preprocess_experience(self, experience: Experience):
        """Preprocess experience.

        preprocess_experience is called for the experiences got from replay
        buffer. An example is to calculate advantages and returns in PPOAlgorithm.

        The shapes of tensors in experience are assumed to be (B, T, ...)

        Args:
            experience (Experience): original experience
        Returns:
            processed experience
        """
        return experience

    def prepare_off_policy_specs(self,
                      env_batch_size,
                      time_step:ActionTimeStep,
                      exp_replayer: str,
                      metrics,
                      observation_transformer: Callable=None):
        """Prepare various tensor specs."""

        def extract_spec(nest):
            return tf.nest.map_structure(
                lambda t: tf.TensorSpec(t.shape[1:], t.dtype), nest)

        self._metrics = metrics
        self._observation_transformer = observation_transformer
        self._time_step_spec = extract_spec(time_step)
        initial_state = common.get_initial_policy_state(
            env_batch_size, self.train_state_spec)
        policy_step = self.rollout(time_step, initial_state)
        info_spec = extract_spec(policy_step.info)

        def _to_distribution_spec(spec):
            if isinstance(spec, tf.TensorSpec):
                return DistributionSpec(
                    tfp.distributions.Deterministic,
                    input_params_spec={"loc": spec},
                    sample_spec=spec)
            return spec

        self._action_distribution_spec = tf.nest.map_structure(
            _to_distribution_spec, self.action_distribution_spec)
        self._action_dist_param_spec = tf.nest.map_structure(
            lambda spec: spec.input_params_spec,
            self._action_distribution_spec)

        self._experience_spec = Experience(
            step_type=self._time_step_spec.step_type,
            reward=self._time_step_spec.reward,
            discount=self._time_step_spec.discount,
            observation=self._time_step_spec.observation,
            prev_action=self._action_spec,
            action=self._action_spec,
            info=info_spec,
            action_distribution=self._action_dist_param_spec,
            state=self.train_state_spec if self._use_rollout_state else
            ())

        if exp_replayer == "one_time":
            self._exp_replayer = OnetimeExperienceReplayer()
        elif exp_replayer == "uniform":
            self._exp_replayer = SyncUniformExperienceReplayer(
                self._experience_spec, env_batch_size)
        else:
            raise ValueError("invalid experience replayer name")
        self.add_experience_observer(self._exp_replayer.observe)

        action_dist_params = common.zero_tensor_from_nested_spec(
            self._experience_spec.action_distribution, env_batch_size)
        action_dist = nested_distributions_from_specs(
            self._action_distribution_spec, action_dist_params)

        exp = Experience(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=time_step.observation,
            prev_action=time_step.prev_action,
            action=time_step.prev_action,
            info=policy_step.info,
            action_distribution=action_dist,
            state=initial_state if self._use_rollout_state else ())

        processed_exp = self.preprocess_experience(exp)
        self._processed_experience_spec = self._experience_spec._replace(
            info=extract_spec(processed_exp.info))

        policy_step = common.algorithm_step(
            algorithm_step_func=self.train_step,
            ob_transformer=self._observation_transformer,
            time_step=exp,
            state=initial_state)
        info_spec = extract_spec(policy_step.info)
        self._training_info_spec = make_training_info(
            action=self._action_spec,
            action_distribution=self._action_dist_param_spec,
            step_type=self._time_step_spec.step_type,
            reward=self._time_step_spec.reward,
            discount=self._time_step_spec.discount,
            info=info_spec,
            collect_info=self._processed_experience_spec.info,
            collect_action_distribution=self._action_dist_param_spec)


    @tf.function
    def train(self,
              experience: Experience,
              num_updates=1,
              mini_batch_size=None,
              mini_batch_length=None):
        """Train using `experience`.

        Args:
            experience (Experience): experience from replay_buffer. It is
                assumed to be batch major.
            num_updates (int): number of optimization steps
            mini_batch_size (int): number of sequences for each minibatch
            mini_batch_length (int): the length of the sequence for each
                sample in the minibatch

        Returns:
            train_steps (int): the actual number of time steps that have been
                trained (a step might be trained multiple times)
        """

        experience = self.preprocess_experience(experience)

        length = experience.step_type.shape[1]
        mini_batch_length = (mini_batch_length or length)
        assert length % mini_batch_length == 0, (
            "length=%s not a multiple of mini_batch_length=%s" %
            (length, mini_batch_length))

        if len(tf.nest.flatten(self.train_state_spec)
               ) > 0 and not self._use_rollout_state:
            if mini_batch_length == 1:
                logging.fatal(
                    "Should use TrainerConfig.use_rollout_state=True "
                    "for off-policy training of RNN when minibatch_length==1.")
            else:
                warning_once(
                    "Consider using TrainerConfig.use_rollout_state=True "
                    "for off-policy training of RNN.")

        experience = tf.nest.map_structure(
            lambda x: tf.reshape(
                x, common.concat_shape([-1, mini_batch_length],
                                       tf.shape(x)[2:])), experience)

        batch_size = tf.shape(experience.step_type)[0]
        mini_batch_size = (mini_batch_size or batch_size)

        def _make_time_major(nest):
            """Put the time dim to axis=0."""
            return tf.nest.map_structure(lambda x: common.transpose2(x, 0, 1),
                                         nest)

        for u in tf.range(num_updates):
            if mini_batch_size < batch_size:
                indices = tf.random.shuffle(
                    tf.range(tf.shape(experience.step_type)[0]))
                experience = tf.nest.map_structure(
                    lambda x: tf.gather(x, indices), experience)
            for b in tf.range(0, batch_size, mini_batch_size):
                batch = tf.nest.map_structure(
                    lambda x: x[b:tf.minimum(batch_size, b + mini_batch_size)],
                    experience)
                batch = _make_time_major(batch)
                is_last_mini_batch = tf.logical_and(
                    tf.equal(u, num_updates - 1),
                    tf.greater_equal(b + mini_batch_size, batch_size))
                common.enable_summary(is_last_mini_batch)
                training_info, loss_info, grads_and_vars = self._update(
                    batch,
                    weight=tf.cast(tf.shape(batch.step_type)[1], tf.float32) /
                    float(mini_batch_size))
                if is_last_mini_batch:
                    self.training_summary(training_info, loss_info,
                                           grads_and_vars)

        self._train_step_counter.assign_add(1)
        train_steps = batch_size * mini_batch_length * num_updates
        return train_steps

    def _update(self, experience, weight):
        batch_size = tf.shape(experience.step_type)[1]
        counter = tf.zeros((), tf.int32)
        initial_train_state = common.get_initial_policy_state(
            batch_size, self.train_state_spec)
        if self._use_rollout_state:
            first_train_state = tf.nest.map_structure(
                lambda state: state[0, ...], experience.state)
        else:
            first_train_state = initial_train_state
        num_steps = tf.shape(experience.step_type)[0]

        def create_ta(s):
            # TensorArray cannot use Tensor (batch_size) as element_shape
            ta_batch_size = experience.step_type.shape[1]
            return tf.TensorArray(
                dtype=s.dtype,
                size=num_steps,
                element_shape=tf.TensorShape([ta_batch_size]).concatenate(
                    s.shape))

        experience_ta = tf.nest.map_structure(create_ta,
                                              self._processed_experience_spec)
        experience_ta = tf.nest.map_structure(
            lambda elem, ta: ta.unstack(elem), experience, experience_ta)
        training_info_ta = tf.nest.map_structure(create_ta,
                                                 self._training_info_spec)

        def _train_loop_body(counter, policy_state, training_info_ta):
            exp = tf.nest.map_structure(lambda ta: ta.read(counter),
                                        experience_ta)
            collect_action_distribution_param = exp.action_distribution
            collect_action_distribution = nested_distributions_from_specs(
                self._action_distribution_spec,
                collect_action_distribution_param)
            exp = exp._replace(action_distribution=collect_action_distribution)

            policy_state = common.reset_state_if_necessary(
                policy_state, initial_train_state,
                tf.equal(exp.step_type, StepType.FIRST))

            policy_step = common.algorithm_step(self.train_step,
                                                self._observation_transformer,
                                                exp, policy_state)

            action_dist_param = common.get_distribution_params(
                policy_step.action)

            training_info = make_training_info(
                action=exp.action,
                action_distribution=action_dist_param,
                reward=exp.reward,
                discount=exp.discount,
                step_type=exp.step_type,
                info=policy_step.info,
                collect_info=exp.info,
                collect_action_distribution=collect_action_distribution_param)

            training_info_ta = tf.nest.map_structure(
                lambda ta, x: ta.write(counter, x), training_info_ta,
                training_info)

            counter += 1

            return [counter, policy_step.state, training_info_ta]

        with tf.GradientTape(
                persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_variables)
            [_, _, training_info_ta] = tf.while_loop(
                cond=lambda counter, *_: tf.less(counter, num_steps),
                body=_train_loop_body,
                loop_vars=[counter, first_train_state, training_info_ta],
                back_prop=True,
                name="train_loop")
            training_info = tf.nest.map_structure(lambda ta: ta.stack(),
                                                  training_info_ta)
            action_distribution = nested_distributions_from_specs(
                self._action_distribution_spec,
                training_info.action_distribution)
            collect_action_distribution = nested_distributions_from_specs(
                self._action_distribution_spec,
                training_info.collect_action_distribution)
            training_info = training_info._replace(
                action_distribution=action_distribution,
                collect_action_distribution=collect_action_distribution)

        loss_info, grads_and_vars = self.train_complete(
            tape=tape, training_info=training_info, weight=weight)

        del tape

        return training_info, loss_info, grads_and_vars

