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

from abc import abstractmethod
from collections import namedtuple

from tf_agents.policies import tf_policy

LossStep = namedtuple("LossStep", ("trajectory", "critic"))


class SimpleAlgorithm(object):
    """
    SimpleAgorithm encapsulates the following pattern for many RL algorithms.
    To follow this pattern, the user need to implement the following functions:
        predict()
        pre_critic_step()
        critic()
        loss_step()
        update()

    The following pseudo-code illustrate how these functions are used by
    SimpleAgent:

    ```python
    for next_time_step in next_time_steps:
        critics[i], state = self.pre_critic_step(next_time_step, state)
    
    critics = self.critic(next_steps, critics)

    with tr.GradientTape() as tape:
        for time_step, critic in zip(time_steps, critics):
            loss_infos[i], state = self.loss_step(time_step, critic)

    total_loss = self._calc_total_loss(loss_infos)
    
    grads = tape.gradient(total_loss, variables_to_train)

    self.update(grads)
    ```
    """

    def __init__(self, model, policy=None, collect_policy=None,
                 optimizer=None):
        """
        Args:
          policy (tf_policy.Base) : policy for evaluation
          collect_policy (tf_policy.Base) : policy for training
          optimizer (None|tf.optimizers.Optimizer): The optimizer for training.
            If None, user must override update().
        """

        self._model = model
        self._optimizer = optimizer

        if policy is None:
            policy = AlgorithmPolicy(self)
        if collect_policy is None:
            collect_policy = policy
        self._policy = policy
        self._collect_policy = collect_policy

    @property
    def model(self):
        """Get the model of this algorithm
        Returns:
            model (alf.RLModel)
        """
        return self._model

    @property
    def time_step_spec(self):
        """Describes the `TimeStep` tensors returned by `step()`.

        Returns:
        A `TimeStep` namedtuple with `TensorSpec` objects instead of Tensors,
        which describe the shape, dtype and name of each tensor returned by
        `step()`.
        """
        return self._model.time_step_spec

    @property
    def action_spec(self):
        """Describes the TensorSpecs of the Tensors expected by `step(action)`.

        `action` can be a single Tensor, or a nested dict, list or tuple of
        Tensors.

        Returns:
        An single BoundedTensorSpec, or a nested dict, list or tuple of
        `BoundedTensorSpec` objects, which describe the shape and
        dtype of each Tensor expected by `step()`.
        """
        return self._model.action_spec

    @property
    def policy_state_spec(self):
        """Describes the Tensors expected by `step(_, policy_state)`.

        `policy_state` can be an empty tuple, a single Tensor, or a nested dict,
        list or tuple of Tensors.

        Returns:
        An single TensorSpec, or a nested dict, list or tuple of
        `TensorSpec` objects, which describe the shape and
        dtype of each Tensor expected by `step(_, policy_state)`.
        """
        return self._model.policy_state_spec

    @property
    def info_spec(self):
        """Describes the Tensors emitted as info by `action` and `distribution`.

        `info` can be an empty tuple, a single Tensor, or a nested dict,
        list or tuple of Tensors.

        Returns:
        An single TensorSpec, or a nested dict, list or tuple of
        `TensorSpec` objects, which describe the shape and
        dtype of each Tensor expected by `step(_, policy_state)`.
        """
        return self._model.info_spec

    @property
    def policy(self):
        return self._policy

    @property
    def collect_policy(self):
        return self._collect_policy

    @property
    def trainable_weights(self):
        """Get the list of trainable weights for the algorithm
        Returns:
            list[Variable]
        """
        self.model.trainable_weights

    #------------- User may/need to implement the following functions -------
    def predict(self, observation, state=None):
        """Predict for one step of observation
        Returns:
            distribution (nested tf.distribution): action distribution
            state (None | nested tf.Tensor): RNN state
        """
        return self._model(observation, state)

    @abstractmethod
    def pre_critic_step(self, next_trajectory, state=None):
        """Prepare quantities for calculating critic for each step.
        For example, value or Q-value at each step.
        Args:
            next_trajectory (Trajectory): trajectory for next step
            state (nested Tensor): RNN state
        Returns:
            pre_critic (nested Tensor): this result from all steps will be
                passed to critic() for further processing (e.g. nstep return)
            next_state (nested Tensor): next state of RNN
        """
        pass

    @abstractmethod
    def critic(self, trajectory, next_trajectory, pre_critic):
        """Calculate final critics from pre_critic
        Args:
            trajectory (Trajectory): trajectory for the first step
            next_trajectory (Trajectory): trajectory for the next step
            pre_critic (nested Tensor): result from pre_critic_step(), assembled
                accross time with shape as (b, t, ...).
        Returns:
            critic (nested Tensor): each step of critic will be passed to
               loss_step()
        """
        pass

    @abstractmethod
    def loss_step(self, input, state=None):
        """
        Args:
            input (LossStep):
            state (nested Tensor): RNN state
        Returns:
            costs (nested Tensor): costs for this step
            next_state (nested Tensor): next state for RNN
        """
        pass

    def update(self, grads_and_vars, train_step_counter):
        """Update the model using the gradients.
        User may override te this function to allow customized update.
        Args:
          grads_and_vars (list[(grad, var)]): gradients and corresponding
            variables
          train_step_counter (0-D Tensor): An optional counter to increment
            every time update is run.
        """
        self._optimizer.apply_gradients(
            grads_and_vars, global_step=train_step_counter)


class AlgorithmPolicy(tf_policy.Base):
    def __init__(self, algorithm, name=None):
        """Wrap an algorithm to get a tf_policy.Base
        Args:
          algorithm(Algorithm): an algorithm instance
        """
        self._alg = algorithm
        super(AlgorithmPolicy, self).__init__(
            time_step_spec=algorithm.time_step_spec,
            action_spec=algorithm.action_spec,
            policy_state_spec=algorithm.policy_state_spec,
            info_spec=algorithm.info_spec,
            name=name)

    def _distribution(self, time_step, policy_state):
        return self._alg.predict(time_step.observation, policy_state)
