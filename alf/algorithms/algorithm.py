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
"""Algorithm base class."""

from abc import abstractmethod
from absl import logging
import copy

import tensorflow as tf

from tf_agents.utils import eager_utils

import alf.utils
from alf.utils.common import namedtuple, LossInfo

AlgorithmStep = namedtuple("AlgorithmStep", ["outputs", "state", "info"])


class Algorithm(tf.Module):
    """Algorithm base class.

    Algorithm is a generic interface for supervised training algorithms.

    User needs to implement train_step() and calc_loss()/train_complete().

    train_step() is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    train_complete() is called every train_interval steps (specified in
    TrainingPolicy). All the training information collected at each previous
    train_step() are batched and provided as arguments for train_complete().

    The following is the pseudo code to illustrate how Algorithm is for
    training.

    ```python
    while training not ends:
        training_info = []
        with GradientTape() as tape:
        for i in range(train_interval):
            get inputs
            outputs, state, info = train_step(inputs, state)
            add info to training_info

        train_complete(tape, batched_training_info)
    ```
    """

    def __init__(self,
                 train_state_spec=None,
                 predict_state_spec=None,
                 optimizer=None,
                 trainable_module_sets=None,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 debug_summaries=False,
                 name="Algorithm"):
        """Create an Algorithm.

        Each algorithm can have zero, one or multiple optimizers. Each optimizer
        is responsible for optimizing a set of modules. The `optimizer` can be
        in one of the following 3 forms:

        None or empty list: there is no default Optimizer. There cannot be any
            non-algorithm child modules, or this algorithm should be a child
            algorithm of another algorithm so that these non-algorithm child
            modules can be optimized by that parent algorithm.

        Optimizer: this is the default Optimizer. All the non-algorithm
            child modules will be optimized by this optimizer.

        non-empty list of Optimizer: `trainable_module_sets` should be a list of
            list of modules. Each optimizer will optimize one list of modules.
            optimizer[0] is the default optimizer. And all the non-algorithm
            child modules which are not in the `trainable_module_sets` will be
            optimized by optimzer[0].

        The child algorithms will be optimized by their own optimizers if they
        have. If a child algorithm does not have optimizer, it will be optimized
        by the default optimizer.

        A requirement for this optimizer structure to work is that there is no
        algorithm which is a submodule of a non-algorithm module. Currently,
        this is not checked by the framework. It's up to the user to make sure
        this is true.

        Args:
            train_state_spec (nested TensorSpec): for the network state of
                `train_step()`
            predict_state_spec (nested TensorSpec): for the network state of
                `predict()`. If None, it's assume to be same as train_state_spec
            optimizer (None|Optimizer|list[Optimizer]): The optimizer for
                training. See comments above for detail.
            trainable_module_sets (list[list]): See comments above for detail.
            gradient_clipping (float): If not None, serve as a positive threshold
            clip_by_global_norm (bool): If True, use tf.clip_by_global_norm to
                clip gradient. If False, use tf.clip_by_norm for each grad.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this algorithm.
        """
        super(Algorithm, self).__init__(name=name)

        self._train_state_spec = train_state_spec
        if predict_state_spec is None:
            predict_state_spec = train_state_spec
        self._predict_state_spec = predict_state_spec
        self._is_rnn = len(tf.nest.flatten(train_state_spec)) > 0

        if not optimizer:
            assert trainable_module_sets is None
            self._init_optimizers = [None]
            self._init_module_sets = [[]]
        elif isinstance(optimizer, tf.optimizers.Optimizer):
            assert trainable_module_sets is None
            self._init_optimizers = [optimizer]
            self._init_module_sets = [[]]
        else:
            assert isinstance(optimizer, list)
            assert isinstance(trainable_module_sets, list)
            assert len(trainable_module_sets) == len(optimizer), (
                "`optimizer` and `trainable_module_sets`"
                "should have same length")
            self._init_optimizers = optimizer
            self._init_module_sets = trainable_module_sets

        self._cached_opt_and_var_sets = None
        self._gradient_clipping = gradient_clipping
        self._clip_by_global_norm = clip_by_global_norm
        self._debug_summaries = debug_summaries

    @property
    def trainable_variables(self):
        return sum([var_set for _, var_set in self._get_opt_and_var_sets()],
                   [])

    def get_optimizer_and_module_sets(self):
        """Get the optimizers and the corresponding module sets.

        Returns:
            list[tuple(Opimizer, list[Module])]: optimizer can be None, which
                means that no optimizer is specified for the corresponding
                modules.
        """

        def _is_alg(obj):
            return isinstance(obj, Algorithm)

        def _is_module_or_var(obj):
            return ((isinstance(obj, tf.Variable) and obj.trainable)
                    or (isinstance(obj, tf.Module)
                        and not isinstance(obj, Algorithm)))

        module_sets = [copy.copy(s) for s in self._init_module_sets]
        optimizers = copy.copy(self._init_optimizers)
        module_ids = set(map(id, sum(module_sets, [])))

        for alg in self._get_children(_is_alg):
            for opt, module_set in alg.get_optimizer_and_module_sets():
                if opt is not None:
                    optimizers.append(opt)
                    module_sets.append(module_set)
                else:
                    module_sets[0].extend(module_set)

        for module in self._get_children(_is_module_or_var):
            if id(module) not in module_ids:
                module_sets[0].append(module)

        return list(zip(optimizers, module_sets))

    def _get_children(self, predicate):
        module_dict = vars(self)
        children = []
        ids = set()
        for key in sorted(module_dict):
            attr = module_dict[key]
            if key in [
                    '_init_optimizers', '_cached_opt_and_var_sets',
                    '_init_module_sets'
            ]:
                continue
            for leaf in tf.nest.flatten(attr):
                if predicate(leaf) and not id(leaf) in ids:
                    children.append(leaf)
                    ids.add(id(leaf))
        return children

    def _get_cached_opt_and_var_sets(self):
        if self._cached_opt_and_var_sets is None:
            self._cached_opt_and_var_sets = self._get_opt_and_var_sets()
        return self._cached_opt_and_var_sets

    def _get_opt_and_var_sets(self):
        opt_and_var_sets = []
        optimizer_and_module_sets = self.get_optimizer_and_module_sets()
        optimizer_info = []
        for opt, module_set in optimizer_and_module_sets:
            optimizer_info.append(
                "optimizer %s: modules %s" % (opt.get_config(), ' '.join(
                    [m.name for m in module_set if m is not None])))
            vars = []
            for module in module_set:
                if module is None:
                    continue
                assert id(module) != id(self)
                if isinstance(module, tf.Variable):
                    vars.append(module)
                elif isinstance(module, tf.Module):
                    vars += list(module.trainable_variables)
                else:
                    raise ValueError("Unsupported module type %s" % module)
            opt_and_var_sets.append((opt, vars))
        self._optimizer_info = '\n\n'.join(optimizer_info)
        return opt_and_var_sets

    @property
    def predict_state_spec(self):
        """Returns the RNN state spec for predict()."""
        return self._predict_state_spec

    @property
    def train_state_spec(self):
        """Returns the RNN state spec for train_step()."""
        return self._train_state_spec

    #------------- User need to implement the following functions -------

    # Subclass may override predict() to allow more efficient implementation
    def predict(self, inputs, state=None):
        """Predict for one step of inputs.

        Args:
            inputs (nested Tensor): inputs for prediction
            state (nested Tensor): network state (for RNN)

        Returns:
            AlgorithmStep
                outputs (nested Tensor): prediction result
                state (nested Tensor): should match `predict_state_spec`
        """
        algorithm_step = self.train_step(inputs, state)
        return algorithm_step._replace(info=None)

    @abstractmethod
    def train_step(self, inputs, state=None):
        """Perform one step of predicting and training computation.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            inputs (nested Tensor): inputs for train
            state (nested Tensor): consistent with train_state_spec

        Returns:
            AlgorithmStep
                outputs (nested Tensor): predict outputs
                state (nested Tensor): should match `predict_state_spec`
                info (nested Tensor): information for training. If this is
                    LossInfo, calc_loss() in Algorithm can be used. Otherwise,
                    the user needs to override calc_loss() to calculate loss or
                    override train_complete() to do customized training.
        """
        pass

    # Subclass may override train_complete() to allow customized training
    def train_complete(self,
                       tape: tf.GradientTape,
                       training_info,
                       valid_masks=None,
                       weight=1.0):
        """Complete one iteration of training.

        `train_complete` should calculate gradients and update parameters using
        those gradients.

        Args:
            tape (tf.GradientTape): the tape which are used for calculating
                gradient. All the previous `train_interval` `train_step()` for
                are called under the context of this tape.
            training_info (nested Tensor): information collected for training.
                It is batched from each `info` returned bt `train_step()`
            valid_masks (tf.Tensor): masks indicating which samples are valid.
                shape=(T, B), dtype=tf.float32
            weight (float): weight for this batch. Loss will be multiplied with
                this weight before calculating gradient
        Returns:
            loss_info (LossInfo): loss information
            grads_and_vars (list[tuple]): list of gradient and variable tuples
        """
        with tape:
            loss_info = self.calc_loss(training_info)
            if valid_masks is not None:
                loss_info = tf.nest.map_structure(
                    lambda l: tf.reduce_mean(l * valid_masks)
                    if len(l.shape) == 2 else l, loss_info)
            else:
                loss_info = tf.nest.map_structure(lambda l: tf.reduce_mean(l),
                                                  loss_info)
            if isinstance(loss_info.scalar_loss, tf.Tensor):
                assert len(loss_info.scalar_loss.shape) == 0
                loss_info = loss_info._replace(
                    loss=loss_info.loss + loss_info.scalar_loss)
            loss = weight * loss_info.loss

        opt_and_var_sets = self._get_cached_opt_and_var_sets()
        all_grads_and_vars = ()
        for i, (optimizer, vars) in enumerate(opt_and_var_sets):
            if len(vars) == 0:
                continue
            assert optimizer is not None, "optimizer needs to be provides at __init__()"
            grads = tape.gradient(loss, vars)
            grads_and_vars = tuple(zip(grads, vars))
            all_grads_and_vars = all_grads_and_vars + grads_and_vars
            if self._gradient_clipping is not None:
                if self._clip_by_global_norm:
                    grads, global_norm = tf.clip_by_global_norm(
                        grads, self._gradient_clipping)
                    grads_and_vars = tuple(zip(grads, vars))
                    alf.utils.common.run_if(
                        alf.utils.common.should_record_summaries(), lambda: tf.
                        summary.scalar("global_grad_norm/%s" % i, global_norm))
                else:
                    grads_and_vars = eager_utils.clip_gradient_norms(
                        grads_and_vars, self._gradient_clipping)

            optimizer.apply_gradients(grads_and_vars)

        return loss_info, all_grads_and_vars

    # Subclass may override calc_loss() to allow more sophisticated loss
    def calc_loss(self, training_info):
        """Calculate the loss at each step for each sample.

        Args:
            training_info (nested Tensor): information collected for training.
                It is batched from each `info` returned bt `train_step()`
        Returns:
            loss_info (LossInfo): loss at each time step for each sample in the
                batch. The shapes of the tensors in loss_info should be (T, B)
        """
        assert isinstance(training_info, LossInfo), (
            "training_info returned by"
            " train_step() should be LossInfo. Otherwise you need override"
            " calc_loss() to generate LossInfo from training_info")
        return training_info
