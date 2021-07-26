from functools import partial
import torch
import torch.distributions as td
import unittest

import alf
from alf.utils import common, dist_utils, tensor_utils
from alf.data_structures import StepType, TimeStep
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.parallel_actor_critic_algorithm import ParallelActorCriticAlgorithm
from alf.algorithms.rl_algorithm_test import MyEnv


def create_algorithm(env, num_parallel_agents):
    config = TrainerConfig(root_dir="dummy", unroll_length=5, num_parallel_agents=num_parallel_agents)
    obs_spec = alf.TensorSpec((2, ), dtype='float32')
    action_spec = alf.BoundedTensorSpec(
        shape=(), dtype='int32', minimum=0, maximum=2)

    fc_layer_params = (10, 8, 6)

    actor_network = partial(
        ActorDistributionNetwork,
        fc_layer_params=fc_layer_params,
        discrete_projection_net_ctor=alf.networks.CategoricalProjectionNetwork)

    value_network = partial(ValueNetwork, fc_layer_params=(10, 8, 1))

    alg = ParallelActorCriticAlgorithm(
        observation_spec=obs_spec,
        action_spec=action_spec,
        actor_network_ctor=actor_network,
        value_network_ctor=value_network,
        env=env,
        config=config,
        optimizer=alf.optimizers.Adam(lr=1e-2),
        debug_summaries=True,
        name="MyActorCritic")
    return alg


class ActorCriticAlgorithmTest(alf.test.TestCase):
    def test_ac_algorithm(self):
        num_env = 3
        env = MyEnv(batch_size=num_env)
        alg1 = create_algorithm(env, num_env)

        iter_num = 50
        for _ in range(iter_num):
            alg1.train_iter()

        time_step = common.get_initial_time_step(env)
        state = alg1.get_initial_predict_state(env.batch_size)
        policy_step = alg1.rollout_step(time_step, state)
        logits = policy_step.info.action_distribution.log_prob(
            torch.arange(3).reshape(3, 1))
        print("logits: ", logits)
        self.assertTrue(torch.all(logits[1, :] > logits[0, :]))
        self.assertTrue(torch.all(logits[1, :] > logits[2, :]))

        # global counter is iter_num due to alg1
        self.assertTrue(alf.summary.get_global_counter() == iter_num)

    def test_ac_algorithm_with_global_counter(self):
        num_env = 3
        env = MyEnv(batch_size=num_env)
        alg2 = create_algorithm(env,num_env)
        new_iter_num = 3
        for _ in range(new_iter_num):
            alg2.train_iter()
        # new_iter_num of iterations done in alg2
        self.assertTrue(alf.summary.get_global_counter() == new_iter_num)


if __name__ == '__main__':
    alf.test.main()