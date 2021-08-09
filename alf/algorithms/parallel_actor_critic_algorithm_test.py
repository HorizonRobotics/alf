from absl import logging
from functools import partial
import unittest

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.parallel_actor_critic_algorithm import ParallelActorCriticAlgorithm, ParallelActorCriticState
from alf.environments.suite_unittest import (PolicyUnittestEnv, ActionType)
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.utils import common
from alf.utils.math_ops import clipped_exp
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm, ActorCriticState
from alf.algorithms.parallel_actor_critic_loss import ParallelActorCriticLoss
from alf.algorithms.actor_critic_loss import ActorCriticLoss

class ParallelActorCriticAlgorithmTest(alf.test.TestCase):
    def test_parallel_ac_algorithm(self):

        env_class = PolicyUnittestEnv
        num_env = 5
        num_eval_env = 1
        num_parallel_agents = num_env
        steps_per_episode = 15
        action_type = ActionType.Discrete
        
        ac_algorithm_cls = ParallelActorCriticAlgorithm
        ac_algorithm_loss = ParallelActorCriticLoss

        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=25,
            num_iterations=2500,
            num_checkpoints=10,
            evaluate=False,
            debug_summaries=False,
            summarize_grads_and_vars=False,
            summary_interval=5,
            epsilon_greedy=0.1,
            num_parallel_agents=num_parallel_agents)
        
        env = env_class(
            num_env,
            steps_per_episode,
            action_type=action_type)

        eval_env = env_class(
            num_eval_env,
            steps_per_episode,
            action_type=action_type)

        obs_spec = env._observation_spec
        action_spec = env._action_spec
        reward_spec = env._reward_spec
        
        fc_layer_params = (100,)
        ac_network = partial(ActorDistributionNetwork, fc_layer_params=fc_layer_params)
        vl_network =  partial(ValueNetwork, fc_layer_params=fc_layer_params)

        alg = ac_algorithm_cls(
            observation_spec=obs_spec,
            action_spec=action_spec,
            reward_spec= reward_spec,
            actor_network_ctor= ac_network,
            value_network_ctor= vl_network,
            env = env,
            config= config,
            loss_class=ac_algorithm_loss,
            optimizer=alf.optimizers.Adam(lr=1e-3),
            debug_summaries=False,
            name="MyParallelActorCritic")

        print(config.initial_collect_steps)
        eval_env.reset()
        sum_reward = 0
        for i in range(2000):
            alg.train_iter()
            if i < config.initial_collect_steps:
                continue
            eval_env.reset()
            eval_time_step = unroll(eval_env, alg, steps_per_episode - 1)
            sum_reward = sum_reward + eval_time_step.reward
            logging.log_every_n_seconds(
                logging.INFO,
                "%d reward=%f" % (i, float(sum_reward / (i+1))),
                n_seconds=1)

        self.assertAlmostEqual(
            1.0, float(sum_reward / (i+1)), delta=0.3)


def unroll(env, algorithm, steps):
    time_step = common.get_initial_time_step(env)
    policy_state = algorithm.get_initial_predict_state(env.batch_size)
    trans_state = algorithm.get_initial_transform_state(env.batch_size)
    for _ in range(steps):
        policy_state = common.reset_state_if_necessary(
            policy_state, algorithm.get_initial_predict_state(env.batch_size),
            time_step.is_first())
        transformed_time_step, trans_state = algorithm.transform_timestep(
            time_step, trans_state)
        action, action_state, _ = algorithm.predict_step(transformed_time_step, ParallelActorCriticState(actors=(), values=()))
        time_step = env.step(action)
        policy_state = action_state
    return time_step


if __name__ == '__main__':
    alf.test.main()