from absl import logging
from functools import partial
import unittest

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.parallel_actor_critic_algorithm import ParallelActorCriticAlgorithm, ActorCriticState
from alf.algorithms.parallel_trac_algorithm import ParallelTracAlgorithm
from alf.environments.suite_unittest import (PolicyUnittestEnv, ActionType)
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.utils import common


def create_parallel_ac_algorithm(observation_spec, action_spec,config, debug_summaries):

    actor_network = partial(
        ActorDistributionNetwork,
        fc_layer_params=(10,))

    value_network = partial(ValueNetwork, fc_layer_params=(10,))
    return ParallelActorCriticAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        actor_network_ctor=actor_network,
        value_network_ctor=value_network,
        config= config,
        optimizer=alf.optimizers.Adam(lr=1e-4),
        debug_summaries=debug_summaries,
        name="MyParallelActorCritic")


class ParallelTracAlgorithmTest(alf.test.TestCase):
    def test_parallel_trac_algorithm(self):
        num_parallel_agents = 5
        num_env = 5
        steps_per_episode = 15
        ac_algorithm_cls = create_parallel_ac_algorithm


        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=10,
            num_iterations=2500,
            num_checkpoints=5,
            evaluate=False,
            debug_summaries=False,
            summarize_grads_and_vars=False,
            summary_interval=5,
            epsilon_greedy=0.1,
            num_parallel_agents=num_parallel_agents)
        env_class = PolicyUnittestEnv
        env = env_class(
            num_env,
            steps_per_episode,
            action_type=ActionType.Discrete)

        eval_env = env_class(
            100,
            steps_per_episode,
            action_type=ActionType.Discrete)

        obs_spec = env._observation_spec
        action_spec = env._action_spec
        reward_spec = env._reward_spec

        alg = ParallelTracAlgorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            ac_algorithm_cls = ac_algorithm_cls,
            env=env,
            config=config)

        eval_env.reset()
        for i in range(750):
            alg.train_iter()
            if i < config.initial_collect_steps:
                continue
            eval_env.reset()
            eval_time_step = unroll(eval_env, alg, steps_per_episode - 1)
            logging.log_every_n_seconds(
                logging.INFO,
                "%d reward=%f" % (i, float(eval_time_step.reward.mean())),
                n_seconds=1)

        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=0.3)


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
        action, action_state, _ = algorithm.predict_step(transformed_time_step, ActorCriticState(actor=(), value=()))
        time_step = env.step(action)
        policy_state = action_state
    return time_step


if __name__ == '__main__':
    alf.test.main()