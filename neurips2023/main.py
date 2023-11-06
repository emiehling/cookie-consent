# Runs a recommender system simulation with user consent decisions.
#
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from functools import partial

from custom_agents import bandit_mf_agent, random_agent_multi
from custom_environments import recommender_environment
from custom_simulator import runner_lib

from experiment_params import get_experiments

def create_agent(environment, env_config, eval_mode):
    """Creates an instance of named agent

    Args:
    environment: A recsim Gym environment.
    env_config: configuration parameters of the environment.
    eval_mode: A bool for whether the agent is in training or evaluation mode.

    Returns:
    An instance of the named agent.
    """
    kwargs = {
      'observation_space': environment.observation_space,
      'action_space': environment.action_space,
      'eval_mode': eval_mode,
      'random_seed': None,
    }

    agent_params = env_config['agent']
    agent_name = agent_params['name']

    if agent_name == 'bandit-mf':
        return bandit_mf_agent.BanditMFAgent(agent_params,
                                             observation_space=environment.observation_space,
                                             action_space=environment.action_space,
                                             **kwargs)
    elif agent_name == 'random-multi':
        return random_agent_multi.RandomAgentMulti(action_space=environment.action_space,
                                                   random_seed=0)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def main():
    # Load experiments
    experiments = get_experiments()

    base_dir = '/tmp/recommendation_with_consent'
    os.makedirs(base_dir, exist_ok=True)

    for experiment_index, (_, experiment_params) in enumerate(experiments.items(), start=1):
        print(f"Running experiment {experiment_index}")

        # Create directory to store results
        experiment_dir = os.path.join(base_dir, f'experiment_{experiment_index}')
        os.makedirs(experiment_dir, exist_ok=True)

        # Create environment and run
        env_config = {**experiment_params}
        create_agent_fn = partial(create_agent, env_config=env_config, eval_mode=False)
        environment = recommender_environment.create_environment(env_config)
        runner = runner_lib.EvalRunner(
            base_dir=experiment_dir,
            create_agent_fn=create_agent_fn,
            env=environment,
            max_eval_episodes=2,
            test_mode=False
        )
        runner.run_experiment()


if __name__ == '__main__':
    main()



    # runner = runner_lib.TrainRunner(
    #     base_dir=FLAGS.base_dir,
    #     create_agent_fn=create_agent,
    #     env=interest_evolution.create_environment(env_config),
    #     episode_log_file=FLAGS.episode_log_file,
    #     max_training_steps=50,
    #     num_iterations=10)
    # runner.run_experiment()
    # experiments = get_experiments()
    #
    # base_dir = 'tmp/recommendation_with_consent'
    # os.makedirs(base_dir, exist_ok=True)
    #
    # for experiment_index, (_, experiment_params) in enumerate(experiments.items(), start=1):
    #     print(f"Running experiment {experiment_index}")
    #
    #     experiment_dir = os.path.join(base_dir, f'experiment_{experiment_index}')
    #     os.makedirs(experiment_dir, exist_ok=True)
    #
    #     env_config = {**experiment_params}
    #
    #     agent_name = env_config['agent']['name']
    #     runner = runner_lib.EvalRunner(
    #         base_dir=base_dir,
    #         # base_dir=f"{FLAGS.base_dir}_exp{experiment_index}",
    #         create_agent_fn=partial(create_agent, agent_name=agent_name),
    #         env=recommender_environment.create_environment(env_config),
    #         max_eval_episodes=5,
    #         test_mode=True)
    #     runner.run_experiment()