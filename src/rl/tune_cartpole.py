import ray
import torch
import gymnasium as gym
from loguru import logger
from ray import tune, air
from ray.rllib.algorithms.algorithm import Algorithm

stopping_criteria = {"time_total_s": 20}

def train_model():
    tuner = tune.Tuner(
        "PPO",
        param_space={
            "env": "CartPole-v1",
            "num_workers": 2,
            "num_cpus": 1,  # number of CPUs to use per trial
            "num_gpus": 0,  # number of GPUs to use per trial
            "disable_env_checking": True,  # avoid weird shape error in env
        },
        run_config=air.RunConfig(
                        stop=stopping_criteria,
                        storage_path='../ray_results',),
    )
    return tuner.fit()


def save_policy(result):
    """
    Load policy and export to .pt file

    :param result:
    :return:
    """

    best_result = result.get_best_result()
    best_checkpoint = best_result.best_checkpoints[-1][0].path
    latest_checkpoint = best_result.checkpoint.path

    ppo_algo = Algorithm.from_checkpoint(latest_checkpoint)
    policy = ppo_algo.get_policy()
    policy.export_model("../ray_output")
    policy.export_model("../ray_output")  # normal torch model


def create_env():
    env = gym.make('CartPole-v1')
    obs, _ = env.reset()
    return env, obs



if __name__ == "__main__":
    ray.init(local_mode=True)

    result = train_model()
    convert_policy(result)

    ray.shutdown()
