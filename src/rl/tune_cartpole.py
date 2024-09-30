import time

import ray
import numpy as np
import onnxruntime as ort
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
            "_enable_rl_module_api": False,
            "_enable_learner_api": False,
        },
        run_config=air.RunConfig(
                        stop=stopping_criteria,
                        storage_path='../ray_results',),
    )
    return tuner.fit()


def save_policy(result):
    """
    Load policy and export to ONNX format

    :param result:
    :return:
    """

    best_result = result.get_best_result()
    best_checkpoint = best_result.best_checkpoints[-1][0].path
    latest_checkpoint = best_result.checkpoint.path

    ppo_algo = Algorithm.from_checkpoint(latest_checkpoint)
    policy = ppo_algo.get_policy()
    policy.export_model("../ray_output", onnx=17)


def create_env():
    env = gym.make('CartPole-v1')
    obs, _ = env.reset()
    return env, obs


class ONNXPPOModel:
    def __init__(self, model_path):
        self.model = ort.InferenceSession(model_path)

    def select_action(self, state):
        state = np.array(state, dtype=np.float32).reshape(1, -1)  # Reshape to match model input
        ort_inputs = {'obs': state, 'state_ins': []}
        ort_outs = self.model.run(None, ort_inputs)
        action_probs = ort_outs[0]  # Output from the policy (action probabilities)

        # Select the action with the highest probability
        action = np.argmax(action_probs, axis=1).item()
        return action


def test_original_model():
    analysis = tune.ExperimentAnalysis("../ray_results")
    ppo_algo = Algorithm.from_checkpoint(analysis.get_last_checkpoint().path)
    ppo_model = ppo_algo.get_policy()

    # Prepare Cartpole-v1 env to test the model
    env, obs = create_env()

    done, truncated, action_i = False, False, 0
    while not (done or truncated):
        logger.info(f'Action: {action_i}')
        action_i += 1

        # Capture the start time
        start = time.time()

        action, _, _ = ppo_model.compute_single_action(obs)

        # Capture the end time
        end = time.time()

        # Calculate the duration in microseconds
        duration = (end - start) * 1_000_000  # Convert seconds to microseconds

        # Print the duration
        logger.info(f"Time taken: {duration:.2f} microseconds")

        # Take the action in the environment
        obs, reward, done, truncated, info = env.step(action)

    env.close()


def test_converted_model():
    ppo_model = ONNXPPOModel('../ray_output/model.onnx')

    # Prepare Cartpole-v1 env to test the model
    env, obs = create_env()

    done, truncated, action_i = False, False, 0
    while not (done or truncated):
        logger.info(f'Action: {action_i}')
        action_i += 1

        # Capture the start time
        start = time.time()

        action = ppo_model.select_action(obs)

        # Capture the end time
        end = time.time()

        # Calculate the duration in microseconds
        duration = (end - start) * 1_000_000  # Convert seconds to microseconds

        # Print the duration
        logger.info(f"Time taken: {duration:.2f} microseconds")

        obs, reward, done, truncated, info = env.step(action)

    env.close()


if __name__ == "__main__":
    training = False

    ray.init(local_mode=True)

    if training:
        result = train_model()
        save_policy(result)

    test_original_model()
    # test_converted_model()

    ray.shutdown()
