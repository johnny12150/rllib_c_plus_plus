import ray
from ray import tune, air

stopping_criteria = {"time_total_s": 20}


if __name__ == "__main__":
    ray.init()

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
    results = tuner.fit()
