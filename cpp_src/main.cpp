#include "CartPoleENV.h"
#include "ONNXPPOModel.h"


int main() {
    // Load the ONNX PPO model
    ONNXPPOModel ppo_model("../ray_output/model.onnx");

    // Initialize CartPole environment
    CartPoleEnv env;
    env.reset();

    bool done = false;
    int action_i = 0;

    // Run the environment loop
    while (!done) {
        std::cout << "Action: " << action_i << std::endl;
        action_i++;

        // Get the current observation (state) from the environment
        std::vector<float> obs = {static_cast<float>(env.position), static_cast<float>(env.velocity),
                                  static_cast<float>(env.pole_angle), static_cast<float>(env.pole_velocity)};

        // Select action using the PPO model
        int action = ppo_model.select_action(obs);

        // Take a step in the environment based on the action
        env.step(action);

        // Check if the episode is done
        done = env.is_done();
    }

    return 0;
}

