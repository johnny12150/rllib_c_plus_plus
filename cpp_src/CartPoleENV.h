#include <iostream>
#include <vector>
#include <cmath>

struct CartPoleEnv {
    double position = 0.0;
    double velocity = 0.0;
    double pole_angle = 0.0;
    double pole_velocity = 0.0;
    bool done = false;

    CartPoleEnv() {
        reset();
    }

    void reset() {
        position = 0.0;
        velocity = 0.0;
        pole_angle = 0.05;
        pole_velocity = 0.0;
        done = false;
    }

    std::vector<float> step(int action) {
        // Simplified physics for CartPole
        double force = (action == 1) ? 10 : -10;
        velocity += force * 0.02;
        position += velocity * 0.02;

        pole_velocity += (force * 0.02 - pole_angle * 0.02);
        pole_angle += pole_velocity * 0.02;

        if (std::abs(position) > 2.4 || std::abs(pole_angle) > 0.209) {
            done = true;
        }

        return {float(position), float(velocity), float(pole_angle), float(pole_velocity)};
    }

    bool is_done() {
        return done;
    }
};
