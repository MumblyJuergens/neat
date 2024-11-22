#pragma once

#include <neat/SimulationInfo.hpp>
#include "Bird.hpp"
#include "Pipe.hpp"
#include "UserData.hpp"
#include "Config.hpp"

namespace flappy
{
    struct Sim : public neat::Simulation
    {
        Bird bird{ .translation = {Bird::starting_x, Config::window_heightf / 2.0f} };
        float vertSpeed = 0.0f;

        void step(neat::SimulationInfo &info) override
        {
            auto *const userData = std::any_cast<UserData>(info.user_data);
            const Pipe *const pipe0 = &userData->pipes->at(0);
            const float input1 = Config::window_heightf / pipe0->top_distance(bird, Config::window_heightf);
            const float input2 = Config::window_heightf / pipe0->bottom_distance(bird, Config::window_heightf);
            const float input3 = Config::window_heightf / bird.translation.y;
            info.assign_inputs(1.0f, input1, input2, input3);
            info.run(std::tanh);
            const bool up = info.outputs[0] > 0.9f;

            if (up)
            {
                vertSpeed = Bird::fly_speed;
            }
            bird.translation.y += vertSpeed * userData->delta;
            vertSpeed -= Bird::gravity * userData->delta;

            if (pipe0->collide(bird, Config::window_heightf) || bird.translation.y < -Bird::radius || bird.translation.y > Config::window_heightf + Bird::radius)
            {
                info.is_done = true;
            }
            else
            {
                info.fitness += 0.1f;
            }

            userData->birds_mbo_data->push_back(bird.translation);

        }

    };

} // End namespace flappy.