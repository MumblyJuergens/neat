#pragma once

#include <neat/SimulationInfo.hpp>
#include <neat/Simulation.hpp>
#include "Bird.hpp"
#include "Pipe.hpp"
#include "UserData.hpp"
#include "Config.hpp"
#include "logic.hpp"

namespace flappy
{
    struct Sim : public neat::Simulation
    {
        Bird bird{ .translation = {Bird::starting_x, Config::window_heightf / 2.0f} };

        void step(neat::SimulationInfo &info) override
        {
            auto *const userData = dynamic_cast<UserData *>(info.user_data);
            const Pipe *const pipe0 = &userData->pipes->at(0);
            const float input1 = Config::window_heightf / pipe0->top_distance(bird, Config::window_heightf);
            const float input2 = Config::window_heightf / pipe0->bottom_distance(bird, Config::window_heightf);
            const float input3 = Config::window_heightf / bird.translation.y;
            info.assign_inputs(1.0f, input1, input2, input3);
            info.run(std::tanh);
            const bool up = info.outputs[0] > 0.9f;

            if (game_logic(bird, up, userData->delta, *pipe0))
            {
                info.is_done = true;
            }
            else
            {
                info.fitness += 0.1f;
            }

            userData->birds_mbo->push_back(bird.translation);

        }

    };

} // End namespace flappy.