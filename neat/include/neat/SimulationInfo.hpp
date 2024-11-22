#pragma once

#include <any>
#include <vector>
#include "neat/Genome.hpp"
#include "neat/types.hpp"

namespace neat
{
    class [[nodiscard]] SimulationInfo
    {
        public:
        Genome &genome;
        std::vector<real_t> inputs;
        std::vector<real_t> outputs;
        real_t fitness{};
        bool is_done{};
        bool is_perfect{};
        std::any *const user_data;

        template<std::floating_point ...Args>
        void assign_inputs(Args ...args)
        {
            inputs.assign({ args... });
        }

        void run(activator_f *activator)
        {
            genome.step(*this, activator);
        }

        [[nodiscard]] SimulationInfo(Genome &genome, real_t fitness, std::any *const user_data)
            : genome{ genome }, fitness{ fitness }, user_data{ user_data } {
        }
    };

} // End namespace neat.