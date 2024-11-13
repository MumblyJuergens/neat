#pragma once

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

        template<std::floating_point ...Args>
        void assign_inputs(Args ...args)
        {
            inputs.assign({ args... });
        }

        void run(activator_f *activator)
        {
            genome.step(*this, activator);
        }

        [[nodiscard]] SimulationInfo(Genome &genome, real_t fitness)
            : genome{ genome }, fitness{ fitness } {
        }
    };

} // End namespace neat.