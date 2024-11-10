#pragma once

#include <span>
#include <neat/Genome.hpp>

namespace neat
{
    class [[nodiscard]] SimulationInfo
    {
        public:
        Genome &genome;
        std::vector<float> inputs;
        std::vector<float> outputs;
        float fitness{};
        bool is_done{};
        bool is_perfect{};

        template<std::floating_point ...Args>
        void assign_inputs(Args ...args)
        {
            inputs.assign({ args... });
        }

        void run()
        {
            genome.step(*this);
        }

        [[nodiscard]] SimulationInfo(Genome &genome, float fitness)
            : genome{ genome }, fitness{ fitness } {
        }
    };

    class [[nodiscard]] Simulation
    {
        public:
        virtual ~Simulation() = default;
        virtual void step(SimulationInfo &info) = 0;
    };
} // End namespace neat.