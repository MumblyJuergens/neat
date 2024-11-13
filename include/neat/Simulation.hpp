#pragma once

#include <span>
#include "neat/Genome.hpp"

namespace neat
{
    class SimulationInfo;

    class [[nodiscard]] Simulation
    {
        public:
        virtual ~Simulation() = default;
        virtual void step(SimulationInfo &info) = 0;
    };
} // End namespace neat.