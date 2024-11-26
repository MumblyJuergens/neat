#pragma once

#include <span>
#include "neat/Genome.hpp"

namespace neat
{
    class SimulationInfo;

    class [[nodiscard]] Simulation
    {
        bool m_is_inited{};

        public:

        [[nodiscard]] auto is_inited() const noexcept { return m_is_inited; }
        auto set_is_inited(const bool value) noexcept { m_is_inited = value; }

        virtual ~Simulation() = default;
        virtual void init([[maybe_unused]] SimulationInfo &info) {}
        virtual void step(SimulationInfo &info) = 0;
        virtual void skip([[maybe_unused]] SimulationInfo &info) {}
    };
} // End namespace neat.