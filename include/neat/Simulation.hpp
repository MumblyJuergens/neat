#pragma once

#include <span>

namespace neat
{
    struct Simulation
    {
        virtual ~Simulation() = default;
        virtual void supply(std::span<float> inputs) = 0;
        [[nodiscard]] virtual float receive(const std::span<float> outputs, const float fitness) = 0;
        [[nodiscard]] virtual bool is_done() const = 0;
        [[nodiscard]] virtual bool is_perfect() const { return false; }
    };
} // End namespace neat.