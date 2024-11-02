#pragma once

#include "neat/innovation.hpp"

namespace neat
{
    enum class NeuronType
    {
        input,
        output,
        hidden,
    };

    class [[nodiscard]] Neuron final
    {
        const innovation_t m_number;
        const NeuronType m_type;
        float m_value{};

    public:
        [[nodiscard]] constexpr Neuron(const innovation_t number, const NeuronType type) noexcept : m_number{number}, m_type{type} {}

        [[nodiscard]] constexpr auto number() const noexcept { return m_number; }
        [[nodiscard]] constexpr auto type() const noexcept { return m_type; }
        [[nodiscard]] constexpr auto value() const noexcept { return m_value; }

        constexpr void set_value(const float value) noexcept { m_value = value; }
    };
} // End namespace neat.