#pragma once

#include <cstddef>
#include "neat/types.hpp"

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
        innovation_t m_number;
        NeuronType m_type;
        real_t m_value{};
        int m_layer{};

        public:
        [[nodiscard]] constexpr Neuron(const innovation_t number, const NeuronType type) noexcept
            : m_number{ number }, m_type{ type }, m_layer{ type == NeuronType::input ? 0 : 1 }
        {
        }

        [[nodiscard]] constexpr auto number() const noexcept { return m_number; }
        [[nodiscard]] constexpr auto type() const noexcept { return m_type; }
        [[nodiscard]] constexpr auto value() const noexcept { return m_value; }
        [[nodiscard]] constexpr auto layer() const noexcept { return m_layer; }

        constexpr void set_number(const innovation_t value) noexcept { m_number = value; }
        constexpr void set_value(const float value) noexcept { m_value = value; }
        constexpr void set_layer(const int value) noexcept { m_layer = value; }

        [[nodiscard]] static constexpr auto is_input(const Neuron &n) noexcept { return n.m_type == NeuronType::input; }
        [[nodiscard]] static constexpr auto is_output(const Neuron &n) noexcept { return n.m_type == NeuronType::output; }
        [[nodiscard]] static constexpr auto is_hidden(const Neuron &n) noexcept { return n.m_type == NeuronType::hidden; }
    };
} // End namespace neat.