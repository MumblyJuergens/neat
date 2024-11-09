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
        std::size_t m_layer{};

    public:
        [[nodiscard]] constexpr Neuron(const innovation_t number, const NeuronType type) noexcept
            : m_number{ number }, m_type{ type }, m_layer{ type == NeuronType::input ? std::size_t{0} : std::size_t{1} }
        {
        }

        [[nodiscard]] constexpr auto number() const noexcept { return m_number; }
        [[nodiscard]] constexpr auto type() const noexcept { return m_type; }
        [[nodiscard]] constexpr auto value() const noexcept { return m_value; }
        [[nodiscard]] constexpr auto layer() const noexcept { return m_layer; }

        constexpr void set_value(const float value) noexcept { m_value = value; }
        constexpr void set_layer(const std::size_t value) noexcept { m_layer = value; }

        [[nodiscard]] static constexpr auto is_input(const Neuron &n) noexcept { return n.m_type == NeuronType::input; }
        [[nodiscard]] static constexpr auto is_output(const Neuron &n) noexcept { return n.m_type == NeuronType::output; }
        [[nodiscard]] static constexpr auto is_hidden(const Neuron &n) noexcept { return n.m_type == NeuronType::hidden; }
    };
} // End namespace neat.