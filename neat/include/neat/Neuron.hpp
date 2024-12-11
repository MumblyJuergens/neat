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

        [[nodiscard]] constexpr auto number() const noexcept { return m_number; }
        [[nodiscard]] constexpr auto type() const noexcept { return m_type; }
        [[nodiscard]] constexpr auto value() const noexcept { return m_value; }
        [[nodiscard]] constexpr auto layer() const noexcept { return m_layer; }

        constexpr void set_number(const innovation_t value) noexcept { m_number = value; }
        constexpr void set_value(const real_t value) noexcept { m_value = value; }
        constexpr void set_layer(const int value) noexcept { m_layer = value; }

        /// @brief Don't use. For serialization only.
        Neuron() = default;

        template<typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_number, m_type, m_value, m_layer);
        }

        [[nodiscard]] constexpr Neuron(const innovation_t number, const NeuronType type) noexcept
            : m_number{ number }, m_type{ type }, m_layer{ type == NeuronType::input ? 0 : 1 }
        {
        }
        [[nodiscard]] constexpr Neuron(const Neuron &that) noexcept { *this = that; }
        constexpr Neuron &operator=(const Neuron &that) noexcept
        {
            m_number = that.m_number;
            m_type = that.m_type;
            m_value = 0;
            m_layer = that.m_layer;
            return *this;
        }
        [[nodiscard]] constexpr Neuron(Neuron &&that) noexcept { *this = std::move(that); }
        constexpr Neuron &operator=(Neuron &&that) noexcept
        {
            m_number = std::exchange(that.m_number, 0);
            m_type = that.m_type;
            m_value = 0;
            m_layer = std::exchange(that.m_layer, 0);
            return *this;
        }

        [[nodiscard]] static constexpr auto is_input(const Neuron &n) noexcept { return n.m_type == NeuronType::input; }
        [[nodiscard]] static constexpr auto is_output(const Neuron &n) noexcept { return n.m_type == NeuronType::output; }
        [[nodiscard]] static constexpr auto is_hidden(const Neuron &n) noexcept { return n.m_type == NeuronType::hidden; }
    };
} // End namespace neat.