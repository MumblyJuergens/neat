#pragma once

#include "neat/types.hpp"
#include <cstdint>
#include <utility>

namespace neat
{

enum class NeuronType : uint8_t
{
    input,
    output,
    hidden,
};

class [[nodiscard]] Neuron final
{
    real_t m_value{};
    innovation_t m_number;
    uint8_t m_layer{};
    NeuronType m_type;

  public:
    [[nodiscard]] constexpr auto number() const noexcept { return m_number; }
    [[nodiscard]] constexpr auto type() const noexcept { return m_type; }
    [[nodiscard]] constexpr auto value() const noexcept { return m_value; }
    [[nodiscard]] constexpr auto layer() const noexcept { return m_layer; }

    constexpr void set_number(const innovation_t value) noexcept { m_number = value; }
    constexpr void set_value(const real_t value) noexcept { m_value = value; }
    constexpr void set_layer(const uint8_t value) noexcept { m_layer = value; }

    /// @brief Don't use. For serialization only.
    Neuron() = default;

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_number, m_type, m_value, m_layer);
    }

    [[nodiscard]] constexpr Neuron(const innovation_t number, const NeuronType type) noexcept
        : m_number{number}, m_layer{static_cast<uint8_t>(type == NeuronType::input ? 0 : 1)}, m_type{type}
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

static_assert(sizeof(Neuron) == sizeof(real_t) + 4);

} // namespace neat