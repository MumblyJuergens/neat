#pragma once

namespace neat
{
    class [[nodiscard]] Synapse final
    {
        const std::size_t m_in;
        const std::size_t m_out;
        float m_weight;
        const bool m_enabled;
        const int m_innovation;

    public:
        [[nodiscard]] constexpr Synapse(const std::size_t in, const std::size_t out, const float weight, const bool enabled, const int innovation) noexcept
            : m_in{in}, m_out{out}, m_weight{weight}, m_enabled{enabled}, m_innovation{innovation} {}

        [[nodiscard]] constexpr auto in() const noexcept { return m_in; }
        [[nodiscard]] constexpr auto out() const noexcept { return m_out; }
        [[nodiscard]] constexpr auto weight() const noexcept { return m_weight; }
        [[nodiscard]] constexpr auto enabled() const noexcept { return m_enabled; }
        [[nodiscard]] constexpr auto innovation() const noexcept { return m_innovation; }

        constexpr void set_weight(const float value) noexcept { m_weight = value; }
    };
} // End namespace neat.