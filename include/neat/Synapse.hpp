#pragma once

#include "neat/innovation.hpp"
#include "neat/Config.hpp"
#include "neat/Random.hpp"

namespace neat
{

    class [[nodiscard]] Synapse final
    {
        innovation_t m_in;
        innovation_t m_out;
        float m_weight;
        bool m_enabled{ true };
        innovation_t m_innovation;

        public:
        [[nodiscard]] constexpr Synapse(const innovation_t in, const innovation_t out, const float weight, const innovation_t innovation) noexcept
            : m_in{ in }, m_out{ out }, m_weight{ weight }, m_innovation{ innovation } {
        }
        [[nodiscard]] Synapse(const Synapse &) noexcept = default;
        [[nodiscard]] Synapse(Synapse &&) noexcept = default;
        Synapse &operator=(const Synapse &) noexcept = default;
        Synapse &operator=(Synapse &&) noexcept = default;

        [[nodiscard]] constexpr auto in() const noexcept { return m_in; }
        [[nodiscard]] constexpr auto out() const noexcept { return m_out; }
        [[nodiscard]] constexpr auto weight() const noexcept { return m_weight; }
        [[nodiscard]] constexpr auto enabled() const noexcept { return m_enabled; }
        [[nodiscard]] constexpr auto innovation() const noexcept { return m_innovation; }

        constexpr void set_in(const innovation_t value) noexcept { m_in = value; }
        constexpr void set_out(const innovation_t value) noexcept { m_out = value; }
        constexpr void set_weight(const float value) noexcept { m_weight = value; }
        constexpr void set_enabled(const bool value) noexcept { m_enabled = value; }

        void mutate_weight(const Config &cfg)
        {
            const auto random = Random::canonical();
            if (random < cfg.mutate_redraw_weight)
            {
                m_weight = Random::weight();
                return;
            }
            m_weight += Random::gaussian() / cfg.mutate_weight_divisor;
            // m_weight = std::clamp(m_weight, -1.0f, 1.0f);

        }
    };
} // End namespace neat.