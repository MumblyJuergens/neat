#pragma once

#include <algorithm>
#include <ranges>
#include <utility>
#include <vector>
#include "neat/Genome.hpp"
#include "neat/Synapse.hpp"
#include "neat/Brain.hpp"

namespace neat
{
    class [[nodiscard]] Species final
    {
        static inline int s_id{};
        int m_id{ ++s_id };
        Brain m_representative;
        int m_staleness{};
        int m_age{};
        real_t m_total_fitness{};
        real_t m_total_adjusted_fitness{};
        real_t m_max_fitness{};
        real_t m_max_fitness_record{};
        int m_size{ 1 };

        public:

        [[nodiscard]] constexpr auto id() const noexcept { return m_id; }
        [[nodiscard]] constexpr auto &representative() const noexcept { return m_representative; }
        [[nodiscard]] constexpr auto staleness() const noexcept { return m_staleness; }
        [[nodiscard]] constexpr auto age() const noexcept { return m_age; }
        [[nodiscard]] constexpr auto total_fitness() const noexcept { return m_total_fitness; }
        [[nodiscard]] constexpr auto average_fitness() const noexcept { return m_total_fitness / static_cast<real_t>(m_size); }
        [[nodiscard]] constexpr auto adjusted_fitness() const noexcept { return m_total_adjusted_fitness / static_cast<real_t>(m_size); }
        [[nodiscard]] constexpr auto max_fitness() const noexcept { return m_max_fitness; }
        [[nodiscard]] constexpr auto size() const noexcept { return m_size; }

        constexpr void set_representative(const Brain &value) noexcept { m_representative = value; }
        constexpr void set_max_fitness(const real_t value) noexcept { m_max_fitness = value; }
        constexpr void set_size(const int value) noexcept { m_size = value; }

        constexpr void increment_size() noexcept { ++m_size; }
        constexpr void test_max_fitness(const real_t value) noexcept { m_max_fitness = std::max(value, m_max_fitness); }
        constexpr void increase_total_fitness(const real_t value) noexcept { m_total_fitness += value; }
        constexpr void increase_total_adjusted_fitness(const real_t value) noexcept { m_total_adjusted_fitness += value; }


        [[nodiscard]] Species(const Brain &representative) noexcept
            : m_representative{ representative }
        {
        }

        constexpr void new_generation() noexcept
        {
            m_size = 0;
            m_total_fitness = 0;
            m_total_adjusted_fitness = 0;
        }

        constexpr void age_gracefully() noexcept
        {
            if (m_max_fitness > m_max_fitness_record)
            {
                m_max_fitness_record = m_max_fitness;
                m_staleness = 0;
            }
            else
            {
                ++m_staleness;
            }
            ++m_age;
        }

    };
} // End namespace neat.