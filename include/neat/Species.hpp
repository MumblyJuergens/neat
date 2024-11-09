#pragma once

#include <algorithm>
#include <ranges>
#include <utility>
#include <vector>
#include "neat/Genome.hpp"
#include "neat/Synapse.hpp"

namespace neat
{
    class [[nodiscard]] Species final
    {
        std::vector<Genome *> m_members;
        float m_average_fitness{};
        std::vector<Synapse> m_representative;
        int m_staleness{};
        int m_age{};
        static inline std::size_t s_id{};
        std::size_t m_id{ s_id++ };
        float m_max_fitness{};

        public:

        [[nodiscard]] constexpr Species(Genome *first) noexcept
            : m_members{ first }, m_representative{ first->synapses() }
        {
        }

        constexpr void clear() noexcept
        {
            m_members.clear();
            m_average_fitness = {};
        }

        constexpr void add(Genome *genome) noexcept { m_members.push_back(genome); }

        constexpr void sort_descending()
        {
            std::ranges::sort(m_members, std::greater{}, &Genome::fitness);
        }

        constexpr void calc_fitness()
        {
            if (m_members.size() == 0) return;
            m_average_fitness = mj::sum(m_members, &Genome::fitness) / static_cast<float>(m_members.size());
            const auto newMax = std::ranges::max(m_members, {}, &Genome::fitness)->fitness();
            if (newMax > m_max_fitness)
            {
                m_max_fitness = newMax;
                m_staleness = 0;
            }
        }

        constexpr void age_gracefully() noexcept
        {
            ++m_staleness;
            ++m_age;
        }

        [[nodiscard]] Genome generate_new(const Config &cfg, std::function<std::shared_ptr<Simulation>()> simulation_factory, InnovationHistory &innovationHistory) const noexcept
        {
            const float random = Random::canonical();
            if (random < cfg.mutate_just_clone_rate)
            {
                return select_genome()->clone(simulation_factory());
            }

            auto *parent0 = select_genome();
            auto *parent1 = select_genome();
            auto [worst, best] = std::ranges::minmax(parent0, parent1, std::less{}, &Genome::fitness);
            auto child = best->crossover(worst, simulation_factory(), cfg);
            child.mutate(cfg, innovationHistory);
            return child;
        }

        [[nodiscard]] const Genome *select_genome() const
        {
            const auto index = static_cast<std::size_t>((1.0f - Random::canonical_skewed(4.0f)) * static_cast<float>(m_members.size() - 1u));
            return m_members.at(index);
        }


        //[[nodiscard]] constexpr auto &members() const noexcept { return m_members; }
        [[nodiscard]] constexpr auto &representative() const noexcept { return m_representative; }
        [[nodiscard]] constexpr auto &champ() const noexcept { return m_members.front(); }
        [[nodiscard]] constexpr auto average_fitness() const noexcept { return m_average_fitness; }
        [[nodiscard]] constexpr auto staleness() const noexcept { return m_staleness; }
        [[nodiscard]] constexpr auto age() const noexcept { return m_age; }
        [[nodiscard]] constexpr auto size() const noexcept { return m_members.size(); }
        [[nodiscard]] constexpr auto id() const noexcept { return m_id; }
        [[nodiscard]] constexpr auto max_fitness() const noexcept { return m_max_fitness; }

        constexpr void set_representative() noexcept { if (size() > 0) m_representative = champ()->synapses(); }
    };
} // End namespace neat.