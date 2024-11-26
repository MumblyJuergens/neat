#pragma once

#include <any>
#include <memory>
#include "neat/neat_export.h"
#include "neat/types.hpp"
#include "neat/Brain.hpp"

namespace neat
{
    class Simulation;
    class SimulationInfo;

    class [[nodiscard]] NEAT_EXPORT Genome final
    {
        Brain m_brain;
        real_t m_fitness{};
        real_t m_adjusted_fitness{};
        std::shared_ptr<Simulation> m_simulation;
        bool m_sim_is_done{};
        bool m_sim_is_perfect{};
        int m_species{};
        static inline int s_id{};
        static inline int s_champ_id{};
        int m_id{ ++s_id };

        [[nodiscard]] Genome(const Genome &other, std::shared_ptr<Simulation> freshSimulation) noexcept
            : m_brain{ other.m_brain },
            m_simulation{ freshSimulation }
        {
        }

        public:
        [[nodiscard]] Genome(std::shared_ptr<Simulation> simulation) noexcept
            : m_simulation{ std::move(simulation) } {
        }

        constexpr Genome(const Genome &) = delete;
        constexpr Genome &operator=(const Genome &) = delete;
        constexpr Genome(Genome &&other) noexcept = default;
        constexpr Genome &operator=(Genome &&other) noexcept = default;

        template <typename Self>
        [[nodiscard]] constexpr auto &&brain(this Self &&self) noexcept { return self.m_brain; }
        [[nodiscard]] constexpr auto fitness() const noexcept { return m_fitness; }
        [[nodiscard]] constexpr auto adjusted_fitness() const noexcept { return m_adjusted_fitness; }
        [[nodiscard]] constexpr auto simulation_is_done() const noexcept { return m_sim_is_done; }
        [[nodiscard]] constexpr auto simulation_is_perfect() const noexcept { return m_sim_is_perfect; }
        [[nodiscard]] constexpr auto &simulation() const noexcept { return *m_simulation; }
        [[nodiscard]] constexpr auto species() const noexcept { return m_species; }
        [[nodiscard]] constexpr auto is_current_champ() const noexcept { return m_id == s_champ_id; }

        constexpr void set_adjusted_fitness(const real_t value) noexcept { m_adjusted_fitness = value; }
        constexpr void set_simulation_is_done(const bool value) noexcept { m_sim_is_done = value; }
        constexpr void set_species(const int value) noexcept { m_species = value; }
        constexpr void make_current_champ() noexcept { s_champ_id = m_id; }

        void step(std::any *const userData);
        void step(SimulationInfo &info, activator_f *activator);
        void skip(SimulationInfo &info);
        void skip(std::any *const userData);

    };

    inline void swap(Genome &a, Genome &b)
    {
        Genome c(std::move(a));
        a = std::move(b);
        b = std::move(c);
    }
} // End namespace neat.