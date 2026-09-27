#pragma once

#include "neat/Brain.hpp"
#include "neat/types.hpp"
#include "neat_export.h"
#include <cereal/types/memory.hpp>
#include <cstdint>
#include <vector>

namespace neat
{
class SimulationInfo;

class [[nodiscard]] NEAT_EXPORT Genome final
{
    static constexpr uint8_t DONE_FLAG = 1, PERFFECT_FLAG = 2;

    Brain m_brain;
    real_t m_fitness{};
    real_t m_adjusted_fitness{};
    int m_species{};
    int m_id;
    uint8_t m_flags{};

  public:
    [[nodiscard]] Genome(int _id) noexcept : m_id{_id} {}

    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_brain, m_fitness, m_adjusted_fitness, m_flags, m_species, m_id);
    }

    constexpr Genome(const Genome &) = delete;
    constexpr Genome &operator=(const Genome &) = delete;
    Genome(Genome &&other) noexcept = default;
    Genome &operator=(Genome &&other) noexcept = default;

    template <typename Self>
    [[nodiscard]] constexpr auto &&brain(this Self &&self) noexcept
    {
        return self.m_brain;
    }
    [[nodiscard]] constexpr auto fitness() const noexcept { return m_fitness; }
    [[nodiscard]] constexpr auto adjusted_fitness() const noexcept { return m_adjusted_fitness; }
    [[nodiscard]] constexpr auto simulation_is_done() const noexcept { return (m_flags & DONE_FLAG) != 0; }
    [[nodiscard]] constexpr auto simulation_is_perfect() const noexcept { return (m_flags & PERFFECT_FLAG) != 0; }
    [[nodiscard]] constexpr auto species() const noexcept { return m_species; }
    [[nodiscard]] auto id() const noexcept { return m_id; }

    constexpr void set_fitness(const real_t value) noexcept { m_fitness = value; }
    constexpr void set_adjusted_fitness(const real_t value) noexcept { m_adjusted_fitness = value; }
    constexpr void set_simulation_is_done(const bool value) noexcept
    {
        if (value) m_flags |= DONE_FLAG;
        else m_flags &= static_cast<uint8_t>(~DONE_FLAG);
    }
    constexpr void set_species(const int value) noexcept { m_species = value; }

    void simple_step(const std::vector<real_t> &inputs, std::vector<real_t> &outputs, activator_f *activator);
};

inline void swap(Genome &a, Genome &b)
{
    Genome c(std::move(a));
    a = std::move(b);
    b = std::move(c);
}

} // namespace neat