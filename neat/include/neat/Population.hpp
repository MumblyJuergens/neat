#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <print>
#include <ranges>
#include <sstream>
#include <vector>
#include <mj/algorithm.hpp>
#include <mj/math.hpp>
#include <mj/size.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include "neat/neat_export.h"
#include "neat/Genome.hpp"
#include "neat/Config.hpp"
#include "neat/Simulation.hpp"
#include "neat/SimulationFactory.hpp"
#include "neat/Species.hpp"
#include "neat/UserData.hpp"

namespace neat
{
    class [[nodiscard]] Population final
    {
        std::vector<Genome> m_genomes;
        std::vector<Species> m_species;
        Config cfg;
        bool m_generation_is_done{};
        std::unique_ptr<SimulationFactory> m_simulation_factory;
        int m_population_size;
        real_t m_max_fitness{};
        int m_generation{};
        std::function<void(std::string)> m_stats_string_handler;
        Brain m_champ;
        int m_champ_id{};
        bool m_finished{};
        UserData *const m_user_data;
        real_t m_generation_max_fitness{};


        public:
        // Methods for giving information about progress.
        [[nodiscard]] constexpr auto species_count() const noexcept { return mj::isize(m_species); }
        [[nodiscard]] constexpr auto population_count() const noexcept { return mj::isize(m_genomes); }
        [[nodiscard]] constexpr auto max_fitness() const noexcept { return m_max_fitness; }
        [[nodiscard]] constexpr auto generation_is_done() const noexcept { return m_generation_is_done; }
        [[nodiscard]] constexpr auto &champ() const noexcept { return m_champ; }
        [[nodiscard]] constexpr auto champ_id() const noexcept { return m_champ_id; }
        [[nodiscard]] constexpr auto generation() const noexcept { return m_generation; }
        [[nodiscard]] constexpr auto finished() const noexcept { return m_finished; }
        [[nodiscard]] constexpr auto &user_data() const noexcept { return m_user_data; }

        // Exposing inner magic to allow stats gathering.
        [[nodiscard]] constexpr auto &genomes() const noexcept { return m_genomes; }
        [[nodiscard]] constexpr auto &species() const noexcept { return m_species; }

        void set_stats_string_handler(std::function<void(std::string)> f) { m_stats_string_handler = f; }
        private:

        std::string latest_stats_output() const noexcept
        {
            std::ostringstream stream;
            std::println(stream, "Statistics | Generation {:6} | Population {:6} | Max Fitness {:6} | SCT: {:6}\n----------------------------------------------------------------------------------------\nSpecies | Members | Staleness | Age   | Avg Fitness | Max Fitness | CNodes | CConns", m_generation, m_genomes.size(), m_max_fitness, cfg.species_compatability_threshold);
            for (const auto &s : m_species) { std::println(stream, "{:7} | {:7} | {:9} | {:5} | {:11} | {:11} | {:6} | {:6}", s.id(), s.size(), s.staleness(), s.age(), s.average_fitness(), s.max_fitness(), s.representative().neuron_count(), s.representative().synapse_count()); }
            return stream.str();
        }

        void build_population(std::vector<Genome> &pop, const Init init)
        {
            for (int i{}; i < m_population_size; ++i)
            {
                pop.emplace_back(m_simulation_factory->create_simulation()).brain().init(cfg, init); // TODO: no init?
            }
        }

        const Genome &brain_roulette(const Species &specie)
        {
            const auto rand = Random::range(specie.total_fitness());
            real_t runningSum{};
            [[maybe_unused]] int sanity{};
            for (const auto &genome : m_genomes | mj::filter(&Genome::species, std::equal_to{}, specie.id()))
            {
                ++sanity;
                runningSum += genome.fitness();
                if (runningSum > rand)
                {
                    return genome;
                }
            }
            // Should be unreachable...
            exit(1);
        }

        public:
        [[nodiscard]] Population(std::unique_ptr<SimulationFactory> simulationFactory, const Config &cfg = {}, UserData *const userData = nullptr) noexcept
            : cfg{ cfg }, m_simulation_factory{ std::move(simulationFactory) }, m_population_size{ cfg.setup_population_size }, m_user_data{ userData }
        {
            build_population(m_genomes, Init::yes);
            m_champ.init(cfg, Init::yes);
        }

        void reset_champ()
        {
            m_champ.init(cfg, Init::yes);
            m_champ_id = 0;
            m_max_fitness = 0;
        }

        template<typename Archive>
        void serialize(Archive &ar)
        {
            ar(m_genomes, m_species, cfg, m_generation_is_done, m_population_size,
                m_generation, m_finished, m_generation_max_fitness);
            if (cfg.tweak_serialize_simulation_factory)
            {
                ar(m_simulation_factory);
            }
            Genome::serialize_static(ar);
            Species::serialize_static(ar);
            InnovationHistory::serialize_static(ar);
        }

        constexpr void step() noexcept
        {
            // m_max_fitness = 0.0f;
            int doneCount{};
            int doneDone{};
            for (auto &genome : m_genomes)
            {
                if (!genome.simulation_is_done())
                {
                    genome.step(m_user_data);
                }
                else
                {
                    genome.skip(m_user_data);
                }
                doneCount += genome.simulation_is_done() ? 1 : 0;
                ++doneDone;
                if (genome.fitness() > m_generation_max_fitness)
                {
                    m_generation_max_fitness = genome.fitness();
                    genome.make_current_champ();
                    m_stats_string_handler(std::format("Generation max: {}\r", m_generation_max_fitness));
                }
                if (genome.fitness() > m_max_fitness)
                {
                    m_max_fitness = genome.fitness();
                    m_champ = genome.brain();
                    ++m_champ_id;
                    if (genome.simulation_is_perfect())
                    {
                        m_finished = true;
                    }
                }
            }
            if (doneCount == doneDone)
            {
                m_generation_is_done = true;
            }
        }

        void new_generation()
        {
            m_generation_is_done = false;
            m_generation_max_fitness = {};

            // Speciate.
            // Crossover.
            // Mutate.

            // Speciate.
            std::ranges::sort(m_genomes, std::greater{}, &Genome::fitness);
            std::ranges::for_each(m_species, [](Species &s) { s.new_generation(); });
            for (Genome &genome : m_genomes)
            {
                bool found{};
                for (auto &specie : m_species)
                {
                    const auto difference = genome.brain().difference(specie.representative(), cfg);
                    if (difference < cfg.species_compatability_threshold)
                    {
                        genome.set_species(specie.id());
                        specie.increment_size();
                        specie.test_max_fitness(genome.fitness());
                        specie.increase_total_fitness(genome.fitness());
                        found = true;
                        break;
                    }
                }
                // All alone in this world :(
                if (!found) genome.set_species(m_species.emplace_back(genome.brain()).id());
            }
            // Stupid erase_if can't use views to drop the first one grrr.
            m_species.erase(std::remove_if(std::next(m_species.begin()), m_species.end(), [this](const Species &s) { return s.staleness() > cfg.species_maximum_staleness; }), m_species.end());
            std::erase_if(m_species, mj::magic_lambda(&Species::size, std::equal_to{}, 0));
            real_t speciesAverageTotal{};
            int speciesPopulationCount{};
            for (auto &specie : m_species)
            {
                specie.age_gracefully();
                for (auto &genome : m_genomes | mj::filter(&Genome::species, std::equal_to{}, specie.id()))
                {
                    if (cfg.crossover_use_adjusted_fitness)
                    {
                        const auto adjustedFitness = genome.fitness() / static_cast<real_t>(specie.size());
                        speciesAverageTotal += adjustedFitness;
                        specie.increase_total_adjusted_fitness(adjustedFitness);
                    }
                    else
                    {
                        speciesAverageTotal += genome.fitness();
                    }
                    ++speciesPopulationCount;
                }
            }
            speciesAverageTotal /= static_cast<real_t>(speciesPopulationCount);

            std::vector<Genome> children;

            if (m_stats_string_handler)
            {
                m_stats_string_handler(latest_stats_output());
            }

            // Crossover & Mutate.
            for (auto &specie : m_species)
            {
                int eliteCopied{};
                if (specie.size() > cfg.crossover_elite_size)
                {
                    eliteCopied = 1;
                    children.emplace_back(m_simulation_factory->create_simulation()).brain() = std::ranges::find(m_genomes, specie.id(), &Genome::species)->brain();
                }
                const real_t averageSpeciesFitness = cfg.crossover_use_adjusted_fitness ? specie.adjusted_fitness() : specie.average_fitness();
                const auto childCount = static_cast<int>((averageSpeciesFitness / speciesAverageTotal) * static_cast<real_t>(specie.size())) - eliteCopied;
                mj::loop(childCount, [&]() {
                    auto &parent0 = brain_roulette(specie);
                    auto &parent1 = brain_roulette(specie);
                    auto [worst, best] = std::ranges::minmax(parent0, parent1, std::less{}, &Genome::fitness);
                    auto child = Brain::crossover(best.brain(), worst.brain(), cfg);
                    child.mutate(cfg);
                    children.emplace_back(m_simulation_factory->create_simulation()).brain() = child;
                    });
            }

            for (auto i{ mj::isize(children) }; i < m_population_size; ++i)
            {
                children.emplace_back(m_simulation_factory->create_simulation()).brain().init(cfg, Init::yes);
            }

            if (mj::isize(m_species) > cfg.species_count_target)
            {
                cfg.species_compatability_threshold += cfg.species_compatability_modifier;
            }
            else if (mj::isize(m_species) < cfg.species_count_target)
            {
                cfg.species_compatability_threshold -= cfg.species_compatability_modifier;
            }

            m_genomes.swap(children);
            ++m_generation;
        }
    };
} // End namesapce neat.