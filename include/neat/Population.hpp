#pragma once

#include <functional>
#include <memory>
#include <numeric>
#include <print>
#include <ranges>
#include <sstream>
#include <vector>
#include <mj/algorithm.hpp>
#include <mj/math.hpp>
#include "neat/neat_export.h"
#include "neat/Genome.hpp"
#include "neat/Config.hpp"
#include "neat/log.hpp"
#include "neat/Simulation.hpp"
#include "neat/Species.hpp"

namespace neat
{
    class [[nodiscard]] Population final
    {
        using population_t = std::vector<Genome>;
        using species_t = std::vector<Species>;

        population_t m_population;
        species_t m_species;
        Config cfg;
        // float m_max_fitness{};
        bool m_generation_is_done{};
        std::function<std::shared_ptr<Simulation>()> m_simulation_factory;
        const std::size_t m_population_size;
        float m_max_fitness{};
        std::size_t m_generation{};
        std::function<void(std::string)> m_stats_string_handler;
        InnovationHistory innovationHistory;

        public:
        // Methods for giving information about progress.
        [[nodiscard]] constexpr auto species_count() const noexcept { return m_species.size(); }
        [[nodiscard]] constexpr auto population_count() const noexcept { return m_population.size(); }
        [[nodiscard]] constexpr auto max_fitness() const noexcept { return m_max_fitness; }
        [[nodiscard]] constexpr auto generation_is_done() const noexcept { return m_generation_is_done; }
        [[nodiscard]] constexpr auto &champ() const noexcept { return m_population.at(0); }
        [[nodiscard]] constexpr auto generation() const noexcept { return m_generation; }
        private:

        constexpr void speciate(Genome &genome) noexcept
        {
            for (auto &specie : m_species)
            {
                const auto difference = genome.difference(specie.representative(), cfg);
                if (difference < cfg.species_compatability_threshold)
                {
                    specie.add(&genome);
                    return;
                }
            }
            // All alone in this world :(
            m_species.emplace_back(&genome);
        }

        std::string latest_stats_output() const noexcept
        {
            std::ostringstream stream;
            std::println(stream, "Statistics | Generation {:6} | Population {:6} | Max Fitness {:6} | SCT: {:6}\n----------------------------------------------------------------------------------------\nSpecies | Members | Staleness | Age   | Avg Fitness | Max Fitness | CNodes | CConns", m_generation, m_population.size(), m_max_fitness, cfg.species_compatability_threshold);
            for (const auto &s : m_species) { std::println(stream, "{:7} | {:7} | {:9} | {:5} | {:11} | {:11} | {:6} | {:6}", s.id(), s.size(), s.staleness(), s.age(), s.average_fitness(), s.max_fitness(), s.champ()->neurons().size(), s.champ()->synapses().size()); }
            return stream.str();
        }

        public:
        [[nodiscard]] Population(const innovation_t inputCount, const innovation_t outputCount, const std::size_t populationSize, std::function<std::shared_ptr<Simulation>()> simulationFactory, const Config &cfg) noexcept
            : cfg{ cfg }, m_simulation_factory{ simulationFactory }, m_population_size{ populationSize }
        {
            Genome first{ inputCount, outputCount, simulationFactory() };
            if (cfg.fully_connect)
            {
                first.fully_connect();
            }
            if (cfg.connect_bias)
            {
                first.connect_bias(cfg.bias_input);
            }
            for (std::size_t i = 0; i < populationSize - 1; ++i)
            {
                m_population.emplace_back(first.clone(simulationFactory()));
            }
            m_population.emplace_back(std::move(first));
        }


        constexpr void set_stats_string_handler(std::function<void(std::string)> f) { m_stats_string_handler = f; }

        constexpr void step() noexcept
        {
            // m_max_fitness = 0.0f;
            std::size_t doneCount{};
            std::size_t doneDone{};
            for (auto &genome : m_population)
            {
                if (!genome.simulation_is_done())
                {
                    genome.step();
                    doneCount += genome.simulation_is_done() ? 1 : 0;
                }
                ++doneDone;
                m_max_fitness = std::max(m_max_fitness, genome.fitness());
            }
            if (doneCount == doneDone)
            {
                m_generation_is_done = true;
            }
        }

        [[nodiscard]] std::ptrdiff_t children_for_specie(const Species &specie, const float averagePopulationFitness) const noexcept
        {
            const auto numKids = static_cast<std::ptrdiff_t>(specie.average_fitness() / averagePopulationFitness * static_cast<float>(specie.size()));
            return std::max(numKids, static_cast<std::ptrdiff_t>(cfg.species_minimum_size));
        }

        constexpr void new_generation()
        {
            m_generation_is_done = false;
            std::ranges::sort(m_population, std::greater{}, &Genome::fitness);
            std::ranges::for_each(m_population, [this](auto &g) { speciate(g); });
            std::ranges::for_each(m_species, &Species::age_gracefully);
            std::ranges::for_each(m_species, &Species::calc_fitness);
            std::ranges::sort(m_species, std::greater{}, &Species::average_fitness);
            const auto averagePopulationFitness = mj::sum(m_population, &Genome::fitness) / static_cast<float>(m_population_size);
            // Yeah I hate this too but...
            m_species.erase(std::remove_if(std::next(m_species.begin()), m_species.end(), [this](const Species &s) { return s.staleness() > cfg.species_maximum_staleness; }), m_species.end());
            std::erase_if(m_species, mj::magic_lambda(&Species::size, std::equal_to{}, 0));
            std::ranges::for_each(m_species, &Species::set_representative);

            population_t children;


            if (m_stats_string_handler)
            {
                m_stats_string_handler(latest_stats_output());
            }

            for (auto &specie : m_species | mj::filter(&Species::size, std::greater{}, 0))
            {
                children.push_back(specie.champ()->clone(m_simulation_factory()));
                const std::ptrdiff_t childCount = children_for_specie(specie, averagePopulationFitness) - 1; // Champ went first.
                mj::loop(childCount, [&]() { children.push_back(specie.generate_new(cfg, m_simulation_factory, innovationHistory)); });
            }

            while (children.size() < m_population_size) {
                children.push_back(m_species.front().generate_new(cfg, m_simulation_factory, innovationHistory));
            }

            if (m_species.size() > cfg.species_count_target)
            {
                cfg.species_compatability_threshold += cfg.species_compatability_modifier;
            }
            else if (m_species.size() < cfg.species_count_target)
            {
                cfg.species_compatability_threshold -= cfg.species_compatability_modifier;
            }

            m_population.swap(children);
            std::ranges::for_each(m_species, &Species::clear);
            ++m_generation;
        }
    };
} // End namesapce neat.