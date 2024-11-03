#pragma once

#include <functional>
#include <vector>
#include <mj/math.hpp>
#include "neat/neat_export.h"
#include "neat/Genome.hpp"
#include "neat/Config.hpp"
#include "neat/log.hpp"

namespace neat
{
    class [[nodiscard]] Population final
    {
        using species_t = std::vector<Genome>;
        std::vector<species_t> m_species;
        const Config cfg;

        constexpr void add(Genome genome) noexcept
        {
            for (auto &specie : m_species)
            {
                const auto difference = genome.difference(specie.front(), cfg);
                if (difference < cfg.compatability_threshold)
                {
                    specie.emplace_back(std::move(genome));
                    return;
                }
            }
            // All alone in this world :(
            m_species.push_back(species_t{std::move(genome)});
        }

    public:
        using fitness_f = float(std::span<float> outputs);

        [[nodiscard]] Population(std::span<Neuron> inputs, std::span<Neuron> outputs, const std::size_t populationSize, const Config &cfg) noexcept
            : cfg{cfg}
        {
            Genome genome{inputs, outputs};
            add(genome);
            for (auto i = 1uz; i < populationSize; ++i)
            {
                add(genome.clone_randomise_weights());
            }
        }

        // Methods for giving information about progress.
        [[nodiscard]] constexpr auto species_count() const noexcept { return m_species.size(); }
    };
} // End namesapce neat.