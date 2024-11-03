#pragma once

#include <cassert>
#include <algorithm>
#include <array>
#include <numbers>
#include <span>
#include <vector>
#include "neat/Synapse.hpp"
#include "neat/Neuron.hpp"
#include "neat/innovation.hpp"
#include "neat/Random.hpp"
#include "neat/Config.hpp"

namespace neat
{
    class [[nodiscard]] Genome final
    {
        std::vector<Synapse> m_synapses;
        std::vector<Neuron> m_hidden_neurons;
        std::vector<Neuron *> m_neurons;
        float m_fitness{};

        [[nodiscard]] static constexpr float sigmoid(float x) noexcept { return 1.0f / (1.0f + std::powf(std::numbers::e_v<float>, -4.9f * x)); }

    public:
        [[nodiscard]] constexpr Genome(std::span<Neuron> inputs, std::span<Neuron> outputs) noexcept
        {
            for (auto &input : inputs)
            {
                m_neurons.push_back(&input);
            }
            for (auto &output : outputs)
            {
                m_neurons.push_back(&output);
            }

            for (auto i = 0uz; i < inputs.size(); ++i)
            {
                for (auto o = 0uz; o < outputs.size(); ++o)
                {
                    m_synapses.emplace_back(inputs[i].number(), outputs[o].number(), Random::weight(), true, next_global_innovation_number());
                }
            }
        }

        [[nodiscard]] constexpr Genome clone_randomise_weights() const noexcept
        {
            Genome genome = *this;
            for (auto &synapse : genome.m_synapses)
            {
                synapse.set_weight(Random::weight());
            }
            return genome;
        }

        template <typename Self>
        [[nodiscard]] constexpr auto &&synapses(this Self &&self) noexcept { return self.m_synapses; }
        [[nodiscard]] constexpr auto fitness() const noexcept { return m_fitness; }

        constexpr void set_fitness(const float value) noexcept { m_fitness = value; }

        [[nodiscard]] constexpr float difference(const Genome &other, const Config &cfg) const noexcept
        {
            const std::size_t maxGenomeCount = std::max(m_synapses.size(), other.m_synapses.size());
            std::array synapse{m_synapses.cbegin(), other.m_synapses.cbegin()};
            std::array sentinal{m_synapses.end(), other.m_synapses.end()};

            std::size_t disjoint{};
            std::size_t excess{};
            float weightDifference{};
            std::size_t matching{};

            while (synapse[0] != sentinal[0] && synapse[1] != sentinal[1])
            {
                if (synapse[0] == sentinal[0])
                {
                    ++synapse[1];
                    ++excess;
                }
                else if (synapse[1] == sentinal[1])
                {
                    ++synapse[0];
                    ++excess;
                }
                else
                {
                    const std::array innovation{synapse[0]->innovation(), synapse[1]->innovation()};
                    if (innovation[0] == innovation[1])
                    {
                        ++matching;
                        weightDifference += mj::difference(synapse[0]->weight(), synapse[1]->weight());
                        ++synapse[0];
                        ++synapse[1];
                    }
                    else if (innovation[0] < innovation[1])
                    {
                        ++synapse[0];
                        ++disjoint;
                    }
                    else // if (innovation[1] < innovation[0])
                    {
                        ++synapse[1];
                        ++disjoint;
                    }
                }
            }

            const float divisor = cfg.absolute_difference ? 1.0f : static_cast<float>(maxGenomeCount);

            return (cfg.disjoint_coefficient * (static_cast<float>(disjoint) / divisor) +
                    cfg.excess_coefficient * (static_cast<float>(excess) / divisor) +
                    cfg.weight_difference_coefficent * (weightDifference / static_cast<float>(matching)));
        }
    };
} // End namespace neat.