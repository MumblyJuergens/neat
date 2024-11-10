#pragma once

#include <cassert>
#include <algorithm>
#include <array>
#include <forward_list>
#include <memory>
#include <numbers>
#include <ranges>
#include <span>
#include <sstream>
#include <vector>
#include <mj/math.hpp>
#include <mj/algorithm.hpp>
#include "neat/Synapse.hpp"
#include "neat/Neuron.hpp"
#include "neat/innovation.hpp"
#include "neat/Random.hpp"
#include "neat/Config.hpp"
#include "neat/flag_types.hpp"
#include "neat/InnovationHistory.hpp"

namespace neat
{
    class Simulation;
    class SimulationInfo;

    class [[nodiscard]] Genome final
    {
        std::vector<Synapse> m_synapses;
        std::vector<Neuron> m_neurons;
        float m_fitness{};
        std::shared_ptr<Simulation> m_simulation;
        bool m_sim_is_done{};
        std::size_t m_layer_count{ 2 };

        [[nodiscard]] static float sigmoid(float x) noexcept
        {
            return std::tanhf(x);
            // return 1.0f / (1.0f + std::powf(std::numbers::e_v<float>, -4.9f * x));
        }

        [[nodiscard]] Genome(const Genome &other, std::shared_ptr<Simulation> freshSimulation) noexcept
            : m_synapses{ other.m_synapses },
            m_neurons{ other.m_neurons },
            // m_fitness{ other.m_fitness },
            m_simulation{ freshSimulation },
            m_layer_count{ other.m_layer_count }
        {
        }

        [[nodiscard]] Genome(const Genome &best, const Genome &worst, std::shared_ptr<Simulation> freshSimulation, const Config &cfg, crossover_t) noexcept
            :
            // m_synapses{ other.m_synapses },
            m_neurons{ best.m_neurons },
            // m_fitness{ other.m_fitness },
            m_simulation{ freshSimulation },
            m_layer_count{ best.m_layer_count }
        {
            for (const auto &synapse : best.m_synapses)
            {
                bool enabled = true;

                const auto matchingSynapse = std::ranges::find_if(worst.m_synapses, mj::magic_callable(&Synapse::innovation, std::equal_to{}, synapse.innovation()));

                // Disjoint or excess gene, comes from best.
                if (matchingSynapse == worst.m_synapses.end())
                {
                    assert(synapse.in() < m_neurons.size());
                    assert(synapse.out() < m_neurons.size());
                    m_synapses.push_back(synapse);
                    continue;
                }

                // Matching, chance of re-enabling.
                assert(synapse.in() == matchingSynapse->in());
                assert(synapse.out() == matchingSynapse->out());
                if (!synapse.enabled() || !matchingSynapse->enabled())
                {
                    if (Random::canonical() < cfg.mutate_disable_node_rate)
                    {
                        enabled = false;
                    }
                }
                if (Random::canonical() < 0.5f)
                {
                    assert(synapse.in() < m_neurons.size());
                    assert(synapse.out() < m_neurons.size());
                    m_synapses.push_back(synapse);
                }
                else
                {
                    assert(matchingSynapse->in() < m_neurons.size());
                    assert(matchingSynapse->out() < m_neurons.size());
                    m_synapses.push_back(*matchingSynapse);
                }
                m_synapses.back().set_enabled(enabled);
            }
        }

        [[nodiscard]] constexpr bool is_fully_connected() const noexcept
        {
            std::vector<std::size_t> neuronsInLayers(m_layer_count);
            std::ranges::for_each(m_neurons, [&neuronsInLayers](std::size_t n) { neuronsInLayers.at(n) += 1; }, &Neuron::layer);
            std::size_t maxConnections{};
            for (std::size_t i{}; i < m_layer_count - 1; ++i)
            {
                std::size_t inFront{};
                for (auto j = i + 1; j < m_layer_count; ++j)
                {
                    inFront += neuronsInLayers[j];
                }
                maxConnections += neuronsInLayers[i] * inFront;
            }
            return maxConnections == m_synapses.size();
        }

        void add_connection(const innovation_t in, const innovation_t out, const innovation_t innovation)
        {
            if (in == out) return;
            if (m_neurons.at(in).layer() >= m_neurons.at(out).layer()) return;
            if (std::ranges::any_of(m_synapses, [in, out](const Synapse &s) { return s.in() == in && s.out() == out; })) return;
            m_synapses.emplace_back(in, out, Random::weight(), innovation);
        }

        void add_connection(const innovation_t in, const innovation_t out, InnovationHistory &innovationHistory)
        {
            if (in == out) return;
            if (m_neurons.at(in).layer() >= m_neurons.at(out).layer()) return;
            if (std::ranges::any_of(m_synapses, [in, out](const Synapse &s) { return s.in() == in && s.out() == out; })) return;
            const auto innovation = innovationHistory.get_innovation_number(this, in, out);
            m_synapses.emplace_back(in, out, Random::weight(), innovation);
        }

        void add_connection(InnovationHistory &innovationHistory) noexcept
        {
            if (is_fully_connected()) return;

            std::vector<std::pair<innovation_t, innovation_t>> possibilities;
            for (std::size_t layer{}; layer < m_layer_count - 1; ++layer)
            {
                for (const auto &neuronIn : m_neurons | mj::filter(&Neuron::layer, std::equal_to{}, layer))
                {
                    for (const auto &neuronOut : m_neurons | mj::filter(&Neuron::layer, std::greater{}, layer))
                    {
                        if (std::ranges::any_of(m_synapses, [&](const Synapse &s) { return s.in() == neuronIn.number() && s.out() == neuronOut.number(); })) continue;
                        if (neuronIn.number() == neuronOut.number()) continue;
                        possibilities.emplace_back(neuronIn.number(), neuronOut.number());
                    }
                }
            }
            const auto &conn = Random::item(possibilities);
            assert(conn.first < m_neurons.size());
            assert(conn.second < m_neurons.size());
            add_connection(conn.first, conn.second, innovationHistory);
        }

        constexpr void add_node(InnovationHistory &innovationHistory) noexcept
        {
            if (m_synapses.size() == 0)
            {
                add_connection(innovationHistory);
                return;
            }

            Synapse &oldSynapse = Random::item(m_synapses);
            oldSynapse.set_enabled(false);
            const auto oldIn = oldSynapse.in();
            const auto oldOut = oldSynapse.out();
            const auto oldWeight = oldSynapse.weight();
            // Do *not* use oldSynapse below here!

            const auto newId = m_neurons.size();
            auto &neuron = m_neurons.emplace_back(newId, NeuronType::hidden);
            assert(oldIn < m_neurons.size());
            assert(oldOut < m_neurons.size());
            const auto innovation0 = innovationHistory.get_innovation_number(this, oldIn, neuron.number());
            const auto innovation1 = innovationHistory.get_innovation_number(this, neuron.number(), oldOut);
            m_synapses.emplace_back(oldIn, neuron.number(), 1.0f, innovation0);
            m_synapses.emplace_back(neuron.number(), oldOut, oldWeight, innovation1);

            neuron.set_layer(m_neurons.at(oldIn).layer() + 1);
            if (neuron.layer() != m_neurons.at(oldOut).layer()) return;


            for (auto &adjustNeuron : m_neurons | mj::filter(&Neuron::layer, std::greater_equal{}, neuron.layer()))
            {
                if (&adjustNeuron == &neuron) continue;
                adjustNeuron.set_layer(adjustNeuron.layer() + 1);
            }
            ++m_layer_count;
        }

        public:
        [[nodiscard]] Genome(const innovation_t inputCount, const innovation_t outputCount, std::shared_ptr<Simulation> simulation) noexcept
            : m_simulation{ std::move(simulation) }

        {
            mj::loop(inputCount, [&](const innovation_t i) { m_neurons.emplace_back(i, NeuronType::input); });
            mj::loop(outputCount, [&](const innovation_t i) { m_neurons.emplace_back(inputCount + i, NeuronType::output); });
            assert(m_neurons.size() == inputCount + outputCount);
        }

        constexpr void fully_connect() noexcept
        {
            assert(!is_fully_connected());
            for (auto const &in : m_neurons | std::views::filter(Neuron::is_input))
            {
                for (auto const &out : m_neurons | std::views::filter(Neuron::is_output))
                {
                    add_connection(in.number(), out.number(), next_global_innovation_number());
                }
            }
            assert(is_fully_connected());
        }

        constexpr void connect_bias(const innovation_t inputNum)
        {
            assert(!is_fully_connected());
            for (auto const &out : m_neurons | std::views::filter(Neuron::is_output))
            {
                add_connection(inputNum, out.number(), next_global_innovation_number());
            }
        }

        constexpr Genome(const Genome &) = delete;
        constexpr Genome &operator=(const Genome &) = delete;
        constexpr Genome(Genome &&other) noexcept = default;
        constexpr Genome &operator=(Genome &&other) noexcept = default;

        [[nodiscard]] Genome clone(std::shared_ptr<Simulation> simulation) const noexcept
        {
            return Genome{ *this, simulation };
        }

        [[nodiscard]] Genome crossover(const Genome *other, std::shared_ptr<Simulation> simulation, const Config &cfg) const noexcept
        {
            return Genome{ *this, *other, simulation, cfg, neat::crossover };
        }

        void mutate(const Config &cfg, InnovationHistory &innovationHistory) noexcept
        {

            // In case we want to mutate from jack for minimal structure.
            if (m_synapses.size() == 0)
            {
                add_connection(innovationHistory);
                return;
            }

            // TODO: Remove nodes/conns.
            const float div = std::max(1.0f, cfg.mutate_new_node_rate + cfg.mutate_new_connection_rate);
            const float rand = Random::canonical();

            if (rand < cfg.mutate_new_node_rate / div)
            {
                add_node(innovationHistory);
            }
            else if (rand < (cfg.mutate_new_node_rate + cfg.mutate_new_connection_rate) / div)
            {
                add_connection(innovationHistory);
            }

            // TODO: Odds...
            std::ranges::for_each(m_synapses, [&cfg](Synapse &s) { s.mutate_weight(cfg); });
        }

        // constexpr void run_neuron(Neuron &neuron, std::vector<int> &done)
        // {
        //     float value{};
        //     for (auto &synapse : m_synapses)
        //     {
        //         if (synapse.enabled() && synapse.out() == neuron.number())
        //         {
        //             auto &in = m_neurons.at(synapse.in());
        //             if (!done.at(synapse.in()))
        //             {
        //                 run_neuron(in, done);
        //             }
        //             value += in.value() * synapse.weight();
        //         }
        //     }
        //     neuron.set_value(sigmoid(value));
        //     done.at(neuron.number()) = true;
        // }

        constexpr void run_network() noexcept
        {
            for (std::size_t i{ 1 }; i < m_layer_count; ++i)
            {
                for (auto &neuron : m_neurons | mj::filter(&Neuron::layer, std::equal_to{}, i))
                {
                    float value = 0;
                    for (const auto &synapse : m_synapses | mj::filter(&Synapse::out, std::equal_to{}, neuron.number()) | std::views::filter(&Synapse::enabled))
                    {
                        value += m_neurons.at(synapse.in()).value() * synapse.weight();
                    }
                    neuron.set_value(sigmoid(value));
                }
            }

            // TODO: Feed-forward instead.
            // std::vector<int> done(m_neurons.size(), false);
            // for (auto &neuron : m_neurons)
            // {
            //     if (neuron.type() == NeuronType::output)
            //     {
            //         run_neuron(neuron, done);
            //     }
            // }
        }

        template <typename Self>
        [[nodiscard]] constexpr auto &&synapses(this Self &&self) noexcept { return self.m_synapses; }
        template <typename Self>
        [[nodiscard]] constexpr auto &&neurons(this Self &&self) noexcept { return self.m_neurons; }
        [[nodiscard]] constexpr auto fitness() const noexcept { return m_fitness; }
        [[nodiscard]] constexpr auto simulation_is_done() const noexcept { return m_sim_is_done; }
        [[nodiscard]] constexpr auto &simulation() const noexcept { return *m_simulation; }

        // constexpr void set_fitness(const float value) noexcept { m_fitness = value; }
        constexpr void set_simulation_is_done(const bool value) noexcept { m_sim_is_done = value; }

        [[nodiscard]] constexpr float difference(const std::vector<Synapse> &rep, const Config &cfg) const noexcept
        {
            const std::size_t maxGenomeCount = std::max(m_synapses.size(), rep.size());
            std::array synapse{ m_synapses.cbegin(), rep.cbegin() };
            std::array sentinal{ m_synapses.end(), rep.end() };

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
                    const std::array innovation{ synapse[0]->innovation(), synapse[1]->innovation() };
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

            const float divisor = maxGenomeCount < 20 || cfg.absolute_difference ? 1.0f : static_cast<float>(maxGenomeCount);

            const float value = (cfg.disjoint_coefficient * (static_cast<float>(disjoint) / divisor) +
                cfg.excess_coefficient * (static_cast<float>(excess) / divisor) +
                cfg.weight_difference_coefficent * (weightDifference / static_cast<float>(matching)));

            return value;
        }

        // constexpr void step() noexcept
        // {
            // const auto inputCount = std::ranges::count_if(m_neurons, Neuron::is_input);
            // const auto outputCount = std::ranges::count_if(m_neurons, Neuron::is_output);
            // std::vector<float> values(static_cast<std::size_t>(inputCount + outputCount), 0.0f);
            // m_simulation->supply(std::span{ values.begin(), static_cast<std::size_t>(inputCount) });

            // for (std::size_t i{}; i < static_cast<std::size_t>(inputCount); ++i)
            // {
            //     m_neurons.at(i).set_value(values.at(i));
            // }
            // run_network();
            // for (auto i{ static_cast<std::size_t>(inputCount) }; i < static_cast<std::size_t>(inputCount + outputCount); ++i)
            // {
            //     values.at(i) = m_neurons.at(i).value();
            // }
            // m_fitness = m_simulation->receive(std::span{ values.begin() + static_cast<std::ptrdiff_t>(inputCount), static_cast<std::size_t>(outputCount) }, m_fitness);
        // }

        NEAT_EXPORT void step();
        NEAT_EXPORT void step(SimulationInfo &info);

        constexpr std::string chart() const noexcept
        {
            std::ostringstream stream;
            for (const auto &synapse : m_synapses | mj::filter(&Synapse::enabled, std::equal_to{}, true))
            {
                std::println(stream, "{} -- {} --> {}", synapse.in(), synapse.weight(), synapse.out());
            }
            return stream.str();
        }

        constexpr std::string code_cpp() const noexcept
        {
            // TODO: code_cpp...
            return "";
        }
    };

    inline void swap(Genome &a, Genome &b)
    {
        Genome c(std::move(a));
        a = std::move(b);
        b = std::move(c);
    }
} // End namespace neat.