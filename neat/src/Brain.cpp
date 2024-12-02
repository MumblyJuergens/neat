#include <print>
#include <sstream>
#include <utility>
#include <cassert>
#include <mj/algorithm.hpp>
#include <mj/iterator.hpp>
#include <mj/math.hpp>
#include <mj/size.hpp>
#include "neat/Brain.hpp"
#include "neat/Config.hpp"
#include "neat/Neuron.hpp"
#include "neat/Synapse.hpp"

namespace neat
{

    void Brain::init(const Config &cfg, const Init init) noexcept
    {
        m_neurons.clear();
        m_synapses.clear();
        m_layer_count = 2;
        if (init == Init::no) return;
        mj::loop(cfg.setup_input_nodes, [&](const index_t i) { m_neurons.emplace_back(i, NeuronType::input); });
        mj::loop(cfg.setup_output_nodes, [&](const index_t i) { m_neurons.emplace_back(cfg.setup_input_nodes + i, NeuronType::output); });
        assert(mj::isize(m_neurons) == cfg.setup_input_nodes + cfg.setup_output_nodes);
        if (cfg.setup_connect_bias) mj::loop(cfg.setup_output_nodes, [&](const index_t i) { add_connection(cfg.setup_bias_input, cfg.setup_input_nodes + i); });
        for (auto const &in : m_neurons | std::views::filter(Neuron::is_input))
        {
            for (auto const &out : m_neurons | std::views::filter(Neuron::is_output))
            {
                if (Random::canonical() < cfg.setup_inital_connection_rate)
                {
                    add_connection(in.number(), out.number());
                }
            }
        }
    }

    [[nodiscard]] real_t Brain::difference(const Brain &representative, const Config &cfg) const noexcept
    {
        std::vector<Synapse> tSynapses, rSynapses;
        std::ranges::copy_if(m_synapses, std::back_inserter(tSynapses), &Synapse::enabled);
        std::ranges::copy_if(representative.m_synapses, std::back_inserter(rSynapses), &Synapse::enabled);
        if (tSynapses.size() == 0 && rSynapses.size() == 0) return 0_r;
        std::ranges::sort(tSynapses, {}, &Synapse::innovation);
        std::ranges::sort(rSynapses, {}, &Synapse::innovation);

        const auto divisor = static_cast<real_t>(std::max(tSynapses.size(), rSynapses.size()));

        mj::InsertCounter disjointCounter;
        std::ranges::set_difference(tSynapses, rSynapses, disjointCounter, {}, &Synapse::innovation, &Synapse::innovation);
        const auto disjoint = static_cast<real_t>(disjointCounter.count);

        std::vector<Synapse> tMatching, rMatching;
        std::ranges::set_intersection(tSynapses, rSynapses, std::back_inserter(tMatching), {}, &Synapse::innovation, &Synapse::innovation);
        std::ranges::set_intersection(rSynapses, tSynapses, std::back_inserter(rMatching), {}, &Synapse::innovation, &Synapse::innovation);
        assert(tMatching.size() == rMatching.size());
        real_t weightDifference{};
        // TODO: Use zip?
        mj::loop(tMatching.size(), [&](const std::size_t i) { weightDifference += mj::difference(tMatching[i].weight(), rMatching[i].weight()); });
        const auto weightAverage = weightDifference / static_cast<real_t>(tMatching.size());

        return ((cfg.species_disjoint_coefficient * disjoint) / divisor) + weightAverage;
    }

    [[nodiscard]] Brain Brain::crossover(const Brain &best, const Brain &worst, const Config &cfg)
    {
        // Taking disjoint and matching synapses from best, so best has the nodes what we want.
        Brain brain;
        brain.m_neurons = best.m_neurons;
        brain.m_layer_count = best.m_layer_count;

        for (const auto &synapse : best.m_synapses)
        {
            // TODO: Don't copy invalid (ie bad layer order) synapses.

            bool enabled = true;

            const auto matchingSynapse = std::ranges::find_if(worst.m_synapses, mj::magic_callable(&Synapse::innovation, std::equal_to{}, synapse.innovation()));

            // Disjoint or excess gene, comes from best.
            if (matchingSynapse == worst.m_synapses.end())
            {
                assert(synapse.in() < mj::isize(brain.m_neurons));
                assert(synapse.out() < mj::isize(brain.m_neurons));
                brain.m_synapses.push_back(synapse);
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
            if (Random::canonical() < 0.5_r)
            {
                assert(synapse.in() < mj::isize(brain.m_neurons));
                assert(synapse.out() < mj::isize(brain.m_neurons));
                brain.m_synapses.push_back(synapse);
            }
            else
            {
                assert(matchingSynapse->in() < mj::isize(brain.m_neurons));
                assert(matchingSynapse->out() < mj::isize(brain.m_neurons));
                brain.m_synapses.push_back(*matchingSynapse);
            }
            brain.m_synapses.back().set_enabled(enabled);
        }
        return brain;
    }

    void Brain::mutate(const Config &cfg) noexcept
    {

        // In case we want to mutate from jack for minimal structure.
        if (m_synapses.size() == 0)
        {
            add_connection();
            return;
        }

        // Thanks to https://github.com/CodeReclaimers/neat-python/blob/37bc8bb73fd6153a115001c2646f9f02bac3ad81/neat/genome.py#L264
        const auto div = std::max(1.0_r, cfg.mutate_new_node_rate + cfg.mutate_new_connection_rate);
        const auto rand = Random::canonical();

        if (rand < cfg.mutate_new_node_rate / div)
        {
            add_node();
        }
        else if (rand < (cfg.mutate_new_node_rate + cfg.mutate_new_connection_rate) / div)
        {
            add_connection();
        }

        std::ranges::for_each(m_synapses, [&cfg](Synapse &s) { s.mutate_weight(cfg); });
    }

    [[nodiscard]] bool Brain::is_fully_connected() const noexcept
    {
        std::vector<std::size_t> neuronsInLayers(mj::sz_t(m_layer_count));
        std::ranges::for_each(m_neurons, [&neuronsInLayers](std::size_t n) { neuronsInLayers.at(n) += 1; }, &Neuron::layer);
        int maxConnections{};
        for (int i{}; i < m_layer_count - 1; ++i)
        {
            std::size_t inFront{};
            for (auto j = i + 1; j < m_layer_count; ++j)
            {
                inFront += neuronsInLayers.at(mj::sz_t(j));
            }
            maxConnections += gsl::narrow<int>(neuronsInLayers.at(mj::sz_t(i)) * inFront);
        }
        return maxConnections == mj::isize(m_synapses);
    }

    void Brain::add_connection(const innovation_t in, const innovation_t out)
    {
        if (in == out) return;
        if (m_neurons.at(mj::sz_t(in)).layer() >= m_neurons.at(mj::sz_t(out)).layer()) return;
        if (std::ranges::any_of(m_synapses, [in, out](const Synapse &s) { return s.in() == in && s.out() == out; })) return;
        const auto innovation = s_innovation_history.get_innovation_number(in, out);
        m_synapses.emplace_back(in, out, Random::weight(), innovation);
        // rebuild_layers();
    }

    void Brain::add_connection() noexcept
    {
        if (is_fully_connected()) return;

        std::vector<std::pair<innovation_t, innovation_t>> possibilities;
        for (int layer{}; layer < m_layer_count - 1; ++layer)
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
        assert(conn.first < mj::isize(m_neurons));
        assert(conn.second < mj::isize(m_neurons));
        add_connection(conn.first, conn.second);
    }

    // TODO: Recurrent connections, better layer checking?
    void Brain::add_node() noexcept
    {
        if (m_synapses.size() == 0)
        {
            add_connection();
            return;
        }

        Synapse &oldSynapse = Random::item(m_synapses);
        oldSynapse.set_enabled(false);
        const auto oldIn = oldSynapse.in();
        const auto oldOut = oldSynapse.out();
        const auto oldWeight = oldSynapse.weight();
        // Do *not* use oldSynapse below here!

        const auto newId = mj::isize(m_neurons);
        auto &neuron = m_neurons.emplace_back(newId, NeuronType::hidden);
        assert(oldIn < mj::isize(m_neurons));
        assert(oldOut < mj::isize(m_neurons));
        const auto innovation0 = s_innovation_history.get_innovation_number(oldIn, neuron.number());
        const auto innovation1 = s_innovation_history.get_innovation_number(neuron.number(), oldOut);
        m_synapses.emplace_back(oldIn, neuron.number(), 1.0_r, innovation0);
        m_synapses.emplace_back(neuron.number(), oldOut, oldWeight, innovation1);

        neuron.set_layer(m_neurons.at(mj::sz_t(oldIn)).layer() + 1);
        if (neuron.layer() != m_neurons.at(mj::sz_t(oldOut)).layer()) return;


        for (auto &adjustNeuron : m_neurons | mj::filter(&Neuron::layer, std::greater_equal{}, neuron.layer()))
        {
            if (&adjustNeuron == &neuron) continue;
            adjustNeuron.set_layer(adjustNeuron.layer() + 1);
        }
        ++m_layer_count;
        // rebuild_layers();

    }

    static void traceLongestPath(const Brain &brain, const Neuron &neuron, int &length)
    {
        for (const auto &synapse : brain.synapses() | mj::filter(&Synapse::out, std::equal_to{}, neuron.number()) | mj::filter(&Synapse::enabled, std::equal_to{}, true))
        {
            ++length;
            traceLongestPath(brain, brain.neurons().at(mj::sz_t(synapse.in())), length);
        }
    }

    void Brain::rebuild_layers() noexcept
    {
        for (auto &neuron : m_neurons)
        {
            int layer{};
            traceLongestPath(*this, neuron, layer);
            neuron.set_layer(layer);
        }
    }

    // TODO: Recurrent connections.
    [[nodiscard]] std::vector<real_t> Brain::run_network(const std::vector<real_t> &inputs, activator_f *activator) noexcept
    {
        const auto inputCount = std::ranges::count_if(m_neurons, Neuron::is_input);
        const auto outputCount = std::ranges::count_if(m_neurons, Neuron::is_output);
        assert(std::ssize(inputs) == inputCount);
        for (int i{}; i < inputCount; ++i)
        {
            m_neurons.at(mj::sz_t(i)).set_value(inputs.at(mj::sz_t(i)));
        }
        for (int i{ 1 }; i < m_layer_count; ++i)
        {
            for (auto &neuron : m_neurons | mj::filter(&Neuron::layer, std::equal_to{}, i))
            {
                real_t value = 0;
                for (const auto &synapse : m_synapses | mj::filter(&Synapse::out, std::equal_to{}, neuron.number()) | std::views::filter(&Synapse::enabled))
                {
                    value += m_neurons.at(mj::sz_t(synapse.in())).value() * synapse.weight();
                }
                neuron.set_value(activator(value));
            }
        }
        std::vector<real_t> outputs(mj::sz_t(outputCount));
        for (int i{}; i < outputCount; ++i)
        {
            outputs.at(mj::sz_t(i)) = m_neurons.at(mj::sz_t(inputCount + i)).value();
        }
        return outputs;
    }

    std::string Brain::chart() const noexcept
    {
        std::ostringstream stream;
        for (const auto &synapse : m_synapses | std::views::filter(&Synapse::enabled))
        {
            std::println(stream, "{}({}) -- {} --> {}({})", synapse.in(), m_neurons.at(mj::sz_t(synapse.in())).layer(), synapse.weight(), synapse.out(), m_neurons.at(mj::sz_t(synapse.out())).layer());
        }
        return stream.str();
    }

    std::string Brain::code_cpp() const noexcept
    {
        // TODO: code_cpp...
        return "";
    }

} // End namespace neat.