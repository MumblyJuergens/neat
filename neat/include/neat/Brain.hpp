#pragma once

#include <string>
#include <vector>
#include <mj/size.hpp>
#include "neat/neat_export.h"
#include "neat/types.hpp"
#include "neat/Config.hpp"
#include "neat/InnovationHistory.hpp"
#include "neat/Neuron.hpp"
#include "neat/Synapse.hpp"

namespace neat
{
    class [[nodiscard]] NEAT_EXPORT Brain final
    {
        std::vector<Synapse> m_synapses;
        std::vector<Neuron> m_neurons;
        int m_layer_count{ 2 };
        static InnovationHistory s_innovation_history;

        public:

        [[nodiscard]] constexpr int synapse_count() const noexcept { return mj::isize(m_synapses); }
        [[nodiscard]] constexpr int neuron_count() const noexcept { return mj::isize(m_neurons); }
        [[nodiscard]] constexpr int layer_count() const noexcept { return m_layer_count; }

        [[nodiscard]] constexpr Brain() noexcept = default;
        [[nodiscard]] constexpr Brain(const Brain &) noexcept = default;
        [[nodiscard]] constexpr Brain(Brain &&) noexcept = default;
        constexpr Brain &operator=(const Brain &) noexcept = default;
        constexpr Brain &operator=(Brain &&) noexcept = default;

        void init(const Config &cfg, const Init init) noexcept;
        [[nodiscard]] real_t difference(const Brain &representative, const Config &cfg) const noexcept;
        [[nodiscard]] static Brain crossover(const Brain &best, const Brain &worst, const Config &cfg);
        void mutate(const Config &cfg) noexcept;
        [[nodiscard]] bool is_fully_connected() const noexcept;
        void add_connection(const innovation_t in, const innovation_t out);
        void add_connection() noexcept;
        void add_node() noexcept;
        [[nodiscard]] std::vector<real_t> run_network(const std::vector<real_t> &inputs, activator_f *activator) noexcept;
        std::string chart() const noexcept;
        std::string code_cpp() const noexcept;
        void diagram() const noexcept {} // TODO: ...
    };

} // End namespace neat.