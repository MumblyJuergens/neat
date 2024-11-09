#include <ranges>
#include "neat/InnovationHistory.hpp"
#include "neat/Genome.hpp"
#include "neat/neat_export.h"

namespace neat
{
    innovation_t sg_global_innovation_number{};

    NEAT_EXPORT innovation_t next_global_innovation_number() noexcept
    {
        return sg_global_innovation_number++;
    }

    [[nodiscard]] innovation_t InnovationHistory::get_innovation_number(const Genome *genome, const innovation_t in, const innovation_t out) noexcept
    {
        for (const auto &entry : data)
        {
            if (entry.innovations.size() != genome->synapses().size()) continue;
            if (entry.in != in || entry.out != out) continue;
            if (!std::ranges::is_permutation(entry.innovations, genome->synapses(), {}, {}, &Synapse::innovation)) continue;
            return entry.innovation;
        }
        const auto num = next_global_innovation_number();
        std::vector<innovation_t> innovations(genome->synapses().size());
        std::ranges::transform(genome->synapses(), innovations.begin(), std::identity{}, &Synapse::innovation);
        data.emplace_back(std::move(innovations), in, out, num);
        return num;
    }

} // End namespace neat;