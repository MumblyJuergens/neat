#include <ranges>
#include "neat/InnovationHistory.hpp"
#include "neat/Genome.hpp"
#include "neat/neat_export.h"
#include "neat/Brain.hpp"

namespace neat
{
    innovation_t InnovationHistory::s_global_innovation_number = 0;
    std::unordered_map<iipair, innovation_t, InnovationHistory::iipair_hash> InnovationHistory::data;

    innovation_t InnovationHistory::next_global_innovation_number() noexcept
    {
        return s_global_innovation_number++;
    }

    [[nodiscard]] innovation_t InnovationHistory::get_innovation_number(const innovation_t in, const innovation_t out) noexcept
    {
        const iipair p{ in, out };
        if (data.contains(p))
        {
            return data.at(p);
        }
        else
        {
            const auto num = next_global_innovation_number();
            data.emplace(p, num);
            return num;
        }
    }

} // End namespace neat;