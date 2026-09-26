#include "neat/InnovationHistory.hpp"

namespace neat
{

[[nodiscard]] innovation_t InnovationHistory::get_innovation_number(const innovation_t in,
                                                                    const innovation_t out) noexcept
{
    const iipair p{in, out};
    if (data.contains(p)) {
        return data.at(p);
    } else {
        const auto num = next_global_innovation_number();
        data.emplace(p, num);
        return num;
    }
}

} // namespace neat