#pragma once

#include "neat/Genome.hpp"
#include "neat/types.hpp"
#include <vector>

namespace neat
{
class [[nodiscard]] SimulationInfo
{
  public:
    Genome &genome;
    std::vector<real_t> inputs;
    std::vector<real_t> outputs;
    real_t fitness{};
    bool is_done{};
    bool is_perfect{};
    UserData *const user_data;

    template <std::floating_point... Args>
    void assign_inputs(Args... args)
    {
        inputs.assign({args...});
    }

    void run(activator_f *activator) { genome.step(*this, activator); }

    [[nodiscard]] SimulationInfo(Genome &genome, real_t fitness, UserData *const user_data)
        : genome{genome}, fitness{fitness}, user_data{user_data}
    {
    }
};

} // namespace neat