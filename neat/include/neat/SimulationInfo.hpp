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

    [[nodiscard]] SimulationInfo(Genome &p_genome, real_t p_fitness, UserData *const p_user_data)
        : genome{p_genome}, fitness{p_fitness}, user_data{p_user_data}
    {
    }
};

} // namespace neat