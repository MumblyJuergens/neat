
#include "neat/Genome.hpp"
#include "neat/types.hpp"

namespace neat
{

void Genome::simple_step(const std::vector<real_t> &inputs, std::vector<real_t> &outputs, activator_f *activator)
{
    outputs = m_brain.run_network(inputs, *activator);
}

} // namespace neat