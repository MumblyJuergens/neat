#include "neat/Genome.hpp"
#include "neat/Simulation.hpp"
#include "neat/SimulationInfo.hpp"

namespace neat
{
    void Genome::step()
    {
        SimulationInfo info{ *this, m_fitness };
        m_simulation->step(info);
        m_fitness = info.fitness;
        m_sim_is_done = info.is_done;
        m_sim_is_perfect = info.is_perfect;
    }

    void Genome::step(SimulationInfo &info, activator_f *activator)
    {
        info.outputs = m_brain.run_network(info.inputs, activator);
    }

} // End namespace neat.