
#include "neat/Genome.hpp"
#include "neat/Simulation.hpp"
#include "neat/SimulationInfo.hpp"

namespace neat
{
    void Genome::step(UserData *const userData)
    {
        SimulationInfo info{ *this, m_fitness, userData };
        if (!m_simulation->is_inited())
        {
            m_simulation->init(info);
            m_simulation->set_is_inited(true);
        }
        m_simulation->step(info);
        m_fitness = info.fitness;
        m_sim_is_done = info.is_done;
        m_sim_is_perfect = info.is_perfect;
    }

    void Genome::skip(UserData *const userData)
    {
        SimulationInfo info{ *this, m_fitness, userData };
        m_simulation->skip(info);
        // m_fitness = info.fitness;
        // m_sim_is_done = info.is_done;
        // m_sim_is_perfect = info.is_perfect;
    }

    void Genome::step(SimulationInfo &info, activator_f *activator)
    {
        info.outputs = m_brain.run_network(info.inputs, activator);
    }

} // End namespace neat.