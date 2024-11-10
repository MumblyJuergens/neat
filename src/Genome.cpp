#include "neat/Genome.hpp"
#include "neat/Simulation.hpp"

namespace neat
{
    NEAT_EXPORT void Genome::step()
    {
        SimulationInfo info{ *this, m_fitness };
        m_simulation->step(info);
        m_fitness = info.fitness;
        m_sim_is_done = info.is_done;
    }

    NEAT_EXPORT void Genome::step(SimulationInfo &info)
    {
        const auto inputCount = static_cast<std::size_t>(std::ranges::count_if(m_neurons, Neuron::is_input));
        const auto outputCount = static_cast<std::size_t>(std::ranges::count_if(m_neurons, Neuron::is_output));
        assert(info.inputs.size() == inputCount);
        for (std::size_t i{}; i < inputCount; ++i)
        {
            m_neurons.at(i).set_value(info.inputs.at(i));
        }
        run_network();
        info.outputs.resize(outputCount);
        for (std::size_t i{}; i < outputCount; ++i)
        {
            info.outputs.at(i) = m_neurons.at(inputCount + i).value();
        }
    }
} // End namespace neat.