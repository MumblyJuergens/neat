// Tests finding a network for solving xor, also used for benchmarks.

#include <neat/Genome.hpp>
#include <neat/SimplePopulation.hpp>
#include <neat/neat.hpp>
#include <neat/types.hpp>
#include <print>
#include <ranges>
#include <vector>

struct Entry
{
    neat::real_t in0;
    neat::real_t in1;
    bool out;
};

int main()
{
    using namespace neat::literals;

    static constexpr std::array<Entry, 4> data{
        Entry{0.0_r, 0.0_r, false},
        Entry{0.0_r, 1.0_r, true},
        Entry{1.0_r, 0.0_r, true},
        Entry{1.0_r, 1.0_r, false},
    };

    neat::Config cfg{};
    cfg.setup_inital_connection_rate = 0.0_r;
    cfg.species_compatibility_threshold = 5.0_r;

    neat::SimplePopulation population{cfg};
    std::vector<neat::real_t> fitnesses;

    std::vector<neat::real_t> in(3uz, 0.0_r), out;
    bool solved{};
    while (!solved) {
        population.step([&](std::vector<neat::Genome> &genomes) {
            if (genomes.size() > fitnesses.size()) {
                fitnesses.resize(genomes.size());
            }
            for (auto [fitness, genome] : std::views::zip(fitnesses, genomes)) {
                fitness = 0;
                for (auto d : data) {
                    in[0] = 1.0_r;
                    in[1] = d.in0;
                    in[2] = d.in1;

                    genome.simple_step(in, out, std::tanh);
                    const auto correct = (out.front() > 0.5_r) == d.out;

                    if (correct) {
                        fitness += 1.0_r;
                        genome.set_fitness(fitness);
                    }
                }
            }
        });
        // std::println("Max fitness: {}", population.max_fitness());
        if (population.max_fitness() >= 3.9_r) {
            std::println("Completed in {} generations", population.generation());
            solved = true;
        } else {
            population.new_generation();
        }
    }
}