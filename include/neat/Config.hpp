#pragma once

#include "neat/types.hpp"

namespace neat
{
    class Config
    {
        public:

        // Setup
        const int setup_population_size = 100;
        const int setup_input_nodes = 3;
        const int setup_output_nodes = 1;
        const bool setup_connect_bias = false;
        const index_t setup_bias_input = 0;
        const real_t setup_inital_connection_rate = 0.8_r;

        // Speciation.
        const int species_maximum_staleness = 15;
        real_t species_compatability_threshold = 0.5_r;
        const real_t species_disjoint_coefficient = 1.0_r;
        const real_t species_weight_coefficent = 0.5_r;
        const real_t species_compatability_modifier = 0.3_r;
        const int species_count_target = 5;

        // Crossover.
        const int crossover_elite_size = 5;

        // Mutation.
        const real_t mutate_weight_rate = 0.8_r;
        const real_t mutate_weight_amount = 0.5_r;
        const real_t mutate_weight_min = -20.0_r;
        const real_t mutate_weight_max = 20.0_r;
        const real_t mutate_redraw_weight = 0.1_r;
        const real_t mutate_new_connection_rate = 0.5_r;
        const real_t mutate_new_node_rate = 0.1_r;
        const real_t mutate_disable_node_rate = 0.75_r;

    };
} // End namespace neat.