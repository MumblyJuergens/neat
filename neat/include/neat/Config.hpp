#pragma once

#include "neat/types.hpp"

namespace neat
{
    class Config
    {
        public:

        // Setup
        int setup_population_size = 100;
        int setup_input_nodes = 3;
        int setup_output_nodes = 1;
        bool setup_connect_bias = false;
        index_t setup_bias_input = 0;
        real_t setup_inital_connection_rate = 0.8_r;

        // Speciation.
        int species_maximum_staleness = 15;
        real_t species_compatability_threshold = 0.5_r;
        real_t species_disjoint_coefficient = 1.0_r;
        real_t species_weight_coefficent = 0.5_r;
        real_t species_compatability_modifier = 0.3_r;
        int species_count_target = 5;

        // Crossover.
        int crossover_elite_size = 5;
        bool crossover_use_adjusted_fitness = true;

        // Mutation.
        real_t mutate_weight_rate = 0.8_r;
        real_t mutate_weight_amount = 0.5_r;
        real_t mutate_weight_min = -20.0_r;
        real_t mutate_weight_max = 20.0_r;
        real_t mutate_redraw_weight = 0.1_r;
        real_t mutate_new_connection_rate = 0.5_r;
        real_t mutate_new_node_rate = 0.1_r;
        real_t mutate_disable_node_rate = 0.75_r;

        // Tweaks.
        bool tweak_serialize_simulation_factory = false;

        template<typename Archive>
        void serialize(Archive &ar)
        {
            ar(
                setup_population_size,
                setup_input_nodes,
                setup_output_nodes,
                setup_connect_bias,
                setup_bias_input,
                setup_inital_connection_rate,

                species_maximum_staleness,
                species_compatability_threshold,
                species_disjoint_coefficient,
                species_weight_coefficent,
                species_compatability_modifier,
                species_count_target,

                crossover_elite_size,
                crossover_use_adjusted_fitness,

                mutate_weight_rate,
                mutate_weight_amount,
                mutate_weight_min,
                mutate_weight_max,
                mutate_redraw_weight,
                mutate_new_connection_rate,
                mutate_new_node_rate,
                mutate_disable_node_rate,

                tweak_serialize_simulation_factory
            );
        }

    };
} // End namespace neat.