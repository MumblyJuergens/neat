#pragma once

#include "neat/innovation.hpp"

namespace neat
{
    struct Config
    {
        // For calculating the difference between genomes.
        float disjoint_coefficient = 1.0f;
        float excess_coefficient = 1.0f;
        float weight_difference_coefficent = 0.5f;
        bool absolute_difference = true;
        float compatability_threshold = 3.0f;

        // For evolution rates.
        float mutate_weight_rate = 0.8f;
        float mutate_weight_divisor = 10.0f;
        float mutate_redraw_weight = 0.1f;
        float mutate_just_clone_rate = 0.25f;
        float mutate_new_connection_rate = 0.5f;
        float mutate_new_node_rate = 0.2f;
        float mutate_disable_node_rate = 0.75f; // Seems high but it's if either parent is disabled.

        // For speciating.
        const int maximum_species_staleness = 15;
        const std::size_t minimum_species_size = 10;

        // Other.
        const bool fully_connect = false;
        const bool connect_bias = false;
        const innovation_t bias_input = 0;
    };
} // End namespace neat.