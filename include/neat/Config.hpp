#pragma once

namespace neat
{
    struct Config
    {
        // For calculating the difference between genomes.
        float disjoint_coefficient = 1.0f;
        float excess_coefficient = 1.0f;
        float weight_difference_coefficent = 1.0f;
        bool absolute_difference = true;
        float compatability_threshold = 3.0f;
    };
} // End namespace neat.