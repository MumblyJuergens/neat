#pragma once

#include <random>

namespace neat
{
    class [[nodiscard]] Random final
    {
        static inline std::default_random_engine engine{std::random_device{}()};

    public:
        static float weight() noexcept { return std::generate_canonical<float, 10>(engine) * 2.0f - 1.0f; }
    };
} // End namespace neat.