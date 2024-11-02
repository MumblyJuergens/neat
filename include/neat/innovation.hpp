#pragma once

#include <cstddef>

namespace neat
{
    using innovation_t = std::ptrdiff_t;
    inline constexpr innovation_t next_global_innovation_number() noexcept
    {
        static innovation_t number{};
        return number++;
    }
} // End namespace neat.