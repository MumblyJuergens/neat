#pragma once

#include <cstddef>
#include "neat/neat_export.h"

namespace neat
{
    using innovation_t = std::size_t;
    NEAT_EXPORT innovation_t next_global_innovation_number() noexcept;
} // End namespace neat.