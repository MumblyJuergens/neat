#pragma once

#include <vector>
#include "neat/innovation.hpp"

namespace neat
{

    class Genome;

    class [[nodiscard]] InnovationHistory final
    {
        struct inner_t {
            const std::vector<innovation_t> innovations;
            const innovation_t in;
            const innovation_t out;
            innovation_t innovation{};
        };
        std::vector<inner_t> data;

        public:
        [[nodiscard]] innovation_t get_innovation_number(const Genome *genome, const innovation_t in, const innovation_t out) noexcept;
    };

} // End namespace neat.