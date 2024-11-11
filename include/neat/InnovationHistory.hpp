#pragma once

#include <unordered_map>
#include "neat/innovation.hpp"

namespace neat
{

    class Genome;
    using iipair = std::pair<innovation_t, innovation_t>;

    class [[nodiscard]] InnovationHistory final
    {
        struct [[nodiscard]] iipair_hash final
        {
            using argument_type = iipair;
            using result_type = std::size_t;

            [[nodiscard]] static std::uint64_t mixup(std::uint64_t x) noexcept
            {
                static constexpr std::uint64_t m = 0xe9846af9b1a615d;
                x ^= x >> 32;
                x *= m;
                x ^= x >> 32;
                x *= m;
                x ^= x >> 28;
                return x;
            }

            [[nodiscard]] static constexpr std::uint32_t mixup(std::uint32_t x) noexcept
            {
                static constexpr std::uint32_t m1 = 0x21f0aaad;
                static constexpr std::uint32_t m2 = 0x735a2d97;
                x ^= x >> 16;
                x *= m1;
                x ^= x >> 15;
                x *= m2;
                x ^= x >> 15;
                return x;
            }

            [[nodiscard]] constexpr std::size_t operator()(const iipair &v) const noexcept
            {
                std::size_t seed = 0;
                seed = mixup(seed + 0x9e3779b9 + static_cast<std::size_t>(v.first));
                seed = mixup(seed + 0x9e3779b9 + static_cast<std::size_t>(v.second));
                return seed;
            }
        };

        std::unordered_map<iipair, innovation_t, iipair_hash> data;

        public:

        [[nodiscard]] innovation_t get_innovation_number(const innovation_t in, const innovation_t out) noexcept;
    };

} // End namespace neat.