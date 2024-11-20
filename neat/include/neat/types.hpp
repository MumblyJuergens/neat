#pragma once

#include "neat/configure.hpp"

namespace neat
{

    using innovation_t = int;
    using index_t = int;
#ifdef NEAT_DOUBLE_PRECISION
    using real_t = double;
#else 
    using real_t = float;
#endif
    using activator_f = real_t(real_t);
    enum class Init { no, yes };

    namespace literals
    {
        [[nodiscard]] constexpr real_t operator""_r(const long double n) noexcept { return static_cast<real_t>(n); }
        [[nodiscard]] constexpr real_t operator""_r(const unsigned long long n) noexcept { return static_cast<real_t>(n); }
    } // End namespace literals.

    [[nodiscard]] constexpr real_t operator""_r(const long double n) noexcept { return static_cast<real_t>(n); }
    [[nodiscard]] constexpr real_t operator""_r(const unsigned long long n) noexcept { return static_cast<real_t>(n); }

} // End namespace neat;