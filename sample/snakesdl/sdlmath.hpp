#pragma once
#include <SDL3/SDL_rect.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace mjsdl::math
{

inline auto operator+(SDL_FPoint lhs, SDL_FPoint rhs) noexcept -> SDL_FPoint { return {lhs.x + rhs.x, lhs.y + rhs.y}; }
inline auto operator+=(SDL_FPoint &lhs, SDL_FPoint rhs) noexcept -> SDL_FPoint
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}
inline auto operator<(SDL_FPoint lhs, float rhs) noexcept -> bool { return lhs.x < rhs || lhs.y < rhs; }
inline auto operator>(SDL_FPoint lhs, float rhs) noexcept -> bool { return lhs.x > rhs || lhs.y > rhs; }
inline auto operator<=(SDL_FPoint lhs, float rhs) noexcept -> bool { return lhs.x <= rhs || lhs.y <= rhs; }
inline auto operator>=(SDL_FPoint lhs, float rhs) noexcept -> bool { return lhs.x >= rhs || lhs.y >= rhs; }

template <class T>
std::enable_if_t<not std::numeric_limits<T>::is_integer, bool> equal_within_ulps(T x, T y, std::size_t n)
{
    // Since `epsilon()` is the gap size (ULP, unit in the last place)
    // of floating-point numbers in interval [1, 2), we can scale it to
    // the gap size in interval [2^e, 2^{e+1}), where `e` is the exponent
    // of `x` and `y`.

    // If `x` and `y` have different gap sizes (which means they have
    // different exponents), we take the smaller one. Taking the bigger
    // one is also reasonable, I guess.
    const T m = std::min(std::fabs(x), std::fabs(y));

    // Subnormal numbers have fixed exponent, which is `min_exponent - 1`.
    const int exp = m < std::numeric_limits<T>::min() ? std::numeric_limits<T>::min_exponent - 1 : std::ilogb(m);

    // We consider `x` and `y` equal if the difference between them is
    // within `n` ULPs.
    return std::fabs(x - y) <= static_cast<T>(n) * std::ldexp(std::numeric_limits<T>::epsilon(), exp);
}

inline bool equal_within_ulps(SDL_FPoint a, SDL_FPoint b, std::size_t n)
{
    return equal_within_ulps(a.x, b.x, n) && equal_within_ulps(a.y, b.y, n);
}

} // namespace mjsdl::math

// These need to specifically be exposed outside the namespace to make some template ranges work.
inline auto operator==(SDL_FPoint lhs, SDL_FPoint rhs) noexcept -> bool { return lhs.x == rhs.x && lhs.y == rhs.y; }
inline auto operator!=(SDL_FPoint lhs, SDL_FPoint rhs) noexcept -> bool { return !(lhs == rhs); }