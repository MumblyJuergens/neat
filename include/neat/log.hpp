#pragma once

#include <print>
#include <source_location>

namespace neat
{
    template <typename... Ts>
    inline void debug(const std::source_location &loc, const std::format_string<Ts...> f, Ts &&...ts)
    {
        std::print("{}({}) : ", loc.file_name(), loc.line());
        std::println(f, std::forward<Ts>(ts)...);
    }

#ifdef NDEBUG
#define LOGGO(...)
#define DEBUGGO(...)
#else
#define LOGGO(f, ...) debug(std::source_location::current(), f, __VA_ARGS__)
#define DEBUGGO(...) std::println(__VA_ARGS__)
#endif

} // End namespace neat.