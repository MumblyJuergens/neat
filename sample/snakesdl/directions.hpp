#pragma once

#include <SDL3/SDL_rect.h>
#include <array>
#include <utility>

namespace snakesdl
{

struct Directions
{
    // clang-format off
    static constexpr SDL_FPoint
        up{ 0, -1 },
        right{ 1, 0 },
        down{ 0, 1 },
        left{ -1, 0 },
        ur{ 1, -1 },
        dr{ 1, 1 },
        dl{ -1, 1 },
        ul{ -1, -1 };
        
    static constexpr std::array<SDL_FPoint, 8> ORDERED{
        up,
        ur,
        right,
        dr,
        down,
        dl,
        left,
        ul,
    };

    enum class Names : size_t {
        UP, UR, RIGHT, DR, DOWN, DL, LEFT, UL,
    };

    enum class Turn
    {
        LEFT_BACK = -3,
        LEFT = -2,
        LEFT_FRONT = -1,
        NOPE = 0,
        RIGHT_FRONT = 1,
        RIGHT = 2,
        RIGHT_BACK = 3, 
    };

    [[nodiscard]] static inline constexpr SDL_FPoint for_name(Names name) noexcept
    {
        return ORDERED[std::to_underlying(name)];
    }

    [[nodiscard]] static inline constexpr auto after_turn(Names direction, Turn turn) noexcept
    {
        return Directions::Names{
            static_cast<size_t>(static_cast<int>(std::to_underlying(direction) + Directions::ORDERED.size()) +
                                std::to_underlying(turn)) %
            Directions::ORDERED.size()};
    }

    // clang-format on
};

static_assert(Directions::after_turn(Directions::Names::RIGHT, Directions::Turn::LEFT) == Directions::Names::UP);
static_assert(Directions::after_turn(Directions::Names::RIGHT, Directions::Turn::RIGHT) == Directions::Names::DOWN);
static_assert(Directions::after_turn(Directions::Names::RIGHT, Directions::Turn::RIGHT_FRONT) == Directions::Names::DR);

} // namespace snakesdl