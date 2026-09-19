#pragma once

#include "directions.hpp"
#include <SDL3/SDL_rect.h>

namespace snakesdl
{

struct RayResult
{
    static constexpr float WALL = 0.0f;
    static constexpr float TAIL = 0.5f;
    static constexpr float FOOD = 1.0f;

    float distance;
    float type;
};

RayResult raycast(const struct Snake &snake, SDL_FPoint food, Directions::Turn turn);

} // namespace snakesdl