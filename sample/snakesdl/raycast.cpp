#include "raycast.hpp"
#include "config.hpp"
#include "sdlmath.hpp"
#include "snake.hpp"
#include <SDL3/SDL_rect.h>
#include <algorithm>

namespace snakesdl
{

RayResult raycast(const Snake &snake, SDL_FPoint food, Directions::Turn turn)
{
    using namespace mjsdl::math;
    float distance{1.0f};
    float type{RayResult::WALL};
    const auto adder = Directions::for_name(Directions::after_turn(snake.direction, turn));
    const auto head = snake.head() + adder;

    for (SDL_FPoint next_head{head};; next_head += adder) {
        if (next_head <= config::COLLIDE_TL || next_head >= config::COLLIDE_BR) {
            type = RayResult::WALL;
            break;
        }
        if (std::ranges::contains(snake.points, next_head)) {
            type = RayResult::TAIL;
            break;
        }
        if (mjsdl::math::equal_within_ulps(next_head, food, 1)) {
            type = RayResult::FOOD;
            break;
        }
        distance += 1.0f;
    }

    return RayResult{.distance = distance / config::GAME_SIZE, .type = type};
}

} // namespace snakesdl