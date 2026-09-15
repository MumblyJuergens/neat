#include "raycast.hpp"
#include "snake.hpp"
#include <SDL3/SDL_rect.h>

namespace snakesdl
{

RayResult raycast(const Snake &snake, const Game &game, Directions::Names direction)
{

    using namespace mjsdl::math;
    float distance{};
    const auto adder = Directions::ORDERED[snake.direction_after_turn(direction)];
    const auto head = snake.points.front() + adder;

    for (SDL_FPoint next_head{head}; !(predicate(next_head, std::forward<Args>(args)...) || wall_collide(next_head));
         next_head += adder)
    {
        distance += 1.0f;
    }

    return distance / config::GAME_SIZE;
}

} // namespace snakesdl