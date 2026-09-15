#pragma once

#include "config.hpp"
#include "neat/Random.hpp"
#include "sdlmath.hpp"
#include <SDL3/SDL_rect.h>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <mjsdl/Renderer.hpp>
#include <utility>
#include <vector>

namespace snakesdl
{

enum class SnakeTurn
{
    TURN_LEFT = -2,
    TURN_NOPE = 0,
    TURN_RIGHT = 2,
};

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

    // clang-format on
};

struct Snake
{

    // Enough energy to do a lap and a half.
    static constexpr int MAX_ENERGY = static_cast<int>(config::GAME_SIZE * 6);

    SDL_Color color;
    std::vector<SDL_FPoint> points;
    size_t direction{};
    bool dead{};
    int energy = MAX_ENERGY;
    float fitness{};
    bool fed{};
    bool is_champ{};

    Snake()
    {
        color = SDL_Color{
            static_cast<uint8_t>(neat::Random::range(255)),
            static_cast<uint8_t>(neat::Random::range(255)),
            static_cast<uint8_t>(neat::Random::range(255)),
            SDL_ALPHA_OPAQUE,
        };
        reset();
    }

    void reset()
    {
        direction = 0;
        dead = false;
        energy = MAX_ENERGY;
        fitness = 0;
        fed = false;
        is_champ = false;

        constexpr auto begin = SDL_FPoint{config::GAME_SIZE / 2, config::GAME_SIZE / 2};
        points.clear();
        points.assign({
            begin,
            {begin.x, begin.y - 1},
            {begin.x, begin.y - 2},
            {begin.x, begin.y - 3},
            {begin.x, begin.y - 4},
            {begin.x, begin.y - 5},
            {begin.x, begin.y - 6},
            {begin.x, begin.y - 7},
            {begin.x, begin.y - 8},
            {begin.x, begin.y - 9},
            {begin.x, begin.y - 10},
            {begin.x, begin.y - 11},
            {begin.x, begin.y - 12},
            {begin.x, begin.y - 13},
            {begin.x, begin.y - 14},
            {begin.x, begin.y - 15},
            {begin.x, begin.y - 16},
        });
    }

    SDL_FPoint head() { return points.front(); }

    void move()
    {
        using namespace mjsdl::math;

        const auto head = points.front();
        if (head <= config::COLLIDE_TL || head >= config::COLLIDE_BR) {
            dead = true;
        }
        if (std::ranges::count_if(points,
                                  [head](SDL_FPoint p) { return mjsdl::math::equal_within_ulps(p, head, 1); }) != 1)
        {
            dead = true;
        }

        if (energy-- <= 0) {
            dead = true;
        }

        if (!fed) {
            points.pop_back();
        } else {
            fed = false;
            energy = MAX_ENERGY;
        }
        points.insert(points.begin(), points.front() + Directions::ORDERED[direction]);
    }

    void draw(const mjsdl::Renderer &renderer) const
    {
        SDL_SetRenderDrawColor(renderer.get(), color.r, color.g, color.b, color.a);
        SDL_RenderPoints(renderer.get(), points.data(), mj::isize(points));
    }

    [[nodiscard]] constexpr auto direction_after_turn(SnakeTurn turn) const noexcept -> size_t
    {
        return static_cast<size_t>(static_cast<int>(direction + Directions::ORDERED.size()) +
                                   std::to_underlying(turn)) %
               Directions::ORDERED.size();
    }
    void turn_left() { direction = direction_after_turn(SnakeTurn::TURN_LEFT); }
    void turn_right() { direction = direction_after_turn(SnakeTurn::TURN_RIGHT); }

    template <typename Pred, typename... Args>
        requires std::predicate<Pred, SDL_FPoint, Args...>
    float raycast(SnakeTurn dir, Pred predicate, Args &&...args)
    {
        using namespace mjsdl::math;
        float distance{};
        const auto adder = Directions::ORDERED[direction_after_turn(dir)];
        const auto head = points.front() + adder;

        for (SDL_FPoint next_head{head};
             !(predicate(next_head, std::forward<Args>(args)...) || wall_collide(next_head)); next_head += adder)
        {
            distance += 1.0f;
        }

        return distance / config::GAME_SIZE;
    }

    static bool just_wall(SDL_FPoint) { return false; }

    static bool wall_collide(SDL_FPoint point)
    {
        using namespace mjsdl::math;
        return point <= config::COLLIDE_TL || point >= config::COLLIDE_BR;
    }

    static bool tail_collide(SDL_FPoint point, const Snake &snake)
    {
        return std::ranges::contains(snake.points, point);
    }
};

} // namespace snakesdl