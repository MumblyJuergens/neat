#pragma once

#include "config.hpp"
#include "directions.hpp"
#include "neat/Random.hpp"
#include "random.hpp"
#include "sdlmath.hpp"
#include <SDL3/SDL_rect.h>
#include <algorithm>
#include <cstdint>
#include <mjsdl/Renderer.hpp>
#include <random>
#include <vector>

namespace snakesdl
{

struct Snake
{

    // Enough energy to do a lap and a half.
    static constexpr int MAX_ENERGY = static_cast<int>(config::GAME_SIZE * 6);

    SDL_Color color;
    std::vector<SDL_FPoint> points;
    Directions::Names direction{Directions::Names::UP};
    bool dead{};
    int energy = MAX_ENERGY;
    float fitness{};
    bool fed{};
    bool is_champ{};

    Snake()
    {
        color = SDL_Color{
            static_cast<uint8_t>(snake_random.range(255)),
            static_cast<uint8_t>(snake_random.range(255)),
            static_cast<uint8_t>(snake_random.range(255)),
            SDL_ALPHA_OPAQUE,
        };
        reset();
    }

    void reset()
    {
        direction = Directions::Names::UP;
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
            // {begin.x, begin.y - 8},
            // {begin.x, begin.y - 9},
            // {begin.x, begin.y - 10},
            // {begin.x, begin.y - 11},
            // {begin.x, begin.y - 12},
            // {begin.x, begin.y - 13},
            // {begin.x, begin.y - 14},
            // {begin.x, begin.y - 15},
            // {begin.x, begin.y - 16},
        });
    }

    [[nodiscard]] constexpr auto head() const noexcept -> SDL_FPoint { return points.front(); }

    void move()
    {
        using namespace mjsdl::math;

        const auto head = this->head();
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
        points.insert(points.begin(), points.front() + Directions::for_name(direction));
    }

    void draw(const mjsdl::Renderer &renderer, uint8_t alpha = 255) const
    {
        SDL_SetRenderDrawColor(renderer.get(), color.r, color.g, color.b, alpha);
        SDL_RenderPoints(renderer.get(), points.data(), mj::isize(points));
    }

    void turn_left() { direction = Directions::after_turn(direction, Directions::Turn::LEFT); }
    void turn_right() { direction = Directions::after_turn(direction, Directions::Turn::RIGHT); }
};

} // namespace snakesdl