#pragma once

#include "config.hpp"
#include "random.hpp"
#include <SDL3/SDL_rect.h>
#include <cmath>
#include <cstddef>
#include <vector>

namespace snakesdl
{

class Food
{

    static constexpr size_t GROW_SIZE = 50;
    std::vector<SDL_FPoint> items;

  public:
    auto operator[](size_t index) -> SDL_FPoint
    {
        if (index >= items.size()) {
            const auto old_size = items.size();
            items.resize(index + GROW_SIZE);
            for (size_t i = old_size; i < items.size(); ++i) {
                items[i] = SDL_FPoint{
                    std::floor(snake_random.range(config::COLLIDE_TL + 1, config::COLLIDE_BR)),
                    std::floor(snake_random.range(config::COLLIDE_TL + 1, config::COLLIDE_BR)),
                };
            }
        }
        return items[index];
    }

    void reset() { items.clear(); }
};

} // namespace snakesdl