#pragma once

#include "Bird.hpp"
#include "Pipe.hpp"

namespace flappy
{

    static bool game_logic(Bird &bird, const bool up, const float delta, const Pipe &pipe0)
    {
        if (up)
        {
            bird.vert_speed = Bird::fly_speed;
        }
        bird.translation.y += bird.vert_speed * delta;
        bird.vert_speed -= Bird::gravity * delta;

        return pipe0.collide(bird, Config::window_heightf) || bird.translation.y < -Bird::radius || bird.translation.y > Config::window_heightf + Bird::radius;
    }

} // End namespace flappy.