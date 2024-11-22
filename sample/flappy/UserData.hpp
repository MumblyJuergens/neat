#pragma once

#include <array>
#include <vector>
#include <glm/vec2.hpp>
#include <glube/BuildingBuffer.hpp>
#include "Pipe.hpp"

namespace flappy
{

    struct UserData
    {
        std::array<Pipe, Pipe::count> *const pipes;
        glube::BuildingBuffer<glm::vec2> *const birds_mbo;
        float delta;
    };

} // End namesapce flappy.