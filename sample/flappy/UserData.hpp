#pragma once

#include <array>
#include <vector>
#include <glm/vec2.hpp>
#include <glube/BuildingBuffer.hpp>
#include "Pipe.hpp"
#include "neat/UserData.hpp"

namespace flappy
{
    struct UserData : public neat::UserData
    {
        std::array<Pipe, Pipe::count> *const pipes;
        glube::BuildingBuffer<glm::vec2> *const birds_mbo;
        float delta{};

        UserData(std::array<Pipe, Pipe::count> *const pipes, glube::BuildingBuffer<glm::vec2> *const birds_mbo)
            : pipes{ pipes }, birds_mbo{ birds_mbo } {
        }
    };

} // End namesapce flappy.