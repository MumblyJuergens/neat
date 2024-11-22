#pragma once

#include <array>
#include <vector>
#include <glm/vec2.hpp>
#include "Pipe.hpp"

struct UserData
{
    std::array<Pipe, Pipe::count> *const pipes;
    std::vector<glm::vec2> *const birds_mbo_data;
    float delta;
};