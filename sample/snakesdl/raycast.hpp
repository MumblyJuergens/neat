#pragma once

namespace snakesdl
{

struct RayResult
{
    float distance;
    float type;
};

RayResult raycast(const struct Snake &snake, const struct Game &game);

} // namespace snakesdl