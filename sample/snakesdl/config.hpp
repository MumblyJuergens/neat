#pragma once

namespace snakesdl::config
{
static constexpr float POINT_SIZE = 6;
static constexpr float BORDER_POINTS = 2;
static constexpr float GAME_SIZE = 42;
static constexpr float COLLIDE_TL = BORDER_POINTS;
static constexpr float COLLIDE_BR = BORDER_POINTS + GAME_SIZE - BORDER_POINTS + 1; // I know how that math works.
static constexpr int WINDOW_SIZE = static_cast<int>((GAME_SIZE + BORDER_POINTS * 2) * POINT_SIZE);

}; // namespace snakesdl::config