#pragma once

#include <algorithm>
#include <vector>
#include <cstddef>
#include <glube/utilities.hpp>
#include "Vertex.hpp"
#include "Config.hpp"

namespace flappy
{

    struct Bird
    {
        static constexpr std::size_t vertex_count = 20;
        static constexpr float radius = 25.0f;
        static constexpr float gravity = 700.0f;
        static constexpr float fly_speed = 200.0f;
        static constexpr float starting_x = 200.0f;

        [[nodiscard]] static std::vector<Vertex> vertices() noexcept
        {
            auto positions = glube::triangle_fan_circle(radius, vertex_count - 2);
            std::vector<Vertex> output;
            output.reserve(positions.size());
            std::ranges::transform(positions, std::back_inserter(output), [](const glm::vec2 &v) { return Vertex{ .position = v, .color = { 0.8f, 0.8f, 0.1f } }; });
            return output;
        }

        void reset() noexcept { translation = { starting_x, Config::window_heightf / 2.0f }; vert_speed = 0.0f; }

        glm::vec2 translation = { starting_x, Config::window_heightf / 2.0f };
        float vert_speed{};
    };

} // ENd namespace flappy.