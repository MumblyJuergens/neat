#pragma once

#include <array>
#include <cstddef>
#include <glm/vec2.hpp>
#include <glm/geometric.hpp>
#include "Bird.hpp"
#include "Vertex.hpp"

struct Pipe
{
    static constexpr float gap = 120.0f;
    static constexpr float width = 100.0f;
    static constexpr std::size_t vertex_count = 12;
    static constexpr std::size_t count = 3;
    static constexpr float speed = 100.0f;

    [[nodiscard]] static constexpr std::array<Vertex, vertex_count> vertices(const float height) noexcept
    {
        const auto half = (height - gap) / 2.0f;
        return {
            Vertex{.position = {0.0f, -half}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {0.0f, half}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, half}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, -half}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {0.0f, -half}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, half}, .color = {0.0f, 0.8f, 0.0f}},
            // Gap.
            Vertex{.position = {0.0f, half + gap}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {0.0f, half + gap + half * 2}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, half + gap + half * 2}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, half + gap}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {0.0f, half + gap}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, half + gap + half * 2}, .color = {0.0f, 0.8f, 0.0f}},
        };
    }

    glm::vec2 translation{};

    [[nodiscard]] constexpr float bottom_distance(const Bird &bird, const float height) const noexcept
    {
        const auto half = (height - gap) / 2.0f;
        return glm::distance(bird.translation, translation + glm::vec2{ 0.0f, half + gap });
    }

    [[nodiscard]] constexpr float top_distance(const Bird &bird, const float height) const noexcept
    {
        const auto half = (height - gap) / 2.0f;
        return glm::distance(bird.translation, translation + glm::vec2{ 0.0f, half });
    }

    [[nodiscard]] constexpr bool collide(const Bird &bird, const float height) const noexcept
    {
        const auto half = (height - gap) / 2.0f;
        return
            bird.translation.x + Bird::radius > translation.x &&
            bird.translation.x - Bird::radius < translation.x + width &&
            (bird.translation.y + Bird::radius > translation.y + half + gap ||
                bird.translation.y - Bird::radius < translation.y + half);
    }
};
