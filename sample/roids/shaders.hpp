#pragma once

#include <string_view>

namespace roids
{
    static constexpr std::string_view vertex_shader
    {
        R"glsl(#version 460 core
        in vec2 position;
        int mat4 model;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * model * vec4(position, 0.0, 1.0);
        }
        )glsl"
    };

    static constexpr std::string_view fragment_shader
    {
        R"glsl(#version 460 core
        out vec4 ocolor;
        void main() {
            ocolor = vec4(0.8, 0.8, 0.8, 1.0);
        }
        )glsl"
    };

} // End namespace roids.