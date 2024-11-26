#pragma once

#include <string_view>

namespace snake
{

    static constexpr std::string_view vertex_shader
    {
        R"glsl(#version 460 core
        in vec2 position;
        in vec4 color;
        out vec4 vcolor;
        uniform mat4 projection;
        uniform float pointsize;
        void main() {
            vcolor = color;
            gl_Position = projection * vec4(position, 0.0, 1.0);
            gl_PointSize = pointsize;
        }
        )glsl"
    };

    static constexpr std::string_view fragment_shader
    {
        R"glsl(#version 460 core
        in vec4 vcolor;
        out vec4 ocolor;
        void main() {
            ocolor = vcolor;
        }
        )glsl"
    };

} // End namespace snake.