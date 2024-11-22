#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include <cstddef>
#include <neat/neat.hpp>
#include <glube/glube.hpp>
#include <mj/timer.hpp>
#include <mj/nameof.hpp>
#include <glm/vec2.hpp>
#include "Bird.hpp"
#include "Pipe.hpp"
#include "Vertex.hpp"
#include "UserData.hpp"
#include "Config.hpp"

namespace flappy
{

    struct Sim : public neat::Simulation
    {


        Bird bird{ .translation = {birdX, window_heightf / 2.0f} };
        float birdVertSpeed = 0.0f;

        static void rebuild_pipes_mbo()
        {
            pipe_mbo_data.clear();
            std::ranges::transform(pipes, std::back_inserter(pipe_mbo_data), &Pipe::translation);
            pipes_mbo.overwrite(sizeof(glm::vec2) * pipe_mbo_data.size(), pipe_mbo_data.data());
        };

        static void upload_bird_mbo()
        {
            bird_count = birds_mbo_data.size();
            birds_mbo.set(sizeof(glm::vec2) * birds_mbo_data.size(), birds_mbo_data.data());
            birds_mbo_data.clear();
        }

        static void init()
        {

            reset();

            {
                const auto pipeVertices = Pipe::vertices(window_heightf);
                pipes_vbo.init(sizeof(Vertex) * pipeVertices.size(), pipeVertices.data());
                pipes_mbo.init(sizeof(glm::vec2) * Pipe::pipe_count);
                rebuild_pipes_mbo();

                const auto birdVertices = Bird::vertices();
                birds_vbo.init(sizeof(Vertex) * birdVertices.size(), birdVertices.data());
            }

            glube::Shader vertShader{ glube::ShaderType::vertex,
            R"glsl(#version 460 core
        in vec2 position;
        in vec2 translation;
        in vec3 color;
        out vec3 vcolor;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * vec4(position + translation, 0.0, 1.0);
            vcolor = color;
        }
        )glsl" };

            glube::Shader fragShader{ glube::ShaderType::fragment,
            R"glsl(#version 460 core
        in vec3 vcolor;
        out vec4 ocolor;
        void main() {
            ocolor = vec4(vcolor, 1.0f);
        }
        )glsl" };

            program.add_shaders(vertShader, fragShader);

            const auto projection = glm::ortho(0.0f, window_widthf, 0.0f, window_heightf, -1.0f, 1.0f);
            program.set_uniform(nameof(projection), projection);

            pipe_vao.add(program, pipes_vbo, nameof(&Vertex::position), &Vertex::position);
            pipe_vao.add(program, pipes_vbo, nameof(&Vertex::color), &Vertex::color);
            pipe_vao.add(program, pipes_mbo, nameof(&Pipe::translation), &Pipe::translation, glube::AttributeConfig{ .binding_index = 1, .divisor = 1, });

            bird_vao.add(program, birds_vbo, nameof(&Vertex::position), &Vertex::position);
            bird_vao.add(program, birds_vbo, nameof(&Vertex::color), &Vertex::color);
            bird_vao.add(program, birds_mbo.buffer(), nameof(&Bird::translation), &Bird::translation, glube::AttributeConfig{ .binding_index = 1, .divisor = 1, });
        }

        static void reset()
        {
            pipes[0].translation = { spacing, 0.0f };
            pipes[1].translation = { spacing * 2.0f, -200.0f };
            pipes[2].translation = { spacing * 3.0f, 200.0f };
        }

        static void pre_step()
        {
            timer.tick();
            glClear(GL_COLOR_BUFFER_BIT);

            for (std::size_t i{}; i < pipes.size(); ++i)
            {
                auto &pipe = pipes[i];
                pipe.translation.x -= pipeSpeed * timer.elapsed();
                if (pipe.translation.x < -Pipe::width) pipe.translation.x = window_widthf;
            }
            if (pipes[0].translation.x + Pipe::width < birdX)
            {
                // std::println("rotating...");
                std::ranges::rotate(pipes, std::next(pipes.begin()));
            }

            rebuild_pipes_mbo();
            upload_bird_mbo();

            program.activate();

            pipe_vao.activate();
            glDrawArraysInstanced(GL_TRIANGLES, 0, Pipe::vertex_count, Pipe::pipe_count);

            bird_vao.activate();
            glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, Bird::vertex_count, bird_count);

        }

        void step(neat::SimulationInfo &info) override
        {
            const float input1 = window_heightf / pipes[0].top_distance(bird, window_heightf);
            const float input2 = window_heightf / pipes[0].bottom_distance(bird, window_heightf);
            const float input3 = window_heightf / bird.translation.y;
            info.assign_inputs(1.0f, input1, input2, input3);
            info.run(std::tanh);
            const bool up = info.outputs[0] > 0.9f;

            if (up)
            {
                birdVertSpeed = birdFlySpeed;
            }
            bird.translation.y += birdVertSpeed * timer.elapsed();
            birdVertSpeed -= birdGravity * timer.elapsed();

            if (pipes[0].collide(bird, window_heightf) || bird.translation.y < -Bird::radius || bird.translation.y > window_heightf + Bird::radius)
            {
                info.is_done = true;
            }
            else
            {
                info.fitness += 0.1f;
            }

            birds_mbo_data.push_back(bird.translation);

        }

    };

} // End namespace flappy.