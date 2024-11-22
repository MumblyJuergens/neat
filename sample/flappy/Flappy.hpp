#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <glm/vec2.hpp>
#include <glube/glube.hpp>
#include <mj/timer.hpp>
#include <mj/console.hpp>
#include "Config.hpp"
#include "Pipe.hpp"

namespace flappy
{
    struct Flappy
    {
        glube::Window window{ Config::window_width, Config::window_height, "Flappy" };
        glube::Buffer pipes_vbo;
        glube::Buffer birds_vbo;
        glube::Buffer pipes_mbo;
        glube::AutoBuffer birds_mbo;
        std::vector<glm::vec2> pipe_mbo_data;
        std::vector<glm::vec2> birds_mbo_data;
        glube::Attributes pipe_vao;
        glube::Attributes bird_vao;
        glube::Program program;
        mj::Timer<float> timer;
        std::size_t bird_count{};
        std::array<Pipe, Pipe::count> pipes;
        std::optional<neat::Population> population;
        Bird playableBird{ .translation = {Bird::starting_x, Config::window_heightf / 2.0f} };
        float playableBirdVertSpeed{};
        std::any userData = UserData{ .pipes = &pipes, .birds_mbo_data = &birds_mbo_data, .delta = {} };
        static constexpr int diagram_width = Config::window_width / 10;
        static constexpr int diagram_height = Config::window_height / 10;
        neat::draw::gl::Diagrammer diagrammer{ {diagram_width, diagram_height} };
        int speciesCount{};
        int champId = -1;
        static inline bool upPressed{};

        bool is_finished() const { return window.should_close(); }

        void rebuild_pipes_mbo()
        {
            pipe_mbo_data.clear();
            std::ranges::transform(pipes, std::back_inserter(pipe_mbo_data), &Pipe::translation);
            pipes_mbo.overwrite(sizeof(glm::vec2) * pipe_mbo_data.size(), pipe_mbo_data.data());
        };

        void upload_bird_mbo()
        {
            bird_count = birds_mbo_data.size();
            birds_mbo.set(sizeof(glm::vec2) * birds_mbo_data.size(), birds_mbo_data.data());
            birds_mbo_data.clear();
        }

        static void key_handler(glube::KeyEvent event)
        {
            if (event.key == glube::Key::escape && event.action == glube::KeyAction::pressed)
            {
                event.window->set_should_close(true);
            }
            if (event.key == glube::Key::up && event.action == glube::KeyAction::pressed)
            {
                upPressed = true;
            }
            if (event.key == glube::Key::up && event.action == glube::KeyAction::released)
            {
                upPressed = false;
            }
        }

        static auto simulationFactory() { return std::make_shared<Sim>(); };

        void init(const bool simulate)
        {
            using namespace neat::literals;

            window.set_key_event_handler(key_handler);
            window.swap_interval(1);


            if (simulate)
            {
                neat::Config cfg{
        .setup_input_nodes = 4,
        .setup_output_nodes = 1,
        .setup_inital_connection_rate = 0.0_r,
        .species_compatability_threshold = 5.0_r,
                };
                population.emplace(simulationFactory, cfg, &userData);
            }

            reset();

            {
                const auto pipeVertices = Pipe::vertices(Config::window_heightf);
                pipes_vbo.init(sizeof(Vertex) * pipeVertices.size(), pipeVertices.data());
                pipes_mbo.init(sizeof(glm::vec2) * Pipe::count);
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

            const auto projection = glm::ortho(0.0f, Config::window_widthf, 0.0f, Config::window_heightf, -1.0f, 1.0f);
            program.set_uniform(nameof(projection), projection);

            pipe_vao.add(program, pipes_vbo, nameof(&Vertex::position), &Vertex::position);
            pipe_vao.add(program, pipes_vbo, nameof(&Vertex::color), &Vertex::color);
            pipe_vao.add(program, pipes_mbo, nameof(&Pipe::translation), &Pipe::translation, glube::AttributeConfig{ .binding_index = 1, .divisor = 1, });

            bird_vao.add(program, birds_vbo, nameof(&Vertex::position), &Vertex::position);
            bird_vao.add(program, birds_vbo, nameof(&Vertex::color), &Vertex::color);
            bird_vao.add(program, birds_mbo.buffer(), nameof(&Bird::translation), &Bird::translation, glube::AttributeConfig{ .binding_index = 1, .divisor = 1, });

            if (population)
            {
                population->set_stats_string_handler([&](const std::string &stats) {
                    if (population->species_count() < speciesCount)
                    {
                        mj::clear_console_properly();
                    }
                    else
                    {
                        mj::clear_console();
                    }
                    speciesCount = population->species_count();
                    std::print("{}", stats);
                    });
            }

        }

        void reset()
        {
            pipes[0].translation = { Config::window_widthf / Pipe::count, 0.0f };
            pipes[1].translation = { Config::window_widthf / Pipe::count * 2.0f, -200.0f };
            pipes[2].translation = { Config::window_widthf / Pipe::count * 3.0f, 200.0f };
        }

        void step()
        {
            timer.tick();
            std::any_cast<UserData &>(userData).delta = timer.elapsed();
            glClear(GL_COLOR_BUFFER_BIT);

            for (std::size_t i{}; i < pipes.size(); ++i)
            {
                auto &pipe = pipes[i];
                pipe.translation.x -= Pipe::speed * timer.elapsed();
                if (pipe.translation.x < -Pipe::width) pipe.translation.x = Config::window_widthf;
            }
            if (pipes[0].translation.x + Pipe::width < Bird::starting_x)
            {
                // std::println("rotating...");
                std::ranges::rotate(pipes, std::next(pipes.begin()));
            }

            if (population)
            {
                population->step();
            }
            else
            {
                if (upPressed)
                {
                    playableBirdVertSpeed = Bird::fly_speed;
                }
                playableBird.translation.y += playableBirdVertSpeed * timer.elapsed();
                playableBirdVertSpeed -= Bird::gravity * timer.elapsed();

                if (pipes[0].collide(playableBird, Config::window_heightf) || playableBird.translation.y < -Bird::radius || playableBird.translation.y > Config::window_heightf + Bird::radius)
                {
                    reset();
                    playableBird.translation = { Bird::starting_x, Config::window_heightf / 2.0f };
                    playableBirdVertSpeed = 0.0f;
                }

                birds_mbo_data.push_back(playableBird.translation);
            }

            rebuild_pipes_mbo();
            upload_bird_mbo();

            program.activate();

            pipe_vao.activate();
            glDrawArraysInstanced(GL_TRIANGLES, 0, Pipe::vertex_count, Pipe::count);

            bird_vao.activate();
            glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, Bird::vertex_count, bird_count);

            if (population)
            {
                if (population->generation_is_done())
                {
                    population->new_generation();
                    reset();
                }

                if (population->champ_id() != champId)
                {
                    champId = population->champ_id();
                    diagrammer.build_diagram(population->champ());
                }

                glViewport(0, 0, diagram_width, diagram_height);
                diagrammer.draw_diagram();
                glViewport(0, 0, Config::window_width, Config::window_height);

            }
            window.poll_events();
            window.swap_buffers();
        }

    };

} // End namespace flappy.