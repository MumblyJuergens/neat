#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <glm/vec2.hpp>
#include <glube/glube.hpp>
#include <mj/timer.hpp>
#include <mj/console.hpp>
#include <mj/lambda.hpp>
#include <neat/neat.hpp>
#include <neat/neatdrawgl.hpp>
#include "Config.hpp"
#include "Pipe.hpp"
#include "Sim.hpp"

namespace flappy
{
    struct Flappy
    {
        glube::Window window{ Config::window_width, Config::window_height, "Flappy" };
        glube::Buffer pipes_vbo;
        glube::Buffer birds_vbo;
        glube::BuildingBuffer<glm::vec2> pipes_mbo;
        glube::BuildingBuffer<glm::vec2> birds_mbo;
        glube::Attributes pipe_vao;
        glube::Attributes bird_vao;
        glube::Program program;
        mj::Timer<float> timer;
        std::array<Pipe, Pipe::count> pipes;

        struct GameHandler
        {
            GameHandler() = default;
            GameHandler(const GameHandler &) = delete;
            GameHandler operator=(const GameHandler &) = delete;
            virtual ~GameHandler() = default;
            virtual void key_handler(glube::KeyEvent) {};
            virtual void step(const float delta, Flappy &) = 0;
            virtual void post_step(Flappy &) {};
        };

        // Simulating...
        struct SimHandler : public GameHandler
        {
            std::optional<neat::Population> population;
            std::any userData;
            static constexpr int diagram_width = Config::window_width / 10;
            static constexpr int diagram_height = Config::window_height / 10;
            neat::draw::gl::Diagrammer diagrammer{ {diagram_width, diagram_height} };
            int champId = -1;

            static auto simulationFactory() { return std::make_shared<Sim>(); }

            SimHandler(Flappy &floppy)
                :userData{ UserData{.pipes = &floppy.pipes, .birds_mbo = &floppy.birds_mbo, .delta = {} } }
            {
                using namespace neat::literals;
                neat::Config cfg{
                    .setup_input_nodes = 4,
                    .setup_output_nodes = 1,
                    .setup_inital_connection_rate = 0.0_r,
                    .species_compatability_threshold = 5.0_r,
                };
                population.emplace(simulationFactory, cfg, &userData);

                population->set_stats_string_handler([&](const std::string &stats) {
                    mj::clear_console_properly();
                    std::print("{}", stats);
                    });
            }

            void step(const float delta, Flappy &) override
            {
                std::any_cast<UserData &>(userData).delta = delta;
                population->step();
            }

            void post_step(Flappy &floppy) override
            {
                if (population->generation_is_done())
                {
                    population->new_generation();
                    floppy.reset();
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

        };

        struct PlayingHandler : public GameHandler
        {
            Bird playableBird{ .translation = {Bird::starting_x, Config::window_heightf / 2.0f} };
            static inline bool upPressed{};

            void key_handler(glube::KeyEvent event) override
            {
                if (event.key == glube::Key::up && event.action == glube::KeyAction::pressed)
                {
                    upPressed = true;
                }
                if (event.key == glube::Key::up && event.action == glube::KeyAction::released)
                {
                    upPressed = false;
                }
            }

            void step(const float delta, Flappy &floppy) override
            {
                if (game_logic(playableBird, upPressed, delta, floppy.pipes[0]))
                {
                    floppy.reset();
                    playableBird.reset();
                }

                floppy.birds_mbo.push_back(playableBird.translation);

            }
        };

        std::unique_ptr<GameHandler> handler;

        bool is_finished() const { return window.should_close(); }

        void rebuild_pipes_mbo()
        {
            std::ranges::transform(pipes, std::back_inserter(pipes_mbo), &Pipe::translation);
            pipes_mbo.set_and_clear();
        };

        void init(const bool simulate)
        {
            using namespace neat::literals;

            if (simulate)
            {
                handler = std::make_unique<SimHandler>(*this);
            }
            else
            {
                handler = std::make_unique<PlayingHandler>();
            }

            window.set_key_event_handler(mj::capturing_lambda_pointer([&](glube::KeyEvent event)
                {
                    if (event.key == glube::Key::escape && event.action == glube::KeyAction::pressed)
                    {
                        event.window->set_should_close(true);
                    }
                    handler->key_handler(event);
                }));
            window.swap_interval(1);

            reset();

            {
                const auto pipeVertices = Pipe::vertices(Config::window_heightf);
                pipes_vbo.init(sizeof(Vertex) * pipeVertices.size(), pipeVertices.data());

                const auto birdVertices = Bird::vertices();
                birds_vbo.init(sizeof(Vertex) * birdVertices.size(), birdVertices.data());
            }

            pipes_mbo.reserve(Pipe::count);
            rebuild_pipes_mbo();

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

            struct dummy { glm::vec2 translation; };

            pipe_vao.add(program, pipes_vbo, nameof(&Vertex::position), &Vertex::position);
            pipe_vao.add(program, pipes_vbo, nameof(&Vertex::color), &Vertex::color);
            pipe_vao.add(program, pipes_mbo.buffer(), nameof(&dummy::translation), &dummy::translation, glube::AttributeConfig{ .binding_index = 1, .divisor = 1, });

            bird_vao.add(program, birds_vbo, nameof(&Vertex::position), &Vertex::position);
            bird_vao.add(program, birds_vbo, nameof(&Vertex::color), &Vertex::color);
            bird_vao.add(program, birds_mbo.buffer(), nameof(&dummy::translation), &dummy::translation, glube::AttributeConfig{ .binding_index = 1, .divisor = 1, });
        }

        void reset()
        {
            for (std::size_t i{}; i < Pipe::count; ++i)
            {
                pipes[i].renew_at(Config::window_widthf / Pipe::count * static_cast<float>(i + 1));
            }
        }

        void step()
        {
            timer.tick();
            glClear(GL_COLOR_BUFFER_BIT);

            for (std::size_t i{}; i < pipes.size(); ++i)
            {
                auto &pipe = pipes[i];
                pipe.translation.x -= Pipe::speed * timer.elapsed();
                if (pipe.translation.x < -Pipe::width)
                {
                    pipe.renew_at(Config::window_widthf);
                }
            }
            if (pipes[0].translation.x + Pipe::width < Bird::starting_x)
            {
                // std::println("rotating...");
                std::ranges::rotate(pipes, std::next(pipes.begin()));
            }


            handler->step(timer.elapsed(), *this);

            rebuild_pipes_mbo();
            birds_mbo.set_and_clear();


            program.activate();

            pipe_vao.activate();
            glDrawArraysInstanced(GL_TRIANGLES, 0, Pipe::vertex_count, Pipe::count);

            bird_vao.activate();
            glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, Bird::vertex_count, birds_mbo.size_set());

            handler->post_step(*this);

            window.poll_events();
            window.swap_buffers();
        }

    };

} // End namespace flappy.