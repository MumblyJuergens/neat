#pragma once

#include "config.hpp"
#include "food.hpp"
#include "neat/Config.hpp"
#include "neat/Genome.hpp"
#include "neat/Random.hpp"
#include "neat/SimplePopulation.hpp"
#include "sdlmath.hpp"
#include "snake.hpp"
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_timer.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mjsdl/Renderer.hpp>
#include <mjsdl/Window.hpp>
#include <print>
#include <vector>

namespace snakesdl
{

struct Game
{
    mjsdl::Window window;
    mjsdl::Renderer renderer;
    std::vector<Snake> snakes;
    std::vector<size_t> current_food;
    std::unique_ptr<neat::SimplePopulation> population;
    double fps{};
    int frame{};
    Food food;
    bool render{true};
    bool vsync{true};
    bool champ_only{};

    static constexpr int POPULATION_SIZE = 1000;

    void init()
    {
        std::tie(window, renderer) = mjsdl::Renderer::create_window_and_renderer(
            "NEAT Snake SDL3 - snakesdl", config::WINDOW_SIZE, config::WINDOW_SIZE, SDL_WINDOW_RESIZABLE);
        SDL_SetRenderScale(renderer.get(), config::POINT_SIZE, config::POINT_SIZE);
        set_vsync(true);

        snakes.resize(POPULATION_SIZE);
        current_food.resize(POPULATION_SIZE, 0uz);

        neat::Config cfg{
            .setup_population_size = POPULATION_SIZE,
            .setup_input_nodes = 10,
            .setup_output_nodes = 2,
            .setup_inital_connection_rate = 0.0f,
            .mutate_new_connection_rate = 2.0f,
            .mutate_new_node_rate = 0.5f,
        };
        population = std::make_unique<neat::SimplePopulation>(cfg);
    }

    void iterate(double delta)
    {
        if (frame++ % 100 == 0) {
            fps = 1.0 / (delta / SDL_MS_PER_SECOND);
        }

        population->step([this](std::vector<neat::Genome> &genomes) {
            if (genomes.size() != snakes.size()) {
                std::println("Resizing snakes for genomes: {}", genomes.size());
                snakes.resize(genomes.size());
                current_food.resize(genomes.size(), 0uz);
            }

            std::vector<float> outputs(3uz, 0.0f);
            size_t i = 0;
            for (auto &genome : genomes) {
                if (genome.simulation_is_done()) {
                    ++i;
                    continue;
                }
                Snake &snake = snakes[i];
                const auto snake_food = food[current_food[i]];
                const auto foodchecker = [snake_food](SDL_FPoint p) {
                    return mjsdl::math::equal_within_ulps(p, snake_food, 1);
                };
                std::vector<float> inputs{
                    snake.raycast(SnakeTurn::TURN_LEFT, Snake::tail_collide, snake),
                    snake.raycast(SnakeTurn::TURN_NOPE, Snake::tail_collide, snake),
                    snake.raycast(SnakeTurn::TURN_RIGHT, Snake::tail_collide, snake),
                    snake.raycast(SnakeTurn::TURN_LEFT, Snake::just_wall),
                    snake.raycast(SnakeTurn::TURN_NOPE, Snake::just_wall),
                    snake.raycast(SnakeTurn::TURN_RIGHT, Snake::just_wall),
                    snake.raycast(SnakeTurn::TURN_LEFT, foodchecker),
                    snake.raycast(SnakeTurn::TURN_NOPE, foodchecker),
                    snake.raycast(SnakeTurn::TURN_RIGHT, foodchecker),
                    static_cast<float>(snake.points.size()) * 0.01f,
                };

                if (mjsdl::math::equal_within_ulps(snake.points[0], snake_food, 1)) {
                    snake.fed = true;
                    snake.fitness += 3.0f;
                    current_food[i] += 1;
                }
                snake.fitness += 0.001f;
                genome.set_fitness(snake.fitness);
                snake.is_champ = genome.is_current_champ();

                genome.simple_step(inputs, outputs, std::tanh);
                if (outputs[0] > 0.05f) snake.turn_left();
                if (outputs[1] > 0.05f) snake.turn_right();
                ++i;
                snake.move();
                if (snake.dead) {
                    genome.set_simulation_is_done(true);
                }
            }
        });

        if (render) {

            SDL_SetRenderDrawColor(renderer.get(), 0, 0, 0, SDL_ALPHA_OPAQUE);
            SDL_RenderClear(renderer.get());

            SDL_SetRenderDrawColor(renderer.get(), 255, 0, 0, SDL_ALPHA_OPAQUE);
            SDL_FRect border{
                .x = config::BORDER_POINTS,
                .y = config::BORDER_POINTS,
                .w = config::GAME_SIZE,
                .h = config::GAME_SIZE,
            };
            SDL_RenderRect(renderer.get(), &border);

            for (size_t i{}; i < snakes.size(); ++i) {
                const auto &snake = snakes[i];
                const auto food_pos = food[current_food[i]];
                if (!snake.dead && (!champ_only || snake.is_champ)) {
                    snake.draw(renderer);
                    SDL_RenderPoint(renderer.get(), food_pos.x, food_pos.y);
                }
            }

            SDL_SetRenderScale(renderer.get(), 1.0f, 1.0f);
            SDL_RenderDebugTextFormat(renderer.get(), 20, 20, "FPS: %f", fps);
            SDL_SetRenderScale(renderer.get(), config::POINT_SIZE, config::POINT_SIZE);

            SDL_RenderPresent(renderer.get());
        }

        if (population->generation_is_done()) {
            std::ranges::for_each(snakes, &Snake::reset);
            current_food.clear();
            current_food.resize(POPULATION_SIZE, 0uz);
            food.reset();
            population->new_generation();
            std::println("Generation: {}", population->generation());
        }
    }

    void set_vsync(bool on) { SDL_SetRenderVSync(renderer.get(), on ? 1 : SDL_RENDERER_VSYNC_DISABLED); }

    SDL_AppResult on_event(SDL_Event *const event)
    {
        if (event->type == SDL_EVENT_QUIT) {
            return SDL_APP_SUCCESS;
        }

        if (event->type == SDL_EVENT_KEY_DOWN) {
            if (event->key.key == SDLK_R) {
                render = !render;
                std::println("Render: {}", render);
            }
            if (event->key.key == SDLK_V) {
                vsync = !vsync;
                set_vsync(vsync);
                std::println("VSync: {}", vsync);
            }
            if (event->key.key == SDLK_C) {
                champ_only = !champ_only;
                std::println("Champ Only: {}", champ_only);
            }
        }

        return SDL_APP_CONTINUE;
    }
};

} // namespace snakesdl