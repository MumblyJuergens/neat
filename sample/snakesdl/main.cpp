#include "CLI/CLI.hpp"
#include "game.hpp"
#include <SDL3/SDL_init.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <print>
#define SDL_MAIN_USE_CALLBACKS 1
#include <CLI/CLI.hpp>
#include <SDL3/SDL_main.h>
#include <memory>

static uint64_t prevms{}, currentms{};

SDL_AppResult SDL_AppInit(void **appstate, [[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
    int seed = static_cast<int>(std::random_device{}());
    int generations_limit = 0;
    bool headless = false;
    int population_size = 300;

    CLI::App app{"snakesdl"};
    argv = app.ensure_utf8(argv);
    app.add_option("-s,--seed", seed, "Seed value for RNG");
    app.add_option("-g,--generations", generations_limit, "Limit the number of generation to run for");
    app.add_flag("--headless", headless, "Run without rendering or vsync");
    app.add_option("-p,--population", population_size, "The suggested population per generation");

    try {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &e) {
        return SDL_APP_FAILURE;
    }

    auto game = std::make_unique<snakesdl::Game>();

    prevms = SDL_GetTicks();
    game->init(seed, generations_limit, headless, population_size);

    *appstate = game.release();
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void *appstate, [[maybe_unused]] SDL_Event *event)
{
    snakesdl::Game *const game = reinterpret_cast<snakesdl::Game *>(appstate);

    return game->on_event(event);
}

SDL_AppResult SDL_AppIterate(void *appstate)
{
    try {
        snakesdl::Game *const game = reinterpret_cast<snakesdl::Game *>(appstate);
        currentms = SDL_GetTicks();
        const double delta = static_cast<double>(currentms - prevms);
        prevms = SDL_GetTicks();

        return game->iterate(delta);
    }
    catch (...) {
        std::println("Exiting on error?");
        return SDL_APP_FAILURE;
    }
}

void SDL_AppQuit(void *appstate, [[maybe_unused]] SDL_AppResult result)
{
    std::unique_ptr<snakesdl::Game> game{reinterpret_cast<snakesdl::Game *>(appstate)};
    const auto duration = std::chrono::high_resolution_clock::now() - game->began;
    std::println("Exiting normally: {} | Time: {}", result == SDL_APP_SUCCESS,
                 std::chrono::duration_cast<std::chrono::milliseconds>(duration));
}