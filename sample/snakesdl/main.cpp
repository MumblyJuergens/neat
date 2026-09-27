#include "game.hpp"
#include <SDL3/SDL_init.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <print>
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>
#include <memory>

static uint64_t prevms{}, currentms{};

SDL_AppResult SDL_AppInit(void **appstate, [[maybe_unused]] int argc, [[maybe_unused]] char *argv[])
{
    uint32_t seed = std::random_device{}();
    uint32_t generations_limit = 0;
    bool headless = false;
    if (argc >= 2) {
        const auto result = std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), seed);
        if (result.ec != std::errc()) {
            std::println("Bad seed value, must be positive integer");
            return SDL_APP_FAILURE;
        }
        std::println("Running with seed value {}", seed);
    }
    if (argc >= 3) {
        const auto result = std::from_chars(argv[2], argv[2] + std::strlen(argv[2]), generations_limit);
        if (result.ec != std::errc()) {
            std::println("Bad generations limit value, must be positive integer");
            return SDL_APP_FAILURE;
        }
        std::println("Running for {} generations", generations_limit);
    }
    if (argc == 4 && std::strcmp(argv[3], "headless") == 0) {
        headless = true;
    }

    auto game = std::make_unique<snakesdl::Game>();

    prevms = SDL_GetTicks();
    game->init(seed, generations_limit, headless);

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