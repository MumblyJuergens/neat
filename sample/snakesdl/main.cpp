#include "game.hpp"
#include <print>
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>
#include <memory>

static uint64_t prevms{}, currentms{};

SDL_AppResult SDL_AppInit(void **appstate, [[maybe_unused]] int argc, [[maybe_unused]] char *argv[])
{
    auto game = std::make_unique<snakesdl::Game>();

    prevms = SDL_GetTicks();
    game->init();

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

        game->iterate(delta);
    }
    catch (...) {
        std::println("Exiting on error?");
    }
    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void *appstate, [[maybe_unused]] SDL_AppResult result)
{
    std::unique_ptr<snakesdl::Game> game{reinterpret_cast<snakesdl::Game *>(appstate)};
    std::println("Exiting with result {}", static_cast<int>(result));
}