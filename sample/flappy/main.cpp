#include "Flappy.hpp"

int main()
{
    flappy::Flappy game;
    game.init(true);

    while (!game.is_finished())
    {
        game.step();
    }
}