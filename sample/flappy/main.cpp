#include <any>
#include <neat/neat.hpp>
#include <glube/glube.hpp>
#include <neat/neatdrawgl.hpp>
#include "Sim.hpp"
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