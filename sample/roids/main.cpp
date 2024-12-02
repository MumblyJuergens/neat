#include <vector>
#include <glm/mat4x4.hpp>
#include <neat/Simulation.hpp>
#include "ezobj.hpp"

namespace roids
{
    using namespace neat::literals;

    struct Vertex
    {

    };

    struct Roid
    {
        glm::mat4 model;
        glm::vec2 vector;
        int size{};
    };

    struct Ship
    {
    };

    struct Sim : public neat::Simulation
    {
        std::vector<Roid> roids;
        Ship ship;

        // void step(neat::SimulationInfo &info) override
        // {

        // }
    };

} // End namespace roids.

int main()
{
    using namespace roids;

    auto vertices = load2D("sample/roids/roid.obj");


}