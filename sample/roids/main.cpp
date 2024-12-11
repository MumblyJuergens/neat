#include <vector>
#include <glm/mat4x4.hpp>
#include <neat/neat.hpp>
#include <glube/glube.hpp>
#include "ezobj.hpp"
#include "shaders.hpp"

namespace roids
{
    using namespace neat::literals;

    template<typename T>
    struct Config
    {
        static constexpr T window_width = 1024;
        static constexpr T window_height = 768;
    };

    struct Vertex
    {
        glm::vec2 position{};
    };

    struct Roid
    {
        glm::mat4 model{ glm::identity<glm::mat4>() };
        glm::vec2 vector;
        int size{};
    };

    struct Ship
    {
        glm::mat4 model{ glm::identity<glm::mat4>() };
    };

    struct Sim : public neat::Simulation
    {
        std::vector<Roid> roids;
        Ship ship;

        void step([[maybe_unused]] neat::SimulationInfo &info) override
        {

        }
    };

} // End namespace roids.

int main()
{
    using namespace roids;

    glube::Window window{ Config<int>::window_width, Config<int>::window_height, "Roids" };
    window.swap_interval(1);
    window.set_key_event_handler([&](glube::KeyEvent event) {
        if (event.key == glube::Key::escape) event.window->set_should_close(true);
        });

    glube::BuildingBuffer<glm::vec2> vertexBuffer;
    std::size_t roidVertCount{}, shipVertCount{};
    {
        const auto roidVertices = load2D("sample/roids/roid.obj");
        const auto shipVertices = load2D("sample/roids/ship.obj");
        roidVertCount = roidVertices.size();
        shipVertCount = shipVertices.size();
        vertexBuffer.append_range(roidVertices);
        vertexBuffer.append_range(shipVertices);
        vertexBuffer.set_and_clear();
    }

    glube::BuildingBuffer<glm::mat4> modelBuffer;

    glube::Program program{ vertex_shader, fragment_shader };
    {
        glm::mat4 projection = glm::ortho(0.0f, Config<float>::window_width, 0.0f, Config<float>::window_height);
        program.set_uniform(nameof(projection), projection);
    }

    glube::Attributes attributes;
    {
        struct dummy { glm::mat4 model; };
        attributes.add(program, vertexBuffer.buffer(), nameof(&Vertex::position), &Vertex::position);
        attributes.add(program, modelBuffer.buffer(), nameof(&dummy::model), &dummy::model, glube::AttributeConfig{ .binding_index = 1, .divisor = 1 });
    }

    neat::Population population{ std::make_unique<neat::EasySimulationFactory<Sim>>() };


    modelBuffer.emplace_back(glm::identity<glm::mat4>());
    modelBuffer.emplace_back(glm::identity<glm::mat4>());
    modelBuffer.set_and_clear();

    while (!window.should_close())
    {
        glClear(GL_COLOR_BUFFER_BIT);

        population.step();

        program.activate();
        attributes.activate();

        glDrawArraysInstancedBaseInstance(GL_LINES, roidVertCount, shipVertCount, 1, 0);
        glDrawArraysInstancedBaseInstance(GL_LINES, 0, roidVertCount, 100, 1);

        glDrawArraysInstanced(GL_LINES, roidVertCount, shipVertCount, 1);
        glDrawArraysInstanced(GL_LINES, 0, roidVertCount, modelBuffer.isize_set());

        if (population.generation_is_done())
        {
            population.new_generation();
        }

        window.poll_events();
        window.swap_buffers();
    }
}