#include <any>
#include <neat/neat.hpp>
#include <glube/glube.hpp>
#include <neat/neatdrawgl.hpp>
#include "Sim.hpp"

using namespace neat::literals;

void key_handler(glube::KeyEvent event)
{
    if (event.key == glube::Key::escape && event.action == glube::KeyAction::pressed)
    {
        event.window->set_should_close(true);
    }
}

void clear_console() { std::print("\x1B[3J\x1B[H"); }
void clear_console_properly() { std::print("\x1B[2J\x1B[H"); }



int main()
{
    using namespace flappy;

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

    Sim::init();

    Sim::window.set_key_event_handler(key_handler);
    Sim::window.swap_interval(1);

    static constexpr int diagram_width = Sim::window_width / 10;
    static constexpr int diagram_height = Sim::window_height / 10;
    neat::draw::gl::Diagrammer diagrammer{ {diagram_width, diagram_height} };

    auto simulationFactory = []() { return std::make_shared<Sim>(); };


    neat::Config cfg{
        .setup_input_nodes = 4,
        .setup_output_nodes = 1,
        .setup_inital_connection_rate = 0.0_r,
        .species_compatability_threshold = 5.0_r,
    };

    // TODO: UserData!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    neat::Population population{ simulationFactory, cfg };
    int speciesCount{};
    int champId = -1;

    population.set_stats_string_handler([&](const std::string &stats) {
        if (population.species_count() < speciesCount)
        {
            clear_console_properly();
        }
        else
        {
            clear_console();
        }
        speciesCount = population.species_count();
        std::print("{}", stats);
        });

    while (!Sim::window.should_close())
    {
        Sim::pre_step();
        population.step();

        if (population.generation_is_done())
        {
            population.new_generation();
            Sim::reset();
        }

        if (population.champ_id() != champId)
        {
            champId = population.champ_id();
            diagrammer.build_diagram(population.champ());
        }

        glViewport(0, 0, diagram_width, diagram_height);
        diagrammer.draw_diagram();
        glViewport(0, 0, Sim::window_width, Sim::window_height);

        Sim::window.poll_events();
        Sim::window.swap_buffers();
    }
}