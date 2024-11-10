#include <array>
#include <print>
#include <neat/neat.hpp>
#include <glube/glube.hpp>

static inline void key_callback(glube::KeyEvent event)
{
    if (event.key == glube::Key::escape && event.action == glube::KeyAction::pressed)
    {
        event.window->set_should_close(true);
    }
}

static inline void clear_console() { std::print("\x1B[3J\x1B[H"); }
static inline void clear_console_properly() { std::print("\x1B[2J\x1B[H"); }

struct Sim : public neat::Simulation
{
    std::size_t i{};
    // bool m_is_perfect{};
    struct Expected
    {
        float in0, in1, out;
    };
    static inline std::array<Expected, 4> expecteds{
        Expected{0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f},
    };

    void step(neat::SimulationInfo &info) override
    {
        info.assign_inputs(1.0f, expecteds[i].in0, expecteds[i].in1);
        info.run();
        info.fitness += (
            (expecteds[i].out > 0.5f && info.outputs[0] > 0.5f)
            || (expecteds[i].out < 0.5f && info.outputs[0] < 0.5f));
        info.is_perfect = info.fitness > 3.5f;

        info.is_done = ++i == expecteds.size();

    }
};

int main()
{
    glube::Window window{ 1280, 720, "NEAT Sample" };
    window.set_key_event_handler(key_callback);

    auto simulationFactory = []() { return std::make_shared<Sim>(); };

    neat::Population population{ 3, 1, 100, simulationFactory, neat::Config{.absolute_difference = false, .compatability_threshold = 1.0f, .fully_connect = true} };
    population.set_stats_string_handler([&population, sc{ 0ull }](const std::string stats) mutable
        {
            if (population.species_count() < sc)
            {
                clear_console_properly();
            }
            else
            {
                clear_console();
            }
            sc = population.species_count();
            std::print("{}", stats);
        });

    while (!window.should_close())
    {
        population.step();

        clear_console();

        if (population.generation_is_done())
        {
            if (population.champ().fitness() > 3.5f)
            {
                clear_console_properly();
                std::println("Perfection! Generation {}", population.generation());
                std::println("{}", population.champ().chart());
                break;
            }
            population.new_generation();
        }

        window.swap_buffers();
        window.poll_events();
    }
}