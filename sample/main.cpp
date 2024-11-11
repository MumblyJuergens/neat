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
    std::size_t step_count{};
    static constexpr std::size_t max_steps = 100;

    struct Entry
    {
        float i0;
        float i1;
        float a;
    };
    static constexpr std::array<Entry, 4> data{
        Entry{0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0},
    };


    [[nodiscard]] static constexpr bool to_bool(const float f) noexcept { return f > 0.5f; }

    void step(neat::SimulationInfo &info) override
    {
        info.assign_inputs(1.0f, data[step_count].i0, data[step_count].i1);

        info.run();

        info.fitness += to_bool(info.outputs[0]) == to_bool(data[step_count].a);
        info.is_perfect = info.fitness > 3.9f;
        info.is_done = ++step_count == std::size(data);
    }
};

int main()
{
    glube::Window window{ 1280, 720, "NEAT Sample" };
    window.set_key_event_handler(key_callback);

    auto simulationFactory = []() { return std::make_shared<Sim>(); };

    neat::Population population{ 3, 1, 100, simulationFactory, neat::Config{.species_compatability_threshold = 5.0f, .fully_connect = true} };
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
            if (population.champ().simulation_is_perfect())
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