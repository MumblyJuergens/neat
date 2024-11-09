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
    bool m_is_perfect{};
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

    void supply(std::span<float> inputs) override
    {
        assert(inputs.size() == 3);
        inputs[0] = 1.0f; // Bias.
        inputs[1] = expecteds[i].in0;
        inputs[2] = expecteds[i].in1;
    }

    [[nodiscard]] float receive(const std::span<float> outputs, const float fitness) override
    {
        assert(outputs.size() == 1);
        const auto ret = fitness + (
            (expecteds[i].out > 0.5f && outputs[0] > 0.5f)
            || (expecteds[i].out < 0.5f && outputs[0] < 0.5f));// (1.0f - mj::difference(outputs[0], expecteds[i].out));
        ++i;
        m_is_perfect = ret > 3.5f;
        return ret;
    }

    [[nodiscard]] bool is_done() const override
    {
        return i == expecteds.size();
    }

    [[nodiscard]] virtual bool is_perfect() const override { return m_is_perfect; }
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
            if (population.champ().simulation().is_perfect())
            {
                clear_console_properly();
                std::println("Perfection!");
                std::println("{}", population.champ().chart());
                break;
            }
            population.new_generation();
        }

        window.swap_buffers();
        window.poll_events();
    }
}