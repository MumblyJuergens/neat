#include <array>
#include <neat/neat.hpp>
#include <glube/glube.hpp>

void key_callback(glube::KeyEvent event)
{
    if (event.key == glube::Key::escape && event.action == glube::KeyAction::pressed)
    {
        event.window->set_should_close(true);
    }
}

int main()
{
    glube::Window window{1280, 720, "NEAT Sample"};
    window.set_key_event_handler(key_callback);

    std::array<neat::Neuron, 2> inputs{
        neat::Neuron{neat::next_global_innovation_number(), neat::NeuronType::input},
        neat::Neuron{neat::next_global_innovation_number(), neat::NeuronType::input},
    };
    std::array<neat::Neuron, 1> outputs{
        neat::Neuron{neat::next_global_innovation_number(), neat::NeuronType::output},
    };

    neat::Population population{inputs, outputs, 100, {}};

    // float result{};
    // auto fitnessFunction = [&result](std::span<float> output)
    // {
    //     return output[0] >= 0.5f ? 1.0f : 0.0f;
    // };

    while (!window.should_close())
    {

        // population.resolve(fitnessFunction, {1.0f, 0.0f}, &result, 1.0f);

        window.swap_buffers();
        window.poll_events();
    }
}