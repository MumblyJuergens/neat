#include <numbers>
#include <neat/neat.hpp>
#include <glube/glube.hpp>
#include <neat/neatdrawgl.hpp>
#include <mj/timer.hpp>

using namespace neat::literals;

void key_handler(glube::KeyEvent event)
{
    if (event.key == glube::Key::escape && event.action == glube::KeyAction::pressed)
    {
        event.window->set_should_close(true);
    }
}

struct Vertex
{
    glm::vec2 position;
    glm::vec3 color;
};

std::vector<Vertex> triangle_fan_circle(const float radius, const std::size_t sides, const glm::vec3 color)
{
    std::vector<Vertex> vertices;
    vertices.reserve(sides + 2);
    vertices.push_back(Vertex{ .position = {0, 0}, .color = color });

    static constexpr float twopi = std::numbers::pi_v<float> *2;
    const float sidesf = static_cast<float>(sides);
    for (std::size_t i{}; i < sides + 2; ++i)
    {
        const float val = static_cast<float>(i) * twopi / sidesf;
        vertices.push_back(Vertex{
            .position = {
                radius * std::cos(val),
                radius * std::sin(val),
            },
            .color = color,
            });
    }
    return vertices;
}

struct Bird
{
    static constexpr std::size_t vertex_count = 20;
    static constexpr float radius = 25.0f;
    [[nodiscard]] static constexpr std::vector<Vertex> vertices() noexcept
    {
        return triangle_fan_circle(radius, vertex_count - 2, { 0.8f, 0.8f, 0.1f });
    }

    glm::vec2 translation;
};

struct Pipe
{
    static constexpr float gap = 120.0f;
    static constexpr float width = 100.0f;
    static constexpr std::size_t vertex_count = 12;
    static constexpr std::size_t pipe_count = 3;
    [[nodiscard]] static constexpr std::array<Vertex, vertex_count> vertices(const float height) noexcept
    {
        const auto half = (height - gap) / 2.0f;
        return {
            Vertex{.position = {0.0f, -half}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {0.0f, half}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, half}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, -half}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {0.0f, -half}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, half}, .color = {0.0f, 0.8f, 0.0f}},
            // Gap.
            Vertex{.position = {0.0f, half + gap}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {0.0f, half + gap + half * 2}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, half + gap + half * 2}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, half + gap}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {0.0f, half + gap}, .color = {0.0f, 0.8f, 0.0f}},
            Vertex{.position = {width, half + gap + half * 2}, .color = {0.0f, 0.8f, 0.0f}},
        };
    }

    glm::vec2 translation{};

    [[nodiscard]] constexpr float bottom_distance(const Bird &bird, const float height) const noexcept
    {
        const auto half = (height - gap) / 2.0f;
        return glm::distance(bird.translation, translation + glm::vec2{ 0.0f, half + gap });
    }

    [[nodiscard]] constexpr float top_distance(const Bird &bird, const float height) const noexcept
    {
        const auto half = (height - gap) / 2.0f;
        return glm::distance(bird.translation, translation + glm::vec2{ 0.0f, half });
    }

    [[nodiscard]] constexpr bool collide(const Bird &bird, const float height)
    {
        const auto half = (height - gap) / 2.0f;
        return
            bird.translation.x + Bird::radius > translation.x &&
            bird.translation.x - Bird::radius < translation.x + width &&
            (bird.translation.y + Bird::radius > translation.y + half + gap ||
                bird.translation.y - Bird::radius < translation.y + half);
    }
};

void clear_console() { std::print("\x1B[3J\x1B[H"); }
void clear_console_properly() { std::print("\x1B[2J\x1B[H"); }

struct Sim : public neat::Simulation
{
    static constexpr int window_width = 1280;
    static constexpr int window_height = 720;
    static constexpr float window_heightf = 720.0f;
    static constexpr float window_widthf = 1280.0f;
    static constexpr float pipeSpeed = 100.0f;
    static constexpr float birdGravity = 700.0f;
    static constexpr float birdFlySpeed = 200.0f;
    static constexpr float birdX = 200.0f;
    static constexpr float spacing = window_widthf / Pipe::pipe_count;

    static inline glube::Window window{ Sim::window_width, Sim::window_height, "Flappy" };
    static inline glube::Buffer pipes_vbo;
    static inline glube::Buffer birds_vbo;
    static inline glube::Buffer pipes_mbo;
    static inline glube::AutoBuffer birds_mbo;
    static inline std::vector<glm::vec2> pipe_mbo_data;
    static inline std::vector<glm::vec2> birds_mbo_data;
    static inline glube::Attributes pipe_vao;
    static inline glube::Attributes bird_vao;
    static inline glube::Program program;
    static inline mj::Timer<float> timer;
    static inline std::size_t bird_count{};

    static inline std::array<Pipe, Pipe::pipe_count> pipes;

    Bird bird{ .translation = {birdX, window_heightf / 2.0f} };
    float birdVertSpeed = 0.0f;

    static void rebuild_pipes_mbo()
    {
        pipe_mbo_data.clear();
        std::ranges::transform(pipes, std::back_inserter(pipe_mbo_data), &Pipe::translation);
        pipes_mbo.overwrite(sizeof(glm::vec2) * pipe_mbo_data.size(), pipe_mbo_data.data());
    };

    static void upload_bird_mbo()
    {
        bird_count = birds_mbo_data.size();
        birds_mbo.set(sizeof(glm::vec2) * birds_mbo_data.size(), birds_mbo_data.data());
        birds_mbo_data.clear();
    }

    static void init()
    {
        window.set_key_event_handler(key_handler);
        window.swap_interval(1);

        reset();

        {
            const auto pipeVertices = Pipe::vertices(window_heightf);
            pipes_vbo.init(sizeof(Vertex) * pipeVertices.size(), pipeVertices.data());
            pipes_mbo.init(sizeof(glm::vec2) * Pipe::pipe_count);
            rebuild_pipes_mbo();

            const auto birdVertices = Bird::vertices();
            birds_vbo.init(sizeof(Vertex) * birdVertices.size(), birdVertices.data());
        }

        glube::Shader vertShader{ glube::ShaderType::vertex,
        R"glsl(#version 460 core
        in vec2 position;
        in vec2 translation;
        in vec3 color;
        out vec3 vcolor;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * vec4(position + translation, 0.0, 1.0);
            vcolor = color;
        }
        )glsl" };

        glube::Shader fragShader{ glube::ShaderType::fragment,
        R"glsl(#version 460 core
        in vec3 vcolor;
        out vec4 ocolor;
        void main() {
            ocolor = vec4(vcolor, 1.0f);
        }
        )glsl" };

        program.add_shaders(vertShader, fragShader);

        const auto projection = glm::ortho(0.0f, window_widthf, 0.0f, window_heightf, -1.0f, 1.0f);
        program.set_uniform(nameof(projection), projection);

        pipe_vao.add(program, pipes_vbo, nameof(&Vertex::position), &Vertex::position);
        pipe_vao.add(program, pipes_vbo, nameof(&Vertex::color), &Vertex::color);
        pipe_vao.add(program, pipes_mbo, nameof(&Pipe::translation), &Pipe::translation, glube::AttributeConfig{ .binding_index = 1, .divisor = 1, });

        bird_vao.add(program, birds_vbo, nameof(&Vertex::position), &Vertex::position);
        bird_vao.add(program, birds_vbo, nameof(&Vertex::color), &Vertex::color);
        bird_vao.add(program, birds_mbo.buffer(), nameof(&Bird::translation), &Bird::translation, glube::AttributeConfig{ .binding_index = 1, .divisor = 1, });
    }

    static void reset()
    {
        pipes[0].translation = { spacing, 0.0f };
        pipes[1].translation = { spacing * 2.0f, -200.0f };
        pipes[2].translation = { spacing * 3.0f, 200.0f };
    }

    static void pre_step()
    {
        timer.tick();
        glClear(GL_COLOR_BUFFER_BIT);

        for (std::size_t i{}; i < pipes.size(); ++i)
        {
            auto &pipe = pipes[i];
            pipe.translation.x -= pipeSpeed * timer.elapsed();
            if (pipe.translation.x < -Pipe::width) pipe.translation.x = window_widthf;
        }
        if (pipes[0].translation.x + Pipe::width < birdX)
        {
            // std::println("rotating...");
            std::ranges::rotate(pipes, std::next(pipes.begin()));
        }

        rebuild_pipes_mbo();
        upload_bird_mbo();

        program.activate();

        pipe_vao.activate();
        glDrawArraysInstanced(GL_TRIANGLES, 0, Pipe::vertex_count, Pipe::pipe_count);

        bird_vao.activate();
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, Bird::vertex_count, bird_count);

    }

    void step(neat::SimulationInfo &info) override
    {
        const float input1 = window_heightf / pipes[0].top_distance(bird, window_heightf);
        const float input2 = window_heightf / pipes[0].bottom_distance(bird, window_heightf);
        const float input3 = window_heightf / bird.translation.y;
        info.assign_inputs(1.0f, input1, input2, input3);
        info.run(std::tanh);
        const bool up = info.outputs[0] > 0.9f;

        if (up)
        {
            birdVertSpeed = birdFlySpeed;
        }
        bird.translation.y += birdVertSpeed * timer.elapsed();
        birdVertSpeed -= birdGravity * timer.elapsed();

        if (pipes[0].collide(bird, window_heightf) || bird.translation.y < -Bird::radius || bird.translation.y > window_heightf + Bird::radius)
        {
            info.is_done = true;
        }
        else
        {
            info.fitness += 0.1f;
        }

        birds_mbo_data.push_back(bird.translation);

    }

};

int main()
{
    Sim::init();

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