#include <algorithm>
#include <array>
#include <fstream>
#include <mdspan>
#include <print>
#include <cstddef>
#include <cstdio>
#include <neat/neat.hpp>
#include <neat/neatdrawgl.hpp>
#include <glube/glube.hpp>
#include <glm/vec2.hpp>
#include <glm/geometric.hpp>
// #define SNAKE_LOGGY
#ifdef SNAKE_LOGGY
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#undef GLM_ENABLE_EXPERIMENTAL
#endif
#include <mj/console.hpp>
#include <mj/nameof.hpp>
#include <mj/lambda.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/json.hpp>
#include "shaders.hpp"

namespace snake
{
    using namespace neat::literals;

#ifdef SNAKE_LOGGY
    std::ofstream loggy{ "loggy.txt" };
#endif

    template<typename T>
    struct Config
    {
        static constexpr T game_size = static_cast<T>(50);
        static constexpr T game_tile_count = game_size * game_size;

        static constexpr T sim_population_count = static_cast<T>(100);
        static constexpr T sim_games_displayed_size = static_cast<T>(10);

        static constexpr T window_champ_pixel_size = static_cast<T>(4);
        static constexpr T window_display_champ_size = game_size * window_champ_pixel_size;
        static constexpr T window_width = game_size * sim_games_displayed_size + window_display_champ_size;
        static constexpr T window_height = game_size * sim_games_displayed_size;

        static constexpr glm::vec<2, T> start_position{ game_size / static_cast<T>(2), game_size / static_cast<T>(2) };
        static constexpr T start_length = static_cast<T>(4);
        static constexpr T auto_grow_size = static_cast<T>(0);
    };

    enum class TileType { none, snake, border, food };

    struct Tile
    {
        TileType type{ TileType::none };
        int value{};

        void reset()
        {
            type = TileType::none;
            value = 0;
        }
    };

    struct Directions
    {
        static constexpr glm::ivec2
            up{ 0, -1 },
            right{ 1, 0 },
            down{ 0, 1 },
            left{ -1, 0 },
            ur{ 1, -1 },
            dr{ 1, 1 },
            dl{ -1, 1 },
            ul{ -1, -1 };
        static constexpr std::array<glm::ivec2, 8> ordered{
            up,
            ur,
            right,
            dr,
            down,
            dl,
            left,
            ul,
        };
    };

    struct Vertex
    {
        Vertex(glm::vec2 position, glm::u8vec4 color) : position{ position }, color{ color } {};

        glm::vec2 position{};
        glm::u8vec4 color{};
    };

    struct UserData : public neat::UserData
    {
        glube::BuildingBuffer<Vertex> *game_vbo{};
        glube::BuildingBuffer<Vertex> *champ_vbo{};
        int simId{};
        float energy_max{ 500 };

        template<typename Archive>
        void serialize(Archive &ar)
        {
            ar(simId, energy_max);
        }
    };

    struct GameState : public neat::Simulation
    {
        std::array<Tile, Config<std::size_t>::game_tile_count> tilesArray;
        std::mdspan<Tile, std::extents<int, Config<int>::game_size, Config<int>::game_size>> tiles{ tilesArray.data() };

        std::size_t snake_facing{};
        int snake_length{ Config<int>::start_length };
        int auto_grow_remaining{ Config<int>::auto_grow_size };
        glm::ivec2 snake_head{ Config<int>::start_position };
        glm::ivec2 food_position{};
        float lifespan{};
        float energy{};

        thread_local static inline int stats_hit_border{};
        thread_local static inline int stats_hit_snake{};
        thread_local static inline int stats_hit_food{};
        thread_local static inline int stats_energy_zero{};
        thread_local static inline int stats_turn_left{};
        thread_local static inline int stats_turn_right{};
        thread_local static inline int stats_turn_none{};

        static void stats_reset()
        {
            stats_hit_border = {};
            stats_hit_snake = {};
            stats_hit_food = {};
            stats_energy_zero = {};
            stats_turn_left = {};
            stats_turn_right = {};
            stats_turn_none = {};
        }

        GameState()
        {
            reset();
        }

        void init(neat::SimulationInfo &info) override
        {
            energy = dynamic_cast<UserData *>(info.user_data)->energy_max;
        }

        void reset()
        {
            snake_facing = 0;
            snake_length = { Config<int>::start_length };
            snake_head = { Config<int>::start_position };
            food_position = {};
            std::ranges::for_each(tilesArray, &Tile::reset);
            for (std::size_t i{}; i < Config<std::size_t>::game_size; ++i)
            {
                auto ii = static_cast<std::ptrdiff_t>(i);
                tiles[0, ii].type = TileType::border;
                tiles[Config<int>::game_size - 1, i].type = TileType::border;
                tiles[i, 0].type = TileType::border;
                tiles[i, Config<int>::game_size - 1].type = TileType::border;
            }
            for (int i{}; i < Config<int>::start_length; ++i)
            {
                auto prevPos = snake_head - Directions::ordered[snake_facing] * i;
                auto &prev = tiles[prevPos.x, prevPos.y];
                prev.type = TileType::snake;
                prev.value = snake_length - i;
            }
            place_food();
        }

        void place_food()
        {
            while (true)
            {
                auto spot = neat::Random::range(Config<int>::game_tile_count - 1);
                Tile &tile = tilesArray[static_cast<std::size_t>(spot)];
                if (tile.type == TileType::none)
                {
                    tile.type = TileType::food;
                    food_position = { spot % Config<int>::game_size, spot / Config<int>::game_size };
                    return;
                }
            }
        }

        void skip(neat::SimulationInfo &info) override
        {
            dynamic_cast<UserData *>(info.user_data)->simId++;
        }

        template<typename T>
        static bool inrange(const T v, const T a, const T b)
        {
            return v >= a && v < b;
        }

        struct DetectorResult {
            neat::real_t type{};
            neat::real_t distance{};
        };
        DetectorResult detect(const std::size_t dirindex)
        {
            static constexpr auto gsi = Config<int>::game_size;
            const auto direction = Directions::ordered[(snake_facing + dirindex) % 8];
            using rvec2 = glm::vec<2, neat::real_t>;
            for (glm::ivec2 loc = snake_head + direction; inrange(loc.x, 0, gsi) && inrange(loc.y, 0, gsi); loc += direction)
            {
                const auto type = tiles[loc.x, loc.y].type;
                if (type == TileType::none) continue;
                return {

                    // Everything is equally dangerous.
                    // B/S = 0.0, F = 1.0
                    type == TileType::food ? 1.0_r : 0.0_r,

                    // DiffVaA: Tails are half as deadly as walls.
                    // B = 0.5, S = 1.0, F = 1.5
                    // std::to_underlying(type) * 0.5_r,

                    // DiffValB: Tails half as deadly as walls, inverted to SameValA snake val
                    // B  = 0.0, S = 0.5, F = 1.0
                    // type == TileType::food ? 1.0_r : (type == TileType::snake ? 0.5_r : 0.0_r),

                    (glm::distance(rvec2{snake_head}, rvec2{loc}) - 1.0_r) / Config<neat::real_t>::game_size,
                };
            }
            exit(1); // Unreachable.
        }

        static constexpr neat::Config population_config{
            .setup_input_nodes = 8 * 2 + 1 + 2,
            .setup_output_nodes = 3,
            .setup_inital_connection_rate = 0.0_r,
            // .species_disjoint_coefficient = 1.3_r,
            // .species_weight_coefficent = 0.5_r,
            // .species_compatability_modifier = 0.3_r,
            // .species_count_target = 10,
            .mutate_new_connection_rate = 2.0_r,
            .mutate_new_node_rate = 0.5_r,
        };

        void step(neat::SimulationInfo &info) override
        {
            UserData *const userData = dynamic_cast<UserData *>(info.user_data);
            const auto offset = glm::ivec2{ userData->simId % Config<int>::sim_games_displayed_size, userData->simId / Config<int>::sim_games_displayed_size } *Config<int>::game_size;

            info.inputs.push_back(1.0_r); // Bias.
            for (std::size_t i{}; i < 8; ++i)
            {
                const auto result = detect(i);
                info.inputs.push_back(result.type);
                info.inputs.push_back(result.distance);
            }
            info.inputs.push_back(static_cast<float>(snake_facing) / 8.0f);
            info.inputs.push_back(static_cast<float>(snake_length));

            // info.run([](neat::real_t x) { return 1.0_r / (1.0_r + std::exp(-x)); });
            info.run(std::tanh);
            const auto maxelement = std::ranges::max_element(info.outputs);
            const auto index = std::ranges::distance(info.outputs.begin(), maxelement);

#ifdef SNAKE_LOGGY
            loggy << "Food " << glm::to_string(food_position) << " Head " << glm::to_string(snake_head) << " Inputs: [";
            for (auto in : info.inputs) loggy << in << ",";
            loggy << " Outputs: [";
            for (auto out : info.outputs) loggy << out << ",";
            loggy << "]\n";
#endif

            if (*maxelement > 0.5_r)
            {
                switch (index)
                {
                case 0: ++stats_turn_left; snake_facing = (snake_facing - 2) % 8; break;
                case 1: ++stats_turn_none; break;
                case 2: ++stats_turn_right; snake_facing = (snake_facing + 2) % 8; break;
                default: exit(1); // Should be impossible.
                }
            }

            bool hasEnergy = energy > 0.0_r;
            if (!hasEnergy) ++stats_energy_zero;

            if (hasEnergy && step(userData->energy_max))
            {

                for (int y{}; y != tiles.extent(0); ++y)
                {
                    for (int x{}; x != tiles.extent(1); ++x)
                    {
                        auto &tile = tiles[x, y];
                        glm::ivec2 pointPos = glm::ivec2{ x, y } + offset;
                        switch (tile.type)
                        {
                        case TileType::snake: userData->game_vbo->emplace_back(pointPos, glm::u8vec4{ 255, 0, 0, 255 }); break;
                        case TileType::food: userData->game_vbo->emplace_back(pointPos, glm::u8vec4{ 0, 255, 0, 255 }); break;
                        case TileType::border: userData->game_vbo->emplace_back(pointPos, glm::u8vec4{ 0, 0, 255, 255 }); break;
                        default: break;
                        }
                        if (info.genome.is_current_champ())
                        {
                            glm::ivec2 champPos = glm::ivec2{ (x * Config<int>::window_champ_pixel_size) + (Config<int>::sim_games_displayed_size) * Config<int>::game_size, y * Config<int>::window_champ_pixel_size };
                            switch (tile.type)
                            {
                            case TileType::snake: userData->champ_vbo->emplace_back(champPos, glm::u8vec4{ 255, 0, 0, 255 }); break;
                            case TileType::food: userData->champ_vbo->emplace_back(champPos, glm::u8vec4{ 0, 255, 0, 255 }); break;
                            case TileType::border: userData->champ_vbo->emplace_back(champPos, glm::u8vec4{ 0, 0, 255, 255 }); break;
                            default: break;
                            }
                        }
                    }
                }


                lifespan += 1.0f;
                energy -= 1.0f;
                info.fitness = static_cast<neat::real_t>(snake_length - (Config<int>::start_length + (Config<int>::auto_grow_size - auto_grow_remaining))) + 0.001f;// * snake_length) + lifespan * 0.01f;
            }
            else
            {
                info.is_done = true;
            }
            userData->simId++;
        }

        bool step(const float energy_max)
        {
            const auto nextPos = snake_head + Directions::ordered[snake_facing];
            auto &next = tiles[nextPos.x, nextPos.y];
            switch (next.type)
            {
            case TileType::border: ++stats_hit_border; return false;
            case TileType::snake: ++stats_hit_snake; return false;
            case TileType::food:
                ++stats_hit_food;
                next.type = TileType::snake;
                next.value = ++snake_length;
                snake_head = nextPos;
                energy = energy_max;
                place_food();
                return true;
            case TileType::none:
                if (auto_grow_remaining > 0)
                {
                    --auto_grow_remaining;
                    ++snake_length;
                    energy = energy_max;
                }
                else
                {
                    for (auto &tile : tilesArray | mj::filter(&Tile::type, std::equal_to{}, TileType::snake))
                    {
                        --tile.value;
                        if (tile.value <= 0)
                        {
                            tile.type = TileType::none;
                        }
                    }
                }
                next.type = TileType::snake;
                next.value = snake_length;
                snake_head = nextPos;
                return true;
            }
        }
    };


} // End namespace snake.

enum class UserCommand
{
    none,
    save,
    load,
};

int main(int argc, char **argv)
{
    using namespace snake;

    UserData userData;

    neat::Population population{
        std::make_unique<neat::EasySimulationFactory<snake::GameState>>(),
        GameState::population_config,
        &userData
    };
    population.set_stats_string_handler([](const std::string stats) { std::print("{}", stats); });

    bool skipgen{};
    UserCommand userCommand = UserCommand::none;

    glube::Window window(Config<int>::window_width, Config<int>::window_height, "Snake");
    window.swap_interval(1);
    window.set_key_event_handler([&](glube::KeyEvent event) {
        if (event.key == glube::Key::escape) event.window->set_should_close(true);
        if (event.key == glube::Key::tab && event.action == glube::KeyAction::pressed) skipgen = true;
        if (event.key == glube::Key::keypad_add && event.action == glube::KeyAction::pressed)
        {
            std::println("Energy Max: {}", userData.energy_max += 1000);
        }
        if (event.key == glube::Key::keypad_subtract && event.action == glube::KeyAction::pressed)
        {
            std::println("Energy Max: {}", userData.energy_max -= 1000);
        }
        if (event.key == glube::Key::s && event.action == glube::KeyAction::pressed)
        {
            userCommand = UserCommand::save;
        }
        if (event.key == glube::Key::l && event.action == glube::KeyAction::pressed)
        {
            userCommand = UserCommand::load;
        }
        });

    glEnable(GL_PROGRAM_POINT_SIZE);

    glube::BuildingBuffer<Vertex> game_vbo;
    glube::BuildingBuffer<Vertex> champ_vbo;

    glube::Program program{ vertex_shader, fragment_shader };

    const auto projection = glm::ortho(0.0f, Config<float>::window_width, Config<float>::window_height, 0.0f, -1.0f, 1.0f);
    program.set_uniform(nameof(projection), projection);

    glube::Attributes game_vao;
    game_vao.add(program, game_vbo.buffer(), nameof(Vertex::position), &Vertex::position);
    game_vao.add(program, game_vbo.buffer(), nameof(Vertex::color), &Vertex::color);

    glube::Attributes champ_vao;
    champ_vao.add(program, champ_vbo.buffer(), nameof(Vertex::position), &Vertex::position);
    champ_vao.add(program, champ_vbo.buffer(), nameof(Vertex::color), &Vertex::color);

    const auto diagrammer_size = Config<int>::sim_games_displayed_size * Config<int>::sim_games_displayed_size * 2;
    neat::draw::gl::Diagrammer diagrammer{ {diagrammer_size, diagrammer_size} };
    int champId = -1;

    GameState state;

    userData.game_vbo = &game_vbo;
    userData.champ_vbo = &champ_vbo;

    std::ofstream csv_fpg{ "fitness_per_generation.csv", std::ios::app };
    csv_fpg.seekp(0, std::ios::end);
    if (csv_fpg.tellp() < 10) // Using 10 because 0 is unreliable.
    {
        std::println(csv_fpg, "Generation,Fitness Max,Fitness Avg,Collide Border,Collide Snake,Collide Food,Energy Zero,Turn Left,Turn Right,Turn None");
    }

    if (argc == 2 && std::strcmp(argv[1], "load") == 0)
    {
        std::ifstream brainin{ "state.json" };
        if (brainin)
        {
            cereal::JSONInputArchive archive{ brainin };
            archive(population, userData);
            population.reset_champ();
        }
    }

    while (!window.should_close())
    {

        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(0, 0, Config<int>::window_width, Config<int>::window_height);

        population.step();
        userData.simId = 0;

        program.activate();

        program.set_uniform("pointsize", 1.0f);
        game_vao.activate();
        game_vbo.set_and_clear();
        glDrawArrays(GL_POINTS, 0, game_vbo.isize_set());

        program.set_uniform("pointsize", Config<float>::window_champ_pixel_size);
        champ_vao.activate();
        champ_vbo.set_and_clear();
        glDrawArrays(GL_POINTS, 0, champ_vbo.isize_set());

        if (population.champ_id() != champId)
        {
            champId = population.champ_id();
            diagrammer.build_diagram(population.champ());
        }
        glViewport(Config<int>::sim_games_displayed_size * Config<int>::game_size, Config<int>::sim_games_displayed_size, diagrammer_size, diagrammer_size);
        diagrammer.draw_diagram();

        if (population.generation_is_done() || skipgen)
        {
            const auto maxFitness = std::ranges::max_element(population.genomes(), {}, &neat::Genome::fitness);
            const auto avgFitness = std::ranges::fold_left(population.genomes() | std::views::transform(&neat::Genome::fitness), 0.0_r, std::plus{}) / static_cast<neat::real_t>(population.population_count());
            // "Generation,Fitness Max,Fitness Avg,Collide Border,Collide Snake,Turn Left,Turn Right,Turn None"
            std::println(csv_fpg, "{},{},{},{},{},{},{},{},{},{}"
                , population.generation()
                , maxFitness->fitness()
                , avgFitness
                , GameState::stats_hit_border
                , GameState::stats_hit_snake
                , GameState::stats_hit_food
                , GameState::stats_energy_zero
                , GameState::stats_turn_left
                , GameState::stats_turn_right
                , GameState::stats_turn_none
            );
            GameState::stats_reset();
            skipgen = false;
            if (userCommand == UserCommand::save)
            {
                std::ofstream brainout{ "state.json" };
                cereal::JSONOutputArchive archive{ brainout };
                archive(population, userData);
            }
            if (userCommand == UserCommand::load)
            {
                std::ifstream brainin{ "state.json" };
                cereal::JSONInputArchive archive{ brainin };
                archive(population, userData);
                population.reset_champ();
            }
            userCommand = UserCommand::none;
            population.new_generation();
        }

        csv_fpg.flush();


        window.swap_buffers();
        window.poll_events();
        // if (population.generation() == 100) window.set_should_close(true);
    }

}