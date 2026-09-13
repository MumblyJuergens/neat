option(NEAT_DOUBLE_PRECISION "Use double instead of float" OFF)

add_library(neat)
target_sources(neat
    PRIVATE
    "src/Brain.cpp"
    "src/Genome.cpp"
    "src/InnovationHistory.cpp"
    "src/neat.cpp"
    "src/Simulation.cpp"
    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
    "include/neat/Brain.hpp"
    "include/neat/Config.hpp"
    "include/neat/Genome.hpp"
    "include/neat/InnovationHistory.hpp"
    "include/neat/neat.hpp"
    "include/neat/Neuron.hpp"
    "include/neat/Population.hpp"
    "include/neat/Random.hpp"
    "include/neat/Simulation.hpp"
    "include/neat/SimulationFactory.hpp"
    "include/neat/SimulationInfo.hpp"
    "include/neat/Species.hpp"
    "include/neat/Synapse.hpp"
    "include/neat/types.hpp"
    "include/neat/UserData.hpp"
)
add_library(neat::neat ALIAS neat)
target_compile_features(neat PUBLIC cxx_std_26)
target_link_libraries(neat PUBLIC mj::mj cereal::cereal)
FullCompilerWarnings(neat PUBLIC)

configure_file("configure.hpp.in" "neat/configure.hpp")
target_include_directories(neat PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
