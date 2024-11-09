cmake_minimum_required(VERSION 3.29)

project(neat
        VERSION 0.0.1
        DESCRIPTION "NEAT (NeuroEvolution of Augmenting Topologies) in C++"
        HOMEPAGE_URL "https://github.com/MumblyJuergens/neat"
        LANGUAGES CXX)

option(NEAT_BUILD_TESTS "Build tests for neat")
option(NEAT_BUILD_SAMPLE "Build sample for neat")

include(cmake/CPM.cmake)
CPMAddPackage("gh:MumblyJuergens/mj#main")

set(neat_src
    "src/neat.cpp"
    "src/InnovationHistory.cpp"
)

set(neat_headers
    "include/neat/Config.hpp"
    "include/neat/flag_types.hpp"
    "include/neat/Genome.hpp"
    "include/neat/innovation.hpp"
    "include/neat/InnovationHistory.hpp"
    "include/neat/log.hpp"
    "include/neat/neat.hpp"
    "include/neat/Neuron.hpp"
    "include/neat/Population.hpp"
    "include/neat/Random.hpp"
    "include/neat/Simulation.hpp"
    "include/neat/Species.hpp"
    "include/neat/Synapse.hpp"
)

add_library(neat ${neat_src} ${neat_headers})
add_library(neat::neat ALIAS neat)
target_compile_features(neat PUBLIC cxx_std_23)
target_link_libraries(neat PUBLIC mj::mj glm::glm)
MJFullCompilerWarnings(neat)
MJLibrarySetup(neat INCLUDE_DIR "include" EXPORT_HEADER_PREFIX "neat/neat" HEADERS ${neat_headers})
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    MJSanitize(neat ON OFF ON OFF OFF)
endif()

if (${NEAT_BUILD_TESTS} OR PROJECT_IS_TOP_LEVEL)
    add_subdirectory("tests")
endif()

if (${NEAT_BUILD_SAMPLE} OR PROJECT_IS_TOP_LEVEL)
    add_subdirectory("sample")
endif()