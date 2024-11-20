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

add_subdirectory("neat")
add_subdirectory("neatdraw")

if (${NEAT_BUILD_TESTS} OR PROJECT_IS_TOP_LEVEL)
    add_subdirectory("tests")
endif()

if (${NEAT_BUILD_SAMPLE} OR PROJECT_IS_TOP_LEVEL)
    add_subdirectory("sample")
endif()