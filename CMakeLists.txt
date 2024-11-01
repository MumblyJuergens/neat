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
)

set(neat_headers
    "include/neat/neat.hpp"
)

add_library(neat ${neat_src} ${neat_headers})
add_library(neat::neat ALIAS neat)
target_compile_features(neat PUBLIC cxx_std_23)
target_link_libraries(neat PUBLIC mj::mj)
MJFullCompilerWarnings(neat)
MJLibrarySetup(neat INCLUDE_DIR "include" EXPORT_HEADER_PREFIX "neat/neat" HEADERS ${neat_headers})

if (${NEAT_BUILD_TESTS} OR PROJECT_IS_TOP_LEVEL)
    add_subdirectory("tests")
endif()

if (${NEAT_BUILD_SAMPLE} OR PROJECT_IS_TOP_LEVEL)
    add_subdirectory("sample")
endif()