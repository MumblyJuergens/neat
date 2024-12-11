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

set(SKIP_PERFORMANCE_COMPARISON ON)
CPMAddPackage("gh:USCiLab/cereal#master")

add_subdirectory("neat")
add_subdirectory("neatdraw")

if (${NEAT_BUILD_TESTS} OR PROJECT_IS_TOP_LEVEL)
    add_subdirectory("tests")
endif()

if (${NEAT_BUILD_SAMPLE} OR PROJECT_IS_TOP_LEVEL)
    add_subdirectory("sample")
endif()

if (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease" AND PROJECT_IS_TOP_LEVEL)
    # first we can indicate the documentation build as an option and set it to ON by default
    option(BUILD_DOC "Build documentation" ON)

    # check if Doxygen is installed
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        # set input and output files
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)

        # request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message("Doxygen build started")

        # note the option ALL which allows to build the docs together with the application
        add_custom_target( doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM )
    else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
    endif (DOXYGEN_FOUND)
endif()
