# Based loosely on the Godot example file.

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${godot-cpp_SOURCE_DIR}/cmake")
include(emsdkHack)
include(GodotCPPModule)

# Create the gen directory at configure time so IDEs can see generated headers immediately after generate_bindings
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/godot-cpp/gen/include")
set(GODOT_PROJECT_DIR "addons/gdneat" CACHE STRING "The directory of a Godot project folder")


get_target_property(GODOTCPP_SUFFIX godot::cpp GODOTCPP_SUFFIX)
get_target_property(GODOTCPP_PLATFORM godot::cpp GODOTCPP_PLATFORM)

project(gdneat
    VERSION 1.0
    DESCRIPTION "Hopefully enabling the use of NEAT with Godot"
    HOMEPAGE_URL "https://github.com/MumblyJuergens/neat"
    LANGUAGES CXX
)

add_library(gdneat SHARED)

target_sources(gdneat
    PRIVATE
    src/GDNeatGenome.hpp
    src/GDNeatGenome.cpp
    src/register_types.cpp
    src/register_types.hpp
    src/GDNeatConfig.hpp
    src/GDNeatConfig.cpp
    src/GDNeatPopulation.hpp
    src/GDNeatPopulation.cpp
)

# # Fetch a list of the xml files to use for documentation and add to our target
# file(GLOB_RECURSE DOC_XML LIST_DIRECTORIES NO CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/doc_classes/*.xml")

# # conditionally add doc data to compile output
# if(DOC_XML)
#     if(GODOTCPP_TARGET MATCHES "editor|template_debug")
#         target_doc_sources(${LIBNAME} "${DOC_XML}")
#     endif()
# endif()

target_link_libraries(gdneat PRIVATE godot-cpp neat::neat)
target_compile_features(gdneat PRIVATE cxx_std_26)

# TODO: Only if building GDNeat?
target_compile_options(neat PUBLIC -fPIC)

set_target_properties(gdneat
    PROPERTIES
    # The generator expression here prevents msvc from adding a Debug or Release subdir.
    RUNTIME_OUTPUT_DIRECTORY "$<1:${PROJECT_SOURCE_DIR}/bin/${GODOTCPP_PLATFORM}>"

    PREFIX ""
    OUTPUT_NAME "gdneat${GODOTCPP_SUFFIX}"
)

set(GODOT_PROJECT_BINARY_DIR "${PROJECT_SOURCE_DIR}/${GODOT_PROJECT_DIR}/bin/${GODOTCPP_PLATFORM}")

add_custom_command(TARGET gdneat POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:gdneat>" "${GODOT_PROJECT_BINARY_DIR}/$<TARGET_FILE_NAME:gdneat>"
)
