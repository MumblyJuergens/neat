CPMAddPackage("gh:MumblyJuergens/glube#main")

add_executable(neat_sample "main.cpp")
target_compile_features(neat_sample PUBLIC cxx_std_23)
target_link_libraries(neat_sample PUBLIC neat::neat glube)
MJFullCompilerWarnings(neat_sample)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    MJSanitize(neat_sample ON OFF ON OFF OFF)
endif()