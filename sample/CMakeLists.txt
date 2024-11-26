CPMAddPackage("gh:MumblyJuergens/cppfront#main")

add_executable(neat_sample_xor "xor/xor.cpp2")
target_compile_features(neat_sample_xor PUBLIC cxx_std_23)
target_link_libraries(neat_sample_xor PUBLIC neat::neat neat::neatdrawgl glube)
MJFullCompilerWarnings(neat_sample_xor)
# if (CMAKE_BUILD_TYPE STREQUAL "Release")
#     MJSanitize(neat_sample_xor ON OFF ON OFF OFF)
# endif()

add_executable(neat_sample_flappy
"flappy/Bird.hpp"
"flappy/Config.hpp"
"flappy/Flappy.hpp"
"flappy/logic.hpp"
"flappy/main.cpp"
"flappy/Pipe.hpp"
"flappy/Sim.hpp"
"flappy/UserData.hpp"
"flappy/Vertex.hpp"
)
target_compile_features(neat_sample_flappy PUBLIC cxx_std_23)
target_link_libraries(neat_sample_flappy PUBLIC neat::neat neat::neatdrawgl glube)
MJFullCompilerWarnings(neat_sample_flappy)
# if (CMAKE_BUILD_TYPE STREQUAL "Release")
#     MJSanitize(neat_sample_flappy ON OFF ON OFF OFF)
# endif()

add_executable(neat_sample_snake "snake/main.cpp")
target_compile_features(neat_sample_snake PUBLIC cxx_std_23)
target_link_libraries(neat_sample_snake PUBLIC neat::neat neat::neatdrawgl glube)
MJFullCompilerWarnings(neat_sample_snake)
# if (CMAKE_BUILD_TYPE STREQUAL "Release")
#     MJSanitize(neat_sample_snake ON OFF ON OFF OFF)
# endif()