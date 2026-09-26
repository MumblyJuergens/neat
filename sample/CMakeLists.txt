# CPMAddPackage("gh:MumblyJuergens/cppfront#main")
# if(MSVC)
# target_compile_options(cppfront_cppfront PUBLIC /bigobj)
# endif()

add_executable(neat_sample_xor "xor/xor.cpp")
target_compile_features(neat_sample_xor PRIVATE cxx_std_26)
target_link_libraries(neat_sample_xor PRIVATE neat::neat)
FullCompilerWarnings(neat_sample_xor PRIVATE)

# if (CMAKE_BUILD_TYPE STREQUAL "Release")
#     MJSanitize(neat_sample_xor ON OFF ON OFF OFF)
# endif()

# add_executable(neat_sample_flappy
#     "flappy/Bird.hpp"
#     "flappy/Config.hpp"
#     "flappy/Flappy.hpp"
#     "flappy/logic.hpp"
#     "flappy/main.cpp"
#     "flappy/Pipe.hpp"
#     "flappy/Sim.hpp"
#     "flappy/UserData.hpp"
#     "flappy/Vertex.hpp"
# )
# target_compile_features(neat_sample_flappy PUBLIC cxx_std_26)
# target_link_libraries(neat_sample_flappy PUBLIC neat::neat neat::neatdrawgl glube)
# MJFullCompilerWarnings(neat_sample_flappy)
# # if (CMAKE_BUILD_TYPE STREQUAL "Release")
# #     MJSanitize(neat_sample_flappy ON OFF ON OFF OFF)
# # endif()

# add_executable(neat_sample_snake "snake/main.cpp")
# target_compile_features(neat_sample_snake PUBLIC cxx_std_23)
# target_compile_options(neat_sample_snake PUBLIC -ftemplate-backtrace-limit=0)
# target_link_libraries(neat_sample_snake PUBLIC neat::neat neat::neatdrawgl glube)
# MJFullCompilerWarnings(neat_sample_snake)

CPMAddPackage("gh:MumblyJuergens/mjsdl#main")
add_executable(neat_sample_snakesdl "snakesdl/main.cpp"
    snakesdl/sdlmath.hpp
    snakesdl/snake.hpp
    snakesdl/config.hpp
    snakesdl/game.hpp
    snakesdl/food.hpp
    snakesdl/raycast.hpp
    snakesdl/raycast.cpp
    snakesdl/directions.hpp
    snakesdl/enums.hpp
    snakesdl/random.cpp
    snakesdl/random.hpp
)
target_compile_features(neat_sample_snakesdl PUBLIC cxx_std_26)
target_compile_options(neat_sample_snakesdl PUBLIC -freflection)
target_link_libraries(neat_sample_snakesdl PUBLIC neat::neat mjsdl::mjsdl)
FullCompilerWarnings(neat_sample_snakesdl PUBLIC)

# add_executable(neat_sample_roids "roids/main.cpp" "roids/ezobj.hpp" "roids/ezobj.cpp" "roids/shaders.hpp")
# target_compile_features(neat_sample_roids PUBLIC cxx_std_23)
# target_link_libraries(neat_sample_roids PUBLIC neat::neat neat::neatdrawgl glube)
# MJFullCompilerWarnings(neat_sample_roids)
