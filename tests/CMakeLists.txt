CPMAddPackage("gh:catchorg/Catch2@3.7.1")
add_executable(neat_tests
    "tests.cpp"
)
target_compile_features(neat_tests PUBLIC cxx_std_23)
target_link_libraries(neat_tests PUBLIC neat Catch2::Catch2WithMain)
MJFullCompilerWarnings(neat_tests)
enable_testing()
add_test(neat_tests neat_tests)