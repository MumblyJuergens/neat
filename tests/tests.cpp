#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include <neat/Random.hpp>
#include <neat/Brain.hpp>

TEST_CASE("Random skewed canonical suitable for index", "[random]")
{
    for (int i = 0; i < 1000; ++i)
    {
        const auto index = neat::Random::canonical_skewed_low(6.0f) * 4;
        REQUIRE(index < 5);
    }
}

TEST_CASE("Random range suitable for index", "[random]")
{
    int size{ 5 };
    for (int i = 0; i < 1000; ++i)
    {
        const auto index = neat::Random::range(size - 1);
        REQUIRE(index < 5);
    }
}

[[nodiscard]] constexpr auto process(float(*f)(float), const float, const float i1, const float i2)
{
    const float h4 = f(i1 * 1.752715f);
    const float h5 = f(i1 * 1.2508526f + i2 * 0.9772244f);
    const float o3 = f(i2 * -0.6679219f + h4 * -0.97733116f + h5 * 2.1024675f);
    return o3;
}

TEST_CASE("Sample xor solution test", "[xor]")
{
    const float result100 = process(std::tanh, 1, 0, 0);
    const float result101 = process(std::tanh, 1, 0, 1);
    const float result110 = process(std::tanh, 1, 1, 0);
    const float result111 = process(std::tanh, 1, 1, 1);

    REQUIRE(result100 <= 0.5f);
    REQUIRE(result101 > 0.5f);
    REQUIRE(result110 > 0.5f);
    REQUIRE(result111 <= 0.5f);
}

TEST_CASE("rebuild brain layers", "[brain, layers]")
{
    neat::Brain brain;
    brain.init({ .setup_input_nodes = 3, .setup_output_nodes = 1, .setup_connect_bias = false, .setup_inital_connection_rate = 0.0f }, neat::Init::yes);
    brain.add_connection(0, 3); // Bias to output.
    brain.add_node(); // Uses only connection available.

    brain.rebuild_layers();

    REQUIRE(brain.neurons().size() == 5);
    REQUIRE(brain.neurons().at(0).layer() == 0);
    REQUIRE(brain.neurons().at(1).layer() == 0);
    REQUIRE(brain.neurons().at(2).layer() == 0);
    REQUIRE(brain.neurons().at(3).layer() == 2);
    REQUIRE(brain.neurons().at(4).layer() == 1);
}