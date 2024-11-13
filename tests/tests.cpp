#include <print>
#include <catch2/catch_test_macros.hpp>
#include <neat/Random.hpp>

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