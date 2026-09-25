#include "neat/Synapse.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cereal/archives/binary.hpp>
#include <sstream>

TEST_CASE("Synapse can serialize", "[serialize]") {

    using namespace neat::literals;

    neat::Synapse source{42, 69, 0.123_r, 7};
    neat::Synapse target{};
    std::stringstream storage;
    
    {
        cereal::BinaryOutputArchive archive{storage};
        archive(source);
    }

    {
        cereal::BinaryInputArchive archive{storage};
        archive(target);
    }

    REQUIRE(source.in() == target.in());
    REQUIRE(source.out() == target.out());
    REQUIRE(source.weight() == target.weight());
    REQUIRE(source.enabled() == target.enabled());
    REQUIRE(source.innovation() == target.innovation());
}