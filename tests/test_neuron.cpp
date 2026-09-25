#include "neat/Neuron.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cereal/archives/binary.hpp>
#include <sstream>

TEST_CASE("Neuron can serialize", "[serialize]") {
    neat::Neuron source{42, neat::NeuronType::input};
    neat::Neuron target{};
    std::stringstream storage;
    
    {
        cereal::BinaryOutputArchive archive{storage};
        archive(source);
    }

    {
        cereal::BinaryInputArchive archive{storage};
        archive(target);
    }

    REQUIRE(source.number() == target.number());
    REQUIRE(source.type() == target.type());
    REQUIRE(source.value() == target.value());
    REQUIRE(source.layer() == target.layer());
}