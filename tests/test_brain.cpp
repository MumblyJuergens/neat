#include "neat/Brain.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cereal/archives/binary.hpp>
#include <sstream>

TEST_CASE("Brain can serialize", "[serialize]") {

    using namespace neat::literals;

    // Init with layer count of two, empty synapses and neurons, no randomness.
    neat::Brain source{};
    source.init(neat::Config{}, neat::Init::no);
    neat::Brain target{};
    std::stringstream storage;
    
    {
        cereal::BinaryOutputArchive archive{storage};
        archive(source);
    }

    {
        cereal::BinaryInputArchive archive{storage};
        archive(target);
    }

    REQUIRE(source.synapse_count() == target.synapse_count());
    REQUIRE(source.synapse_count() == 0);
    REQUIRE(source.neuron_count() == target.neuron_count());
    REQUIRE(source.neuron_count() == 0);
    REQUIRE(source.layer_count() == target.layer_count());
    REQUIRE(source.layer_count() == 2);
}