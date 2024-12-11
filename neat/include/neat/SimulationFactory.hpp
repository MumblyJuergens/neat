#pragma once

#include <memory>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>

// Needed to make inheriting junk work.
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>

namespace neat
{
    class Simulation;

    /**
     * @brief Base class used to feed fresh simulation child objects
     * to Population as requested for genomes.
     */
    class [[nodiscard]] SimulationFactory
    {
        public:

        virtual ~SimulationFactory() = default;
        virtual std::shared_ptr<Simulation> create_simulation() = 0;

        template<typename Archive>
        void serialize(Archive &) {}
    };

    /**
     * @brief Creates creates multiple shard pointers of a type
     * for consumption by Population.
     */
    template<typename T>
    class [[nodiscard]] EasySimulationFactory : public SimulationFactory
    {
        public:

        std::shared_ptr<Simulation> create_simulation() override { return std::make_shared<T>(); };

        template<typename Archive>
        void serialize(Archive &ar)
        {
            ar(cereal::base_class<SimulationFactory>(this));
        }
    };

    /**
 * @brief Repeatedly returns shard pointers of a single object
 * for consumption by Population.
 */
    template<typename T>
    class [[nodiscard]] SingletonSimulationFactory : public SimulationFactory
    {
        std::shared_ptr<T> sim;

        public:

        template<typename ...Args>
        SingletonSimulationFactory(Args &&...args) : sim{ std::make_shared<T>(args...) } {}

        std::shared_ptr<Simulation> create_simulation() override { return sim; };

        template<typename Archive>
        void serialize(Archive &ar)
        {
            ar(cereal::base_class<SimulationFactory>(this), sim);
        }
    };

} // End namespace neat.