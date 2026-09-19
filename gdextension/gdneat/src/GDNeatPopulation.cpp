#include "GDNeatPopulation.hpp"
#include "GDNeatConfig.hpp"
#include "GDNeatGenome.hpp"
#include "neat/Genome.hpp"
#include "neat/SimplePopulation.hpp"
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/print_string.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <memory>
#include <vector>

namespace gdneat
{

void GDNeatPopulation::create(godot::Ref<GDNeatConfig> cfg)
{
    // Need to keep a reference, it's just how Population works atm.
    config = cfg;
    population = std::make_unique<neat::SimplePopulation>(config->config);
}

void GDNeatPopulation::_bind_methods()
{
    // clang-format off
    godot::ClassDB::bind_method(godot::D_METHOD("create", "config"), &GDNeatPopulation::create);
    godot::ClassDB::bind_method(godot::D_METHOD("step", "handler"), &GDNeatPopulation::step);
    godot::ClassDB::bind_method(godot::D_METHOD("_generation_is_done"), &GDNeatPopulation::_generation_is_done);
    godot::ClassDB::bind_method(godot::D_METHOD("new_generation"), &GDNeatPopulation::new_generation);
    godot::ClassDB::bind_method(godot::D_METHOD("_generation"), &GDNeatPopulation::_generation);

    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "generation_is_done", godot::PROPERTY_HINT_NONE, "", godot::PROPERTY_USAGE_READ_ONLY), "", "_generation_is_done");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "generation", godot::PROPERTY_HINT_NONE, "", godot::PROPERTY_USAGE_READ_ONLY), "", "_generation");
    // clang-format on
}

void GDNeatPopulation::step(godot::Callable handler)
{
    population->step([&handler](std::vector<neat::Genome> &genomes) {
        auto gnomes = godot::TypedArray<godot::Ref<GDNeatGenome>>();
        for (auto &genome : genomes) {
            godot::Ref<GDNeatGenome> gnome;
            gnome.instantiate();
            gnome->_genome = &genome;
            gnomes.append(gnome);
        }
        handler.call(gnomes);
    });
}

bool GDNeatPopulation::_generation_is_done() const { return population->generation_is_done(); }

void GDNeatPopulation::new_generation() { population->new_generation(); }

int GDNeatPopulation::_generation() const { return population->generation(); }
} // namespace gdneat