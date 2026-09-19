#pragma once

#include "neat/Config.hpp"
#include "neat/types.hpp"
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/math_defs.hpp>

namespace gdneat
{

class GDNeatConfig : public godot::Resource
{
    GDCLASS(GDNeatConfig, godot::Resource)

  private:
  protected:
    static void _bind_methods();

  public:
    // May as well be public, access from GDNeatPopulation required.
    neat::Config config;

    // clang-format off

    // Setup
    int get_setup_population_size() const { return config.setup_population_size; }
    int get_setup_input_nodes() const { return config.setup_input_nodes; }
    int get_setup_output_nodes() const { return config.setup_output_nodes; }
    bool get_setup_connect_bias() const { return config.setup_connect_bias; }
    neat::index_t get_setup_bias_input() const { return config.setup_bias_input; }
    double get_setup_inital_connection_rate() const { return static_cast<double>(config.setup_inital_connection_rate); }
    void set_setup_population_size(int value) { config.setup_population_size = value; }
    void set_setup_input_nodes(int value) { config.setup_input_nodes = value; }
    void set_setup_output_nodes(int value) { config.setup_output_nodes = value; }
    void set_setup_connect_bias(bool value) { config.setup_connect_bias = value; }
    void set_setup_bias_input(neat::index_t value) { config.setup_bias_input = value; }
    void set_setup_inital_connection_rate(double value) { config.setup_inital_connection_rate = static_cast<neat::real_t>(value); }

    // Speciation.
    int get_species_maximum_staleness() const { return config.species_maximum_staleness; }
    double get_species_compatibility_threshold() const { return static_cast<double>(config.species_compatibility_threshold); }
    double get_species_disjoint_coefficient() const { return static_cast<double>(config.species_disjoint_coefficient); }
    double get_species_weight_coefficient() const { return static_cast<double>(config.species_weight_coefficient); }
    double get_species_compatibility_modifier() const { return static_cast<double>(config.species_compatibility_modifier); }
    int get_species_count_target() const { return config.species_count_target; }
    void set_species_maximum_staleness(int value) { config.species_maximum_staleness = value; }
    void set_species_compatibility_threshold(double value) { config.species_compatibility_threshold = static_cast<neat::real_t>(value); }
    void set_species_disjoint_coefficient(double value) { config.species_disjoint_coefficient = static_cast<neat::real_t>(value); }
    void set_species_weight_coefficient(double value) { config.species_weight_coefficient = static_cast<neat::real_t>(value); }
    void set_species_compatibility_modifier(double value) { config.species_compatibility_modifier = static_cast<neat::real_t>(value); }
    void set_species_count_target(int value) { config.species_count_target = value; }

    // Crossover.
    int get_crossover_elite_size() const { return config.crossover_elite_size; }
    bool get_crossover_use_adjusted_fitness() const { return config.crossover_use_adjusted_fitness; }
    void set_crossover_elite_size(int value) { config.crossover_elite_size = value; }
    void set_crossover_use_adjusted_fitness(bool value) { config.crossover_use_adjusted_fitness = value; }

    // Mutation.
    double get_mutate_weight_rate() const { return static_cast<double>(config.mutate_weight_rate); }
    double get_mutate_weight_amount() const { return static_cast<double>(config.mutate_weight_amount); }
    double get_mutate_weight_min() const { return static_cast<double>(config.mutate_weight_min); }
    double get_mutate_weight_max() const { return static_cast<double>(config.mutate_weight_max); }
    double get_mutate_redraw_weight() const { return static_cast<double>(config.mutate_redraw_weight); }
    double get_mutate_new_connection_rate() const { return static_cast<double>(config.mutate_new_connection_rate); }
    double get_mutate_new_node_rate() const { return static_cast<double>(config.mutate_new_node_rate); }
    double get_mutate_disable_node_rate() const { return static_cast<double>(config.mutate_disable_node_rate); }
    void set_mutate_weight_rate(double value) { config.mutate_weight_rate = static_cast<neat::real_t>(value); }
    void set_mutate_weight_amount(double value) { config.mutate_weight_amount = static_cast<neat::real_t>(value); }
    void set_mutate_weight_min(double value) { config.mutate_weight_min = static_cast<neat::real_t>(value); }
    void set_mutate_weight_max(double value) { config.mutate_weight_max = static_cast<neat::real_t>(value); }
    void set_mutate_redraw_weight(double value) { config.mutate_redraw_weight = static_cast<neat::real_t>(value); }
    void set_mutate_new_connection_rate(double value) { config.mutate_new_connection_rate = static_cast<neat::real_t>(value); }
    void set_mutate_new_node_rate(double value) { config.mutate_new_node_rate = static_cast<neat::real_t>(value); }
    void set_mutate_disable_node_rate(double value) { config.mutate_disable_node_rate = static_cast<neat::real_t>(value); }

    // clang-format on
};

} // namespace gdneat