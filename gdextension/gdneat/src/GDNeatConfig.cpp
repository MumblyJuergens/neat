#include "GDNeatConfig.hpp"
#include <gdextension_interface.h>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/core/property_info.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace gdneat
{
void GDNeatConfig::_bind_methods()
{
    // clang-format off
    godot::ClassDB::bind_method(godot::D_METHOD("get_setup_population_size"), &GDNeatConfig::get_setup_population_size);
    godot::ClassDB::bind_method(godot::D_METHOD("get_setup_input_nodes"), &GDNeatConfig::get_setup_input_nodes);
    godot::ClassDB::bind_method(godot::D_METHOD("get_setup_output_nodes"), &GDNeatConfig::get_setup_output_nodes);
    godot::ClassDB::bind_method(godot::D_METHOD("get_setup_connect_bias"), &GDNeatConfig::get_setup_connect_bias);
    godot::ClassDB::bind_method(godot::D_METHOD("get_setup_bias_input"), &GDNeatConfig::get_setup_bias_input);    
    godot::ClassDB::bind_method(godot::D_METHOD("get_setup_inital_connection_rate"), &GDNeatConfig::get_setup_inital_connection_rate);
    godot::ClassDB::bind_method(godot::D_METHOD("get_species_maximum_staleness"), &GDNeatConfig::get_species_maximum_staleness);
    godot::ClassDB::bind_method(godot::D_METHOD("get_species_compatibility_threshold"), &GDNeatConfig::get_species_compatibility_threshold);
    godot::ClassDB::bind_method(godot::D_METHOD("get_species_disjoint_coefficient"), &GDNeatConfig::get_species_disjoint_coefficient);
    godot::ClassDB::bind_method(godot::D_METHOD("get_species_weight_coefficient"), &GDNeatConfig::get_species_weight_coefficient);
    godot::ClassDB::bind_method(godot::D_METHOD("get_species_compatibility_modifier"), &GDNeatConfig::get_species_compatibility_modifier);
    godot::ClassDB::bind_method(godot::D_METHOD("get_species_count_target"), &GDNeatConfig::get_species_count_target);
    godot::ClassDB::bind_method(godot::D_METHOD("get_crossover_elite_size"), &GDNeatConfig::get_crossover_elite_size);
    godot::ClassDB::bind_method(godot::D_METHOD("get_crossover_use_adjusted_fitness"), &GDNeatConfig::get_crossover_use_adjusted_fitness);
    godot::ClassDB::bind_method(godot::D_METHOD("get_mutate_weight_rate"), &GDNeatConfig::get_mutate_weight_rate);
    godot::ClassDB::bind_method(godot::D_METHOD("get_mutate_weight_amount"), &GDNeatConfig::get_mutate_weight_amount);
    godot::ClassDB::bind_method(godot::D_METHOD("get_mutate_weight_min"), &GDNeatConfig::get_mutate_weight_min);
    godot::ClassDB::bind_method(godot::D_METHOD("get_mutate_weight_max"), &GDNeatConfig::get_mutate_weight_max);
    godot::ClassDB::bind_method(godot::D_METHOD("get_mutate_redraw_weight"), &GDNeatConfig::get_mutate_redraw_weight);
    godot::ClassDB::bind_method(godot::D_METHOD("get_mutate_new_connection_rate"), &GDNeatConfig::get_mutate_new_connection_rate);
    godot::ClassDB::bind_method(godot::D_METHOD("get_mutate_new_node_rate"), &GDNeatConfig::get_mutate_new_node_rate);
    godot::ClassDB::bind_method(godot::D_METHOD("get_mutate_disable_node_rate"), &GDNeatConfig::get_mutate_disable_node_rate);

    godot::ClassDB::bind_method(godot::D_METHOD("set_setup_population_size", "value"), &GDNeatConfig::set_setup_population_size);
    godot::ClassDB::bind_method(godot::D_METHOD("set_setup_input_nodes", "value"), &GDNeatConfig::set_setup_input_nodes);
    godot::ClassDB::bind_method(godot::D_METHOD("set_setup_output_nodes", "value"), &GDNeatConfig::set_setup_output_nodes);
    godot::ClassDB::bind_method(godot::D_METHOD("set_setup_connect_bias", "value"), &GDNeatConfig::set_setup_connect_bias);
    godot::ClassDB::bind_method(godot::D_METHOD("set_setup_bias_input", "value"), &GDNeatConfig::set_setup_bias_input);
    godot::ClassDB::bind_method(godot::D_METHOD("set_setup_inital_connection_rate", "value"), &GDNeatConfig::set_setup_inital_connection_rate);
    godot::ClassDB::bind_method(godot::D_METHOD("set_species_maximum_staleness", "value"), &GDNeatConfig::set_species_maximum_staleness);
    godot::ClassDB::bind_method(godot::D_METHOD("set_species_compatibility_threshold", "value"), &GDNeatConfig::set_species_compatibility_threshold);
    godot::ClassDB::bind_method(godot::D_METHOD("set_species_disjoint_coefficient", "value"), &GDNeatConfig::set_species_disjoint_coefficient);
    godot::ClassDB::bind_method(godot::D_METHOD("set_species_weight_coefficient", "value"), &GDNeatConfig::set_species_weight_coefficient);
    godot::ClassDB::bind_method(godot::D_METHOD("set_species_compatibility_modifier", "value"), &GDNeatConfig::set_species_compatibility_modifier);
    godot::ClassDB::bind_method(godot::D_METHOD("set_species_count_target", "value"), &GDNeatConfig::set_species_count_target);
    godot::ClassDB::bind_method(godot::D_METHOD("set_crossover_elite_size", "value"), &GDNeatConfig::set_crossover_elite_size);
    godot::ClassDB::bind_method(godot::D_METHOD("set_crossover_use_adjusted_fitness", "value"), &GDNeatConfig::set_crossover_use_adjusted_fitness);
    godot::ClassDB::bind_method(godot::D_METHOD("set_mutate_weight_rate", "value"), &GDNeatConfig::set_mutate_weight_rate);
    godot::ClassDB::bind_method(godot::D_METHOD("set_mutate_weight_amount", "value"), &GDNeatConfig::set_mutate_weight_amount);
    godot::ClassDB::bind_method(godot::D_METHOD("set_mutate_weight_min", "value"), &GDNeatConfig::set_mutate_weight_min);
    godot::ClassDB::bind_method(godot::D_METHOD("set_mutate_weight_max", "value"), &GDNeatConfig::set_mutate_weight_max);
    godot::ClassDB::bind_method(godot::D_METHOD("set_mutate_redraw_weight", "value"), &GDNeatConfig::set_mutate_redraw_weight);
    godot::ClassDB::bind_method(godot::D_METHOD("set_mutate_new_connection_rate", "value"), &GDNeatConfig::set_mutate_new_connection_rate);
    godot::ClassDB::bind_method(godot::D_METHOD("set_mutate_new_node_rate", "value"), &GDNeatConfig::set_mutate_new_node_rate);
    godot::ClassDB::bind_method(godot::D_METHOD("set_mutate_disable_node_rate", "value"), &GDNeatConfig::set_mutate_disable_node_rate);

    ADD_GROUP("Setup", "");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "population_size"), "set_setup_population_size", "get_setup_population_size");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "input_nodes"), "set_setup_input_nodes", "get_setup_input_nodes");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "output_nodes"), "set_setup_output_nodes", "get_setup_output_nodes");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "connect_bias"), "set_setup_connect_bias", "get_setup_connect_bias");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "bias_input"), "set_setup_bias_input", "get_setup_bias_input");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "inital_connection_rate"), "set_setup_inital_connection_rate", "get_setup_inital_connection_rate");
    
    ADD_GROUP("Speciation", "");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "maximum_staleness"), "set_species_maximum_staleness", "get_species_maximum_staleness");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "compatibility_threshold"), "set_species_compatibility_threshold", "get_species_compatibility_threshold");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "disjoint_coefficient"), "set_species_disjoint_coefficient", "get_species_disjoint_coefficient");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "weight_coefficient"), "set_species_weight_coefficient", "get_species_weight_coefficient");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "compatibility_modifier"), "set_species_compatibility_modifier", "get_species_compatibility_modifier");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "count_target"), "set_species_count_target", "get_species_count_target");

    ADD_GROUP("Crossover", "");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "elite_size"), "set_crossover_elite_size", "get_crossover_elite_size");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "use_adjusted_fitness"), "set_crossover_use_adjusted_fitness", "get_crossover_use_adjusted_fitness");

    ADD_GROUP("Mutation", "");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "weight_rate"), "set_mutate_weight_rate", "get_mutate_weight_rate");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "weight_amount"), "set_mutate_weight_amount", "get_mutate_weight_amount");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "weight_min"), "set_mutate_weight_min", "get_mutate_weight_min");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "weight_max"), "set_mutate_weight_max", "get_mutate_weight_max");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "redraw_weight"), "set_mutate_redraw_weight", "get_mutate_redraw_weight");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "new_connection_rate"), "set_mutate_new_connection_rate", "get_mutate_new_connection_rate");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "new_node_rate"), "set_mutate_new_node_rate", "get_mutate_new_node_rate");
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "disable_node_rate"), "set_mutate_disable_node_rate", "get_mutate_disable_node_rate");

    // clang-format on
}

} // namespace gdneat