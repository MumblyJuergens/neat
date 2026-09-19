#pragma once

#include "neat/Genome.hpp"
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/typed_array.hpp>

namespace gdneat
{

class GDNeatGenome : public godot::RefCounted
{
    GDCLASS(GDNeatGenome, godot::RefCounted)

  protected:
    static void _bind_methods();

  public:
    neat::Genome *_genome{};
    godot::TypedArray<double> simple_step(const godot::TypedArray<double> inputs);
    void set_fitness(double value);
    double get_fitness() const;
};

} // namespace gdneat