#pragma once

#include "GDNeatConfig.hpp"
#include "neat/SimplePopulation.hpp"
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <memory>

namespace gdneat
{

class GDNeatPopulation : public godot::RefCounted
{
    GDCLASS(GDNeatPopulation, godot::RefCounted)

  private:
    std::unique_ptr<neat::SimplePopulation> population;
    godot::Ref<GDNeatConfig> config;

    void create(godot::Ref<GDNeatConfig> cfg);
    void step(godot::Callable handler);
    bool _generation_is_done() const;
    void new_generation();
    int _generation() const;

  protected:
    static void _bind_methods();
};

} // namespace gdneat