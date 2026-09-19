#include "GDNeatGenome.hpp"
#include "neat/types.hpp"

namespace gdneat
{

void GDNeatGenome::_bind_methods()
{
    godot::ClassDB::bind_method(godot::D_METHOD("simple_step"), &GDNeatGenome::simple_step);
    godot::ClassDB::bind_method(godot::D_METHOD("set_fitness"), &GDNeatGenome::set_fitness);
    godot::ClassDB::bind_method(godot::D_METHOD("get_fitness"), &GDNeatGenome::get_fitness);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "fitness"), "set_fitness", "get_fitness");
}

godot::TypedArray<double> GDNeatGenome::simple_step(const godot::TypedArray<double> inputs)
{
    godot::TypedArray<double> result;
    std::vector<neat::real_t> instuff, outstuff;
    for (auto d : inputs) {
        neat::real_t dd = d;
        instuff.push_back(static_cast<neat::real_t>(dd));
    }
    _genome->simple_step(instuff, outstuff, std::tanh);
    for (auto d : outstuff) {
        result.push_back(static_cast<double>(d));
    }
    return result;
}

void GDNeatGenome::set_fitness(double value) { _genome->set_fitness(static_cast<neat::real_t>(value)); }
double GDNeatGenome::get_fitness() const
{
    // Godot can try to read properties before the objext is constructed properly, causing a crash.
    return _genome ? static_cast<double>(_genome->fitness()) : 0.0;
}

} // namespace gdneat