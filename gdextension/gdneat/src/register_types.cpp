#include "GDNeatConfig.hpp"
#include "GDNeatGenome.hpp"
#include "GDNeatPopulation.hpp"

#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

void initialize_gdneat_module(godot::ModuleInitializationLevel level)
{
    if (level != godot::MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    GDREGISTER_CLASS(gdneat::GDNeatGenome);
    GDREGISTER_CLASS(gdneat::GDNeatConfig);
    GDREGISTER_RUNTIME_CLASS(gdneat::GDNeatPopulation);
}

void uninitialize_gdneat_module(godot::ModuleInitializationLevel level)
{
    if (level != godot::MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
}

extern "C" GDExtensionBool GDE_EXPORT gdneat_library_init(GDExtensionInterfaceGetProcAddress get_proc,
                                                          const GDExtensionClassLibraryPtr library,
                                                          GDExtensionInitialization *initialization)
{
    godot::GDExtensionBinding::InitObject init_obj{get_proc, library, initialization};

    init_obj.register_initializer(initialize_gdneat_module);
    init_obj.register_terminator(uninitialize_gdneat_module);
    init_obj.set_minimum_library_initialization_level(godot::MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}