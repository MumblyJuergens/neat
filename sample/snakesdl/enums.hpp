#pragma once

#include <meta>
#include <type_traits>

template <typename T>
    requires std::is_enum_v<T>
consteval auto make_enum_to_string_table() noexcept
{
    // 1. Reflexpr operator fetches metadata token from AST
    static constexpr auto r = ^^T;
    static constexpr auto members = std::define_static_array(std::meta::enumerators_of(r));

    std::size_t index{};
    std::array<std::pair<T, std::string_view>, members.size()> table{};

    // 2. Compile-time loop expansion
    template for (constexpr auto item : members)
    {
        table[index].first = [:item:]; // 3. Splicer operator (meta to value)
        table[index].second = std::meta::identifier_of(item);
        ++index;
    }

    return table;
}

template <typename T>
    requires std::is_enum_v<T>
constexpr std::string_view enum_to_string(T value) noexcept
{
    static constexpr auto table = make_enum_to_string_table<T>();

    template for (constexpr auto item : table)
    {
        if (item.first == value) {
            return item.second;
        }
    }

    return {};
}
