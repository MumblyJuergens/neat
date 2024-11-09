#pragma once

#include <random>
#include <set>

namespace neat
{
    class [[nodiscard]] Random final
    {
        static inline std::default_random_engine engine{ std::random_device{}() };
        static inline std::uniform_real_distribution<float> randf;
        static inline std::uniform_int_distribution<std::size_t> randull;
        static inline std::normal_distribution<float> randgaussian{ 0.0f, 1.0f };

        public:
        [[nodiscard]] static float weight() noexcept { return range(-1.0f, 1.0f); }
        [[nodiscard]] static float canonical() noexcept { return randf(engine, std::uniform_real_distribution<float>::param_type{}); }
        [[nodiscard]] static float range(const float max) { return randf(engine, std::uniform_real_distribution<float>::param_type{ 0.0f, max }); }
        template<typename Range> [[nodiscard]] static auto &item(Range &&r) { return r.at(range(size(r) - 1)); }
        template<typename T> [[nodiscard]] static auto &item(std::set<T> &r) { return *std::next(r.begin(), range(size(r) - 1)); }
        [[nodiscard]] static float range(const float min, const float max) { return randf(engine, std::uniform_real_distribution<float>::param_type{ min, max }); }
        [[nodiscard]] static std::size_t range(const std::size_t max) { return randull(engine, std::uniform_int_distribution<std::size_t>::param_type{ 0ull, max }); }
        [[nodiscard]] static float gaussian() { return randgaussian(engine); }
        [[nodiscard]] static float canonical_skewed(const float strength) { return 1.0f - std::powf(1.0f - canonical(), strength); }
    };
} // End namespace neat.