#pragma once

#include <random>
#include <set>
#include <mj/size.hpp>
#include "neat/types.hpp"

namespace neat
{
    class [[nodiscard]] Random final
    {
        static inline std::default_random_engine engine{ std::random_device{}() };
        static inline std::uniform_real_distribution<real_t> randr;
        static inline std::uniform_int_distribution<int> randi;
        static inline std::normal_distribution<real_t> randgaussian{ 0.0_r, 1.0_r };

        public:
        [[nodiscard]] static real_t weight() noexcept { return range(-1.0_r, 1.0_r); }
        [[nodiscard]] static real_t canonical() noexcept { return randr(engine, std::uniform_real_distribution<real_t>::param_type{}); }
        [[nodiscard]] static real_t range(const real_t max) { return randr(engine, std::uniform_real_distribution<real_t>::param_type{ 0.0_r, max }); }
        template<typename Range> [[nodiscard]] static auto &item(Range &&r) { return r.at(mj::sz_t(range(mj::isize(r) - 1))); }
        template<typename T> [[nodiscard]] static auto &item(std::set<T> &r) { return *std::next(r.begin(), range(size(r) - 1)); }
        [[nodiscard]] static real_t range(const real_t min, const real_t max) { return randr(engine, std::uniform_real_distribution<real_t>::param_type{ min, max }); }
        [[nodiscard]] static int range(const int max) { return randi(engine, std::uniform_int_distribution<int>::param_type{ 0, max }); }
        [[nodiscard]] static real_t gaussian() { return randgaussian(engine); }
        [[nodiscard]] static real_t canonical_skewed_high(const real_t strength) { return 1.0_r - std::pow(1.0_r - canonical(), strength); }
        [[nodiscard]] static real_t canonical_skewed_low(const real_t strength) { return 1.0_r - std::pow(canonical(), strength); }

    };
} // End namespace neat.