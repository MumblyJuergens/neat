#pragma once

#include "neat/types.hpp"
#include <cstdint>
#include <mj/size.hpp>
#include <random>
#include <set>

namespace neat
{

class [[nodiscard]] Random final
{
    std::default_random_engine engine;
    std::uniform_real_distribution<real_t> randr;
    std::uniform_int_distribution<int> randi;
    std::normal_distribution<real_t> randgaussian{0.0_r, 1.0_r};

  public:
    [[nodiscard]] Random(uint32_t seed) : engine{seed} {}

    [[nodiscard]] real_t weight() noexcept { return range(-1.0_r, 1.0_r); }

    [[nodiscard]] real_t canonical() noexcept
    {
        return randr(engine, std::uniform_real_distribution<real_t>::param_type{});
    }

    [[nodiscard]] real_t range(const real_t max)
    {
        return randr(engine, std::uniform_real_distribution<real_t>::param_type{0.0_r, max});
    }

    template <typename Range>
    [[nodiscard]] auto &item(Range &&r)
    {
        return r.at(mj::sz_t(range(mj::isize(r) - 1)));
    }

    template <typename T>
    [[nodiscard]] auto &item(std::set<T> &r)
    {
        return *std::next(r.begin(), range(size(r) - 1));
    }

    [[nodiscard]] real_t range(const real_t min, const real_t max)
    {
        return randr(engine, std::uniform_real_distribution<real_t>::param_type{min, max});
    }

    [[nodiscard]] int range(const int max)
    {
        return randi(engine, std::uniform_int_distribution<int>::param_type{0, max});
    }

    [[nodiscard]] real_t gaussian() { return randgaussian(engine); }

    [[nodiscard]] real_t canonical_skewed_high(const real_t strength)
    {
        return 1.0_r - std::pow(1.0_r - canonical(), strength);
    }

    [[nodiscard]] real_t canonical_skewed_low(const real_t strength) { return 1.0_r - std::pow(canonical(), strength); }
};

} // namespace neat