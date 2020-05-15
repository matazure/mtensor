#pragma once

#include <benchmark/benchmark.h>
#include <cmath>
#include <mtensor.hpp>
#include "padding_layout.hpp"

using namespace matazure;

using matazure::device_t;
using matazure::host_t;

constexpr int_t operator"" _G(unsigned long long v) {
    return static_cast<int_t>(1000 * 1000 * 1000 * v);
}
constexpr int_t operator"" _M(unsigned long long v) { return static_cast<int_t>(1000 * 1000 * v); }
constexpr int_t operator"" _K(unsigned long long v) { return static_cast<int_t>(1000 * v); }
