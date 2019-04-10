#pragma once

#include <matazure/config.hpp>

namespace matazure {

/// sequence execution policy
struct sequence_policy {};

/// sequence execution policy with auto vectorization
struct sequence_vectorized_policy {};

#ifdef MATAZURE_OPENMP

/// openmp parallel execution policy
struct omp_policy {};

#endif

}
