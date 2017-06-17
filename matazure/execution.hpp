#pragma once

#include <matazure/config.hpp>

namespace matazure{

struct sequence_policy{

};

struct sequence_vectorized_policy{

};

#ifdef _OPENMP

struct omp_policy{

};

struct omp_vectorized_policy{

};

#endif

}
