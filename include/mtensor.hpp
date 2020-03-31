#pragma once

#include <matazure/binary_operator.hpp>
#include <matazure/dynamic_tensor.hpp>
#include <matazure/geometry.hpp>
#include <matazure/io.hpp>
#include <matazure/meta.hpp>
#include <matazure/view/view.hpp>

#ifdef MATAZURE_OPENMP
#include <matazure/omp_for_index.hpp>
#endif

#ifdef MATAZURE_CUDA
#include <matazure/cuda/algorithm.hpp>
#include <matazure/cuda/exception.hpp>
#include <matazure/cuda/execution.hpp>
#include <matazure/cuda/runtime.hpp>
#include <matazure/cuda/tensor.hpp>
#endif
