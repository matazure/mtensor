#pragma once

#include <matazure/view/broadcast.hpp>
#include <matazure/view/cast.hpp>
#include <matazure/view/clamp_zero.hpp>
#include <matazure/view/eye.hpp>
#include <matazure/view/gather.hpp>
#include <matazure/view/map.hpp>
#include <matazure/view/meshgrid.hpp>
#include <matazure/view/one.hpp>
#include <matazure/view/permute.hpp>
#include <matazure/view/slice.hpp>
#include <matazure/view/stride.hpp>
#include <matazure/view/zero.hpp>
#ifndef MATAZURE_CUDA
#include <matazure/view/zip.hpp>
#else
#ifdef __CUDACC_RELAXED_CONSTEXPR__
#include <matazure/view/zip.hpp>
#endif
#endif