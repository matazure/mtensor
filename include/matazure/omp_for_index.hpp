#pragma once

#include <matazure/point.hpp>
#include <matazure/type_traits.hpp>

#define MATAZURE_STRINGIFY(a) #a

#ifdef MATAZURE_OPENMP
#if defined(_MSC_VER)
#define MATAZURE_OPENMP_PARALLEL_FOR(n) __pragma(omp parallel for)
#else
// omp collapse will effect the memory continues
// #if _OPENMP >= 200805
//     // #define PRIVATE_MATAZURE_PARALLEL_FOR(n) "omp parallel for schedule(dynamic, 1)
// collapse(" #n ")"
//     #define MATAZURE_OPENMP_PARALLEL_FOR(n) _Pragma(MATAZURE_STRINGIFY(omp parallel for
//  schedule(dynamic, 1) collapse(n)))
// #else
#define MATAZURE_OPENMP_PARALLEL_FOR(n) _Pragma("omp parallel for")
// #endif
#endif
#endif

namespace matazure {

struct omp_policy {};

/**
 * @brief for each linear index, apply fun by the openmp parallel policy
 * @param first the first index
 * @param last the last index
 * @param fun the functor,  int_t -> value pattern.
 */
template <typename _Fun>
inline void for_index(omp_policy policy, int_t first, int_t last, _Fun fun) {
    MATAZURE_OPENMP_PARALLEL_FOR(1)
    for (int_t i = first; i < last; ++i) {
        fun(i);
    }
}

/**
 * @brief for each 1-dim array index, apply fun by the openmp parallel policy
 * @param origin the origin index of the 1-dim range
 * @param end the end index
 * @param fun the functor,  pointi<1> -> value pattern.
 */
template <typename _Fun>
inline void for_index(omp_policy, pointi<1> origin, pointi<1> end, _Fun fun) {
    MATAZURE_OPENMP_PARALLEL_FOR(1)
    for (int_t i = origin[0]; i < end[0]; ++i) {
        fun(pointi<1>{{i}});
    }
}

/**
 * @brief for each 2-dim array index, apply fun by the openmp parallel policy
 * @param origin the origin index of the 2-dim range
 * @param end the end index
 * @param fun the functor,  pointi<2> -> value pattern.
 */
template <typename _Fun>
inline void for_index(omp_policy, pointi<2> origin, pointi<2> end, _Fun fun) {
    MATAZURE_OPENMP_PARALLEL_FOR(2)
    for (int_t i = origin[0]; i < end[0]; ++i) {
        for (int_t j = origin[1]; j < end[1]; ++j) {
            fun(pointi<2>{{i, j}});
        }
    }
}

/**
 * @brief for each 3-dim array index, apply fun by the openmp parallel policy
 * @param origin the origin index of the 3-dim range
 * @param end the end index
 * @param fun the functor,  pointi<3> -> value pattern.
 */
template <typename _Fun>
inline void for_index(omp_policy, pointi<3> origin, pointi<3> end, _Fun fun) {
    MATAZURE_OPENMP_PARALLEL_FOR(3)
    for (int_t i = origin[0]; i < end[0]; ++i) {
        for (int_t j = origin[1]; j < end[1]; ++j) {
            for (int_t k = origin[2]; k < end[2]; ++k) {
                fun(pointi<3>{{i, j, k}});
            }
        }
    }
}

/**
 * @brief for each 4-dim array index, apply fun by the openmp parallel policy
 * @param origin the origin index of the 4-dim range
 * @param end the end index
 * @param fun the functor,  pointi<4> -> value pattern.
 */
template <typename _Fun>
inline void for_index(omp_policy, pointi<4> origin, pointi<4> end, _Fun fun) {
    MATAZURE_OPENMP_PARALLEL_FOR(4)
    for (int_t i = origin[0]; i < end[0]; ++i) {
        for (int_t j = origin[1]; j < end[1]; ++j) {
            for (int_t k = origin[2]; k < end[2]; ++k) {
                for (int_t l = origin[3]; k < end[3]; ++k) {
                    fun(pointi<4>{{i, j, k, l}});
                }
            }
        }
    }
}

}
