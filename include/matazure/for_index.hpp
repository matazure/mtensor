#pragma once

#include <matazure/point.hpp>
#include <matazure/type_traits.hpp>

namespace matazure {

struct sequence_policy {};

/// special is_linear_array for point
template <typename _Type, int_t _Rank>
struct _Is_linear_array<point<_Type, _Rank>> : bool_constant<true> {};

/**
 * @brief for each linear index, apply fun by the sequence policy
 * @param first the first index
 * @param last the last index
 * @param fun the functor,  int_t -> value pattern.
 */
MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(sequence_policy, int_t first, int_t last, _Fun fun) {
    for (int_t i = first; i < last; ++i) {
        fun(i);
    }
}

/**
 * @brief for each linear index, apply fun by the sequence policy
 * @param first the first index
 * @param last the last index
 * @param fun the functor,  int_t -> value pattern.
 */
MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(int_t first, int_t last, _Fun fun) {
    sequence_policy seq{};
    for_index(seq, first, last, fun);
}

/**

*/
MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(int_t last, _Fun fun) {
    for_index(0, last, fun);
}

/**
 * @brief for each 1-dim array index, apply fun by the sequence policy
 * @param origin the origin index of the 1-dim range
 * @param end the end index
 * @param fun the functor,  pointi<1> -> value pattern.
 */
MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(sequence_policy, pointi<1> origin, pointi<1> end, _Fun fun) {
    for (int_t i = origin[0]; i < end[0]; ++i) {
        fun(pointi<1>{{i}});
    }
}

/**
 * @brief for each 2-dim array index, apply fun by the sequence policy
 * @param origin the origin index of the 2-dim range
 * @param end the end index
 * @param fun the functor,  pointi<2> -> value pattern.
 */
MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(sequence_policy, pointi<2> origin, pointi<2> end, _Fun fun) {
    for (int_t i = origin[0]; i < end[0]; ++i) {
        for (int_t j = origin[1]; j < end[1]; ++j) {
            fun(pointi<2>{{i, j}});
        }
    }
}

/**
 * @brief for each 3-dim array index, apply fun by the sequence policy
 * @param origin the origin index of the 3-dim range
 * @param end the end index
 * @param fun the functor,  pointi<3> -> value pattern.
 */
MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(sequence_policy, pointi<3> origin, pointi<3> end, _Fun fun) {
    for (int_t i = origin[0]; i < end[0]; ++i) {
        for (int_t j = origin[1]; j < end[1]; ++j) {
            for (int_t k = origin[2]; k < end[2]; ++k) {
                fun(pointi<3>{{i, j, k}});
            }
        }
    }
}

/**
 * @brief for each 4-dim array index, apply fun by the sequence policy
 * @param origin the origin index of the 4-dim range
 * @param end the end index
 * @param fun the functor,  pointi<4> -> value pattern.
 */
MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(sequence_policy, pointi<4> origin, pointi<4> end, _Fun fun) {
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

/**
 * @brief for each array index, apply fun by the sequence policy
 * @param origin the origin index of the range
 * @param end the end index
 * @param fun the functor,  pointi -> value pattern.
 */
MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun, int_t _Rank>
MATAZURE_GENERAL inline void for_index(pointi<_Rank> origin, pointi<_Rank> end, _Fun fun) {
    sequence_policy policy{};
    for_index(policy, origin, end, fun);
}

MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun, int_t _Rank>
MATAZURE_GENERAL inline void for_index(pointi<_Rank> end, _Fun fun) {
    for_index(zero<pointi<_Rank>>::value(), end, fun);
}

MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Policy, typename _Fun>
MATAZURE_GENERAL inline void for_index(_Policy policy, int_t last, _Fun fun) {
    for_index(policy, 0, last, fun);
}

MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Policy, typename _Fun, int_t _Rank>
MATAZURE_GENERAL inline void for_index(_Policy policy, pointi<_Rank> end, _Fun fun) {
    for_index(policy, zero<pointi<_Rank>>::value(), end, fun);
}

}  // namespace matazure
