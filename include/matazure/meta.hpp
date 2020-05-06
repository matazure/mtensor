#pragma once

#include <matazure/config.hpp>
#include <matazure/point.hpp>

namespace matazure {
/// meta is a sub namespace under matazure, it's for meta programming
namespace meta {

namespace __walkground {
template <int_t... _Values>
MATAZURE_GENERAL inline constexpr pointi<sizeof...(_Values)> get_pointi() {
    return {_Values...};
}
}  // namespace __walkground

/// alias for std::integral_constant<int_t, _V>
template <int_t _V>
using int_t_c = std::integral_constant<int_t, _V>;

/**
 * @brief a compile time array
 * @tparam _Values packed array values
 */
template <int_t... _Values>
struct array {
    /// returns the length of meta::array
    MATAZURE_GENERAL static constexpr int_t size() { return sizeof...(_Values); }

    const static int_t rank = size();

    /// returns a runtime pointi equal to meta::array
    MATAZURE_GENERAL static constexpr pointi<sizeof...(_Values)> value() {
        return __walkground::get_pointi<_Values...>();
    }
};

/**
 * @brief gets meta::array element value by index
 * @tparam _Idx the element index
 * @param a input meta::array
 */
template <int_t _Idx, int_t... _Values>
inline MATAZURE_GENERAL constexpr int_t at_c(array<_Values...> a) {
    static_assert(_Idx < sizeof...(_Values), "_Idx should be less than meta array size");
    return array<_Values...>::value()[_Idx];
}

#define MATAZURE_META_BINARY_FUNC(name, op)                                                 \
    template <int_t... _Values0, int_t... _Values1>                                         \
    inline MATAZURE_GENERAL constexpr auto name##_c(array<_Values0...>, array<_Values1...>) \
        ->array<_Values0 op _Values1...> {                                                  \
        return array<_Values0 op _Values1...>{};                                            \
    }                                                                                       \
                                                                                            \
    template <int_t... _Values, int_t _V>                                                   \
    inline MATAZURE_GENERAL constexpr auto name##_c(array<_Values...>, int_t_c<_V>)         \
        ->array<_Values op _V...> {                                                         \
        return array<_Values op _V...>{};                                                   \
    }                                                                                       \
                                                                                            \
    template <int_t _V, int_t... _Values>                                                   \
    inline MATAZURE_GENERAL constexpr auto name##_c(int_t_c<_V>, array<_Values...>)         \
        ->array<_V op _Values...> {                                                         \
        return array<_V op _Values...>{};                                                   \
    }

/// @brief transposes meta::array
template <int_t _V0, int_t _V1>
inline MATAZURE_GENERAL constexpr auto transpose(array<_V0, _V1>) -> array<_V1, _V0> {
    return array<_V1, _V0>{};
}

MATAZURE_META_BINARY_FUNC(add, +)
MATAZURE_META_BINARY_FUNC(sub, -)
MATAZURE_META_BINARY_FUNC(mul, *)
MATAZURE_META_BINARY_FUNC(div, /)
MATAZURE_META_BINARY_FUNC(mod, %)

/// convert a meta::array to pointi
template <int_t... _Values>
inline MATAZURE_GENERAL constexpr pointi<sizeof...(_Values)> array_to_pointi(array<_Values...>) {
    return {_Values...};
}

}  // namespace meta
}  // namespace matazure
