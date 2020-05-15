#pragma once

#include <type_traits>

namespace matazure {

#ifdef __cpp_lib_integer_sequence
//使用标准库的integer_sequence
using std::integer_sequence;
using std::make_integer_sequence;
#else

template <typename _T, _T... Indices>
struct integer_sequence;

template <>
struct integer_sequence<int_t, 0> {
    typedef int_t value_type;

    static constexpr std::size_t size() noexcept { return 1; }
};
template <>
struct integer_sequence<int_t, 0, 1> {
    typedef int_t value_type;

    static constexpr std::size_t size() noexcept { return 2; }
};
template <>
struct integer_sequence<int_t, 0, 1, 2> {
    typedef int_t value_type;

    static constexpr std::size_t size() noexcept { return 3; }
};
template <>
struct integer_sequence<int_t, 0, 1, 2, 3> {
    typedef int_t value_type;

    static constexpr std::size_t size() noexcept { return 4; }
};
template <>
struct integer_sequence<int_t, 0, 1, 2, 3, 4> {
    typedef int_t value_type;

    static constexpr std::size_t size() noexcept { return 5; }
};
template <>
struct integer_sequence<int_t, 0, 1, 2, 3, 4, 5> {
    typedef int_t value_type;

    static constexpr std::size_t size() noexcept { return 6; }
};

template <typename _T, _T _N>
struct make_integer_sequence_helper;

template <>
struct make_integer_sequence_helper<int_t, 1> {
    typedef integer_sequence<int_t, 0> type;
};
template <>
struct make_integer_sequence_helper<int_t, 2> {
    typedef integer_sequence<int_t, 0, 1> type;
};
template <>
struct make_integer_sequence_helper<int_t, 3> {
    typedef integer_sequence<int_t, 0, 1, 2> type;
};
template <>
struct make_integer_sequence_helper<int_t, 4> {
    typedef integer_sequence<int_t, 0, 1, 2, 3> type;
};
template <>
struct make_integer_sequence_helper<int_t, 5> {
    typedef integer_sequence<int_t, 0, 1, 2, 3, 4> type;
};
template <>
struct make_integer_sequence_helper<int_t, 6> {
    typedef integer_sequence<int_t, 0, 1, 2, 3, 4, 5> type;
};

template <typename _T, _T _N>
using make_integer_sequence = typename make_integer_sequence_helper<_T, _N>::type;

#endif
}