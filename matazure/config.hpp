#pragma once

#include <cstdlib>
#include <cassert>
#include <cstring>
#include <type_traits>
#include <stdexcept>
#include <limits>
#include <memory>
#include <tuple>

//for using
namespace matazure {

typedef int int_t;

using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;
using std::move;

typedef unsigned char byte;

using std::decay;
using std::remove_const;
using std::remove_all_extents;
using std::forward;

using std::is_same;
using std::conditional;
using std::enable_if;
using std::integral_constant;
using std::is_integral;
using std::numeric_limits;

using std::tuple;
using std::make_tuple;
using std::tuple_size;
using std::tuple_element;
using std::tie;
using std::get;

template<bool _Val>
using bool_constant = integral_constant<bool, _Val>;

template<class _Ty>
using decay_t = typename decay<_Ty>::type;

template<bool _Test, class _Ty = void>
using enable_if_t = typename enable_if<_Test, _Ty>::type;

template<bool _Test, class _Ty1, class _Ty2>
using conditional_t = typename conditional<_Test, _Ty1, _Ty2>::type;

template<class _Ty>
using remove_const_t = typename remove_const<_Ty>::type;

struct blank_t {};

}

//for assert
#define MATAZURE_STATIC_ASSERT_DIM_MATCHED(T1, T2) static_assert(T1::rank == T2::rank, "the rank is not matched")

#define MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(T1, T2) static_assert(std::is_same<typename T1::value_type, typename T2::value_type>::value, \
"the value type is not matched")

#define MATAZURE_STATIC_ASSERT_IS_MATRIX(_Matrix) static_assert(std::is_same<matazure::tensor_expression<_Matrix> , _Matrix>::value && _Matrix::rank == 2)

#define MATAZURE_CURRENT_FUNCTION "(unknown)"

#if defined(__has_builtin)
#if __has_builtin(__builtin_expect)
#define MATAZURE_LIKELY(x) __builtin_expect(x, 1)
#define MATAZURE_UNLIKELY(x) __builtin_expect(x, 0)
#else
#define MATAZURE_LIKELY(x) x
#define MATAZURE_UNLIKELY(x) x
#endif
#else
#define MATAZURE_LIKELY(x) x
#define MATAZURE_UNLIKELY(x) x
#endif

#if defined(MATAZURE_DISABLE_ASSERTS) || defined(NDEBUG)

# define MATAZURE_ASSERT(expr) ((void)0)
# define MATAZURE_ASSERT_MSG(expr, msg) ((void)0)

#elif defined(MATAZURE_ENABLE_ASSERT_HANDLER)
#error not work

namespace matazure
{
void assertion_failed(char const * expr, char const * function, char const * file, long line); //
void assertion_failed_msg(char const * expr, char const * msg, char const * function, char const * file, long line); //
}

#define MATAZURE_ASSERT(expr) (MATAZURE_LIKELY(!!(expr))? ((void)0): ::matazure::assertion_failed(#expr, MATAZURE_CURRENT_FUNCTION, __FILE__, __LINE__))
#define MATAZURE_ASSERT_MSG(expr, msg) (MATAZURE_LIKELY(!!(expr))? ((void)0): ::matazure::assertion_failed_msg(#expr, msg, MATAZURE_CURRENT_FUNCTION, __FILE__, __LINE__))

#else

# include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same

# define MATAZURE_ASSERT(expr) assert(expr)
# define MATAZURE_ASSERT_MSG(expr, msg) assert((expr)&&(msg))

#endif

//for cuda
#ifdef __CUDACC__

#define MATAZURE_CUDA

#endif

#ifdef MATAZURE_CUDA

#define MATAZURE_GENERAL __host__ __device__
#define MATAZURE_DEVICE __device__
#define MATAZURE_GLOBAL __global__

#define __matazure__ MATAZURE_GENERAL

#else

#define MATAZURE_DEVICE
#define MATAZURE_GENERAL
#define MATAZURE_GLOBAL

#define __matazure__

#endif
