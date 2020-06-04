#pragma once

#include <matazure/lambda_tensor.hpp>
#ifdef MATAZURE_CUDA
#include <matazure/cuda/lambda_tensor.hpp>
#endif

namespace matazure {

#define __MATAZURE_LINEAR_ACCESS_TENSOR_BINARY_OPERATOR(name, op)   \
    template <typename _T1, typename _T2>                           \
    struct name {                                                   \
       private:                                                     \
        _T1 x1_;                                                    \
        _T2 x2_;                                                    \
                                                                    \
       public:                                                      \
        MATAZURE_STATIC_ASSERT_DIM_MATCHED(_T1, _T2);               \
        MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_T1, _T2);        \
                                                                    \
        MATAZURE_GENERAL name(_T1 x1, _T2 x2) : x1_(x1), x2_(x2) {} \
                                                                    \
        MATAZURE_GENERAL auto operator()(int_t i) const             \
            -> decltype(this->x1_[i] op this->x2_[i]) {             \
            return x1_[i] op x2_[i];                                \
        }                                                           \
    };

#define __MATAZURE_ARRAY_INDEX_TENSOR_BINARY_OPERATOR(name, op)              \
    template <typename _T1, typename _T2>                                    \
    struct name {                                                            \
       private:                                                              \
        _T1 x1_;                                                             \
        _T2 x2_;                                                             \
                                                                             \
       public:                                                               \
        MATAZURE_STATIC_ASSERT_DIM_MATCHED(_T1, _T2);                        \
        MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_T1, _T2);                 \
        MATAZURE_GENERAL name(_T1 x1, _T2 x2) : x1_(x1), x2_(x2) {}          \
                                                                             \
        MATAZURE_GENERAL auto operator()(const pointi<_T1::rank>& idx) const \
            -> decltype(this->x1_(idx) op this->x2_(idx)) {                  \
            return x1_(idx) op x2_(idx);                                     \
        }                                                                    \
    };

#define __MATAZURE_LINEAR_ACCESS_TENSOR_WITH_VALUE_BINARY_OPERATOR(name, op)                   \
    template <typename _T>                                                                     \
    struct name {                                                                              \
       private:                                                                                \
        typedef typename _T::value_type value_type;                                            \
                                                                                               \
        _T x_;                                                                                 \
        value_type v_;                                                                         \
                                                                                               \
       public:                                                                                 \
        MATAZURE_GENERAL name(_T x, value_type v) : x_(x), v_(v) {}                            \
                                                                                               \
        MATAZURE_GENERAL auto operator()(int_t i) const -> decltype(this->x_[i] op this->v_) { \
            return x_[i] op v_;                                                                \
        }                                                                                      \
    };

#define __MATAZURE_ARRAY_INDEX_TENSOR_WITH_VALUE_BINARY_OPERATOR(name, op)  \
    template <typename _T>                                                  \
    struct name {                                                           \
       private:                                                             \
        typedef typename _T::value_type value_type;                         \
                                                                            \
        _T x_;                                                              \
        value_type v_;                                                      \
                                                                            \
       public:                                                              \
        MATAZURE_GENERAL name(_T x, value_type v) : x_(x), v_(v) {}         \
                                                                            \
        MATAZURE_GENERAL auto operator()(const pointi<_T::rank>& idx) const \
            -> decltype(this->x_(idx) op this->v_) {                        \
            return x_(idx) op v_;                                           \
        }                                                                   \
    };

#define __MATAZURE_VALUE_WITH_LINEAR_ACCESS_TENSOR_BINARY_OPERATOR(name, op) \
    template <typename _T>                                                   \
    struct name {                                                            \
       private:                                                              \
        typedef typename _T::value_type value_type;                          \
                                                                             \
        value_type v_;                                                       \
        _T x_;                                                               \
                                                                             \
       public:                                                               \
        MATAZURE_GENERAL name(value_type v, _T x) : v_(v), x_(x) {}          \
                                                                             \
        MATAZURE_GENERAL auto operator()(const int_t& i) const               \
            -> decltype((this->v_)op(this->x_[i])) {                         \
            return v_ op x_[i];                                              \
        }                                                                    \
    };

#define __MATAZURE_VALUE_WITH_ARRAY_INDEX_TENSOR_BINARY_OPERATOR(name, op)  \
    template <typename _T>                                                  \
    struct name {                                                           \
       private:                                                             \
        typedef typename _T::value_type value_type;                         \
                                                                            \
        value_type v_;                                                      \
        _T x_;                                                              \
                                                                            \
       public:                                                              \
        MATAZURE_GENERAL name(_T x, value_type v) : v_(v), x_(x) {}         \
                                                                            \
        MATAZURE_GENERAL auto operator()(const pointi<_T::rank>& idx) const \
            -> decltype(this->v_ op this->x_(idx)) {                        \
            return v_ op x_(idx);                                           \
        }                                                                   \
    };

// host tensor operations
#define TENSOR_BINARY_OPERATOR(name, op)                                                       \
    __MATAZURE_LINEAR_ACCESS_TENSOR_BINARY_OPERATOR(__##name##_are_linear_index_tensor__, op)  \
    template <typename _TS1, typename _TS2>                                                    \
    inline enable_if_t<                                                                        \
        none_device_memory<_TS1, _TS2>::value && are_linear_index<_TS1, _TS2>::value,          \
        lambda_tensor<_TS1::rank, __##name##_are_linear_index_tensor__<_TS1, _TS2>>>           \
    operator op(const tensor_expression<_TS1>& e_lhs, const tensor_expression<_TS2>& e_rhs) {  \
        return make_lambda(e_lhs().shape(),                                                    \
                           __##name##_are_linear_index_tensor__<_TS1, _TS2>(e_lhs(), e_rhs()), \
                           host_t{}, layout_t<_TS1>{});                                        \
    }                                                                                          \
    __MATAZURE_ARRAY_INDEX_TENSOR_BINARY_OPERATOR(__##name##_array_indexensor__, op)           \
    template <typename _TS1, typename _TS2>                                                    \
    inline enable_if_t<none_device_memory<_TS1, _TS2>::value &&                                \
                           !are_linear_index<_TS1, _TS2>::value,                               \
                       lambda_tensor<_TS1::rank, __##name##_array_indexensor__<_TS1, _TS2>>>   \
    operator op(const tensor_expression<_TS1>& e_lhs, const tensor_expression<_TS2>& e_rhs) {  \
        return make_lambda(e_lhs().shape(),                                                    \
                           __##name##_array_indexensor__<_TS1, _TS2>(e_lhs(), e_rhs()),        \
                           layout_t<_TS1>{});                                                  \
    }

#define TENSOR_WITH_VALUE_BINARY_OPERATOR(name, op)                                             \
    __MATAZURE_LINEAR_ACCESS_TENSOR_WITH_VALUE_BINARY_OPERATOR(                                 \
        __##name##_are_linear_index_tensor_with_value__, op)                                    \
                                                                                                \
    template <typename _TS>                                                                     \
    inline enable_if_t<                                                                         \
        none_device_memory<_TS>::value && are_linear_index<_TS>::value,                         \
        lambda_tensor<_TS::rank, __##name##_are_linear_index_tensor_with_value__<_TS>>>         \
    operator op(const tensor_expression<_TS>& e_ts, typename _TS::value_type v) {               \
        return make_lambda(e_ts().shape(),                                                      \
                           __##name##_are_linear_index_tensor_with_value__<_TS>(e_ts(), v),     \
                           layout_t<_TS>{});                                                    \
    }                                                                                           \
                                                                                                \
    __MATAZURE_VALUE_WITH_LINEAR_ACCESS_TENSOR_BINARY_OPERATOR(                                 \
        __##name##_value_with_are_linear_index_tensor__, op)                                    \
    template <typename _TS>                                                                     \
    inline enable_if_t<                                                                         \
        none_device_memory<_TS>::value && are_linear_index<_TS>::value,                         \
        lambda_tensor<_TS::rank, __##name##_value_with_are_linear_index_tensor__<_TS>>>         \
    operator op(typename _TS::value_type v, const tensor_expression<_TS>& e_ts) {               \
        return make_lambda(e_ts().shape(),                                                      \
                           __##name##_value_with_are_linear_index_tensor__<_TS>(v, e_ts()),     \
                           layout_t<_TS>{});                                                    \
    }                                                                                           \
                                                                                                \
    __MATAZURE_ARRAY_INDEX_TENSOR_WITH_VALUE_BINARY_OPERATOR(                                   \
        __##name##_array_indexensor_with_value__, op)                                           \
    template <typename _TS>                                                                     \
    inline enable_if_t<none_device_memory<_TS>::value && !are_linear_index<_TS>::value,         \
                       lambda_tensor<_TS::rank, __##name##_array_indexensor_with_value__<_TS>>> \
    operator op(const tensor_expression<_TS>& e_ts, typename _TS::value_type v) {               \
        return make_lambda(e_ts().shape(),                                                      \
                           __##name##_array_indexensor_with_value__<_TS>(e_ts(), v),            \
                           layout_t<_TS>{});                                                    \
    }                                                                                           \
                                                                                                \
    __MATAZURE_VALUE_WITH_ARRAY_INDEX_TENSOR_BINARY_OPERATOR(                                   \
        __##name##_value_with_array_indexensor__, op)                                           \
    template <typename _TS>                                                                     \
    inline enable_if_t<none_device_memory<_TS>::value && !are_linear_index<_TS>::value,         \
                       lambda_tensor<_TS::rank, __##name##_value_with_array_indexensor__<_TS>>> \
    operator op(typename _TS::value_type v, const tensor_expression<_TS>& e_ts) {               \
        return make_lambda(e_ts().shape(),                                                      \
                           __##name##_value_with_array_indexensor__<_TS>(e_ts(), v),            \
                           layout_t<_TS>{});                                                    \
    }

// device tensor operations
#define CU_TENSOR_BINARY_OPERATOR(name, op)                                                        \
    template <typename _TS1, typename _TS2>                                                        \
    inline enable_if_t<                                                                            \
        are_device_memory<_TS1, _TS2>::value && are_linear_index<_TS1, _TS2>::value,               \
        cuda::lambda_tensor<_TS1::rank, __##name##_are_linear_index_tensor__<_TS1, _TS2>>>         \
    operator op(const tensor_expression<_TS1>& e_lhs, const tensor_expression<_TS2>& e_rhs) {      \
        return make_lambda(e_lhs().shape(),                                                        \
                           __##name##_are_linear_index_tensor__<_TS1, _TS2>(e_lhs(), e_rhs()),     \
                           device_t{}, layout_t<_TS1>{});                                          \
    }                                                                                              \
    template <typename _TS1, typename _TS2>                                                        \
    inline enable_if_t<are_device_memory<_TS1, _TS2>::value &&                                     \
                           !are_linear_index<_TS1, _TS2>::value,                                   \
                       cuda::lambda_tensor<_TS1::rank, __##name##_array_indexensor__<_TS1, _TS2>>> \
    operator op(const tensor_expression<_TS1>& e_lhs, const tensor_expression<_TS2>& e_rhs) {      \
        return make_lambda(e_lhs().shape(),                                                        \
                           __##name##_array_indexensor__<_TS1, _TS2>(e_lhs(), e_rhs()),            \
                           device_t{}, layout_t<_TS1>{});                                          \
    }

#define CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(name, op)                                           \
                                                                                                 \
    template <typename _TS>                                                                      \
    inline enable_if_t<                                                                          \
        are_device_memory<_TS>::value && are_linear_index<_TS>::value,                           \
        cuda::lambda_tensor<_TS::rank, __##name##_are_linear_index_tensor_with_value__<_TS>>>    \
    operator op(const tensor_expression<_TS>& e_ts, typename _TS::value_type v) {                \
        return make_lambda(e_ts().shape(),                                                       \
                           __##name##_are_linear_index_tensor_with_value__<_TS>(e_ts(), v),      \
                           device_t{}, layout_t<_TS>{});                                         \
    }                                                                                            \
                                                                                                 \
    template <typename _TS>                                                                      \
    inline enable_if_t<                                                                          \
        are_device_memory<_TS>::value && are_linear_index<_TS>::value,                           \
        cuda::lambda_tensor<_TS::rank, __##name##_value_with_are_linear_index_tensor__<_TS>>>    \
    operator op(typename _TS::value_type v, const tensor_expression<_TS>& e_ts) {                \
        return make_lambda(e_ts().shape(),                                                       \
                           __##name##_value_with_are_linear_index_tensor__<_TS>(v, e_ts()),      \
                           device_t{}, layout_t<_TS>{});                                         \
    }                                                                                            \
                                                                                                 \
    template <typename _TS>                                                                      \
    inline enable_if_t<                                                                          \
        are_device_memory<_TS>::value && !are_linear_index<_TS>::value,                          \
        cuda::lambda_tensor<_TS::rank, __##name##_array_indexensor_with_value__<_TS>>>           \
    operator op(const tensor_expression<_TS>& e_ts, typename _TS::value_type v) {                \
        return make_lambda(e_ts().shape(),                                                       \
                           __##name##_array_indexensor_with_value__<_TS>(e_ts(), v), device_t{}, \
                           layout_t<_TS>{});                                                     \
    }                                                                                            \
                                                                                                 \
    template <typename _TS>                                                                      \
    inline enable_if_t<                                                                          \
        are_device_memory<_TS>::value && !are_linear_index<_TS>::value,                          \
        cuda::lambda_tensor<_TS::rank, __##name##_value_with_array_indexensor__<_TS>>>           \
    operator op(typename _TS::value_type v, const tensor_expression<_TS>& e_ts) {                \
        return make_lambda(e_ts().shape(),                                                       \
                           __##name##_value_with_array_indexensor__<_TS>(v, e_ts()), device_t{}, \
                           layout_t<_TS>{});                                                     \
    }

// Arithmetic
TENSOR_BINARY_OPERATOR(add, +)
TENSOR_BINARY_OPERATOR(sub, -)
TENSOR_BINARY_OPERATOR(mul, *)
TENSOR_BINARY_OPERATOR(div, /)
TENSOR_BINARY_OPERATOR(mod, %)
TENSOR_WITH_VALUE_BINARY_OPERATOR(add, +)
TENSOR_WITH_VALUE_BINARY_OPERATOR(sub, -)
TENSOR_WITH_VALUE_BINARY_OPERATOR(mul, *)
TENSOR_WITH_VALUE_BINARY_OPERATOR(div, /)
TENSOR_WITH_VALUE_BINARY_OPERATOR(mod, %)
// Bit
TENSOR_BINARY_OPERATOR(left_shift, <<)
TENSOR_BINARY_OPERATOR(right_shift, >>)
TENSOR_BINARY_OPERATOR(bit_or, |)
TENSOR_BINARY_OPERATOR(bit_and, &)
TENSOR_BINARY_OPERATOR(bit_xor, ^)
TENSOR_WITH_VALUE_BINARY_OPERATOR(left_shift, <<)
TENSOR_WITH_VALUE_BINARY_OPERATOR(right_shift, >>)
TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_or, |)
TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_and, &)
TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_xor, ^)
// Logic
TENSOR_BINARY_OPERATOR(or, ||)
TENSOR_BINARY_OPERATOR(and, &&)
TENSOR_WITH_VALUE_BINARY_OPERATOR(or, ||)
TENSOR_WITH_VALUE_BINARY_OPERATOR(and, &&)
// Compapre
TENSOR_BINARY_OPERATOR(gt, >)
TENSOR_BINARY_OPERATOR(lt, <)
TENSOR_BINARY_OPERATOR(ge, >=)
TENSOR_BINARY_OPERATOR(le, <=)
TENSOR_BINARY_OPERATOR(equal, ==)
TENSOR_BINARY_OPERATOR(not_equal, !=)
TENSOR_WITH_VALUE_BINARY_OPERATOR(gt, >)
TENSOR_WITH_VALUE_BINARY_OPERATOR(lt, <)
TENSOR_WITH_VALUE_BINARY_OPERATOR(ge, >=)
TENSOR_WITH_VALUE_BINARY_OPERATOR(le, <=)
TENSOR_WITH_VALUE_BINARY_OPERATOR(equal, ==)
TENSOR_WITH_VALUE_BINARY_OPERATOR(not_equal, !=)

#ifdef MATAZURE_CUDA
// Arithmetic
CU_TENSOR_BINARY_OPERATOR(add, +)
CU_TENSOR_BINARY_OPERATOR(sub, -)
CU_TENSOR_BINARY_OPERATOR(mul, *)
CU_TENSOR_BINARY_OPERATOR(div, /)
CU_TENSOR_BINARY_OPERATOR(mod, %)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(add, +)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(sub, -)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(mul, *)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(div, /)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(mod, %)
// Bit
CU_TENSOR_BINARY_OPERATOR(left_shift, <<)
CU_TENSOR_BINARY_OPERATOR(right_shift, >>)
CU_TENSOR_BINARY_OPERATOR(bit_or, |)
CU_TENSOR_BINARY_OPERATOR(bit_and, &)
CU_TENSOR_BINARY_OPERATOR(bit_xor, ^)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(left_shift, <<)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(right_shift, >>)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_or, |)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_and, &)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_xor, ^)
// Logic
CU_TENSOR_BINARY_OPERATOR(or, ||)
CU_TENSOR_BINARY_OPERATOR(and, &&)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(or, ||)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(and, &&)
// Compapre
CU_TENSOR_BINARY_OPERATOR(gt, >)
CU_TENSOR_BINARY_OPERATOR(lt, <)
CU_TENSOR_BINARY_OPERATOR(ge, >=)
CU_TENSOR_BINARY_OPERATOR(le, <=)
CU_TENSOR_BINARY_OPERATOR(equal, ==)
CU_TENSOR_BINARY_OPERATOR(not_equal, !=)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(gt, >)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(lt, <)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(ge, >=)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(le, <=)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(equal, ==)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(not_equal, !=)

#endif
}
