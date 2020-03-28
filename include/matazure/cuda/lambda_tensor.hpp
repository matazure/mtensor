#pragma once

#include <cuda_runtime.h>
#include <matazure/cuda/algorithm.hpp>
#include <matazure/cuda/runtime.hpp>
#include <matazure/cuda/tensor.hpp>

namespace matazure {
namespace cuda {

template <int_t _Rank, typename _Func, typename _Layout = first_major_layout<_Rank>>
class general_lambda_tensor
    : public tensor_expression<general_lambda_tensor<_Rank, _Func, _Layout>> {
    typedef function_traits<_Func> functor_traits;

   public:
    static const int_t rank = _Rank;
    typedef typename functor_traits::result_type reference;
    typedef remove_reference_t<reference> value_type;
    typedef typename matazure::internal::get_functor_accessor_type<_Rank, _Func>::type index_type;
    typedef device_tag memory_type;
    typedef _Layout layout_type;

   public:
    general_lambda_tensor(const pointi<rank>& ext, _Func fun)
        : shape_(ext), layout_(ext), functor_(fun) {}

    MATAZURE_GENERAL reference operator[](int_t i) const { return offset_imp<index_type>(i); }

    MATAZURE_GENERAL reference operator[](const pointi<rank>& idx) const {
        return index_imp<index_type>(idx);
    }

    template <typename... _Idx>
    MATAZURE_GENERAL reference operator()(_Idx... idx) const {
        return (*this)[pointi<rank>{idx...}];
    }

    template <typename _ExecutionPolicy>
    tensor<decay_t<value_type>, rank> persist(_ExecutionPolicy policy) const {
        tensor<decay_t<value_type>, rank> re(this->shape());
        cuda::copy(policy, *this, re);
        return re;
    }

    tensor<decay_t<value_type>, rank> persist() const {
        parallel_execution_policy policy{};
        policy.total_size(this->size());
        return persist(policy);
    }

    MATAZURE_GENERAL pointi<rank> shape() const { return shape_; }
    MATAZURE_GENERAL int_t size() const { return layout_.stride()[rank - 1]; }

   public:
    template <typename _Mode>
    MATAZURE_GENERAL enable_if_t<is_same<_Mode, array_index>::value, reference> index_imp(
        pointi<rank> index) const {
        return functor_(index);
    }

    template <typename _Mode>
    MATAZURE_GENERAL enable_if_t<is_same<_Mode, linear_index>::value, reference> index_imp(
        pointi<rank> index) const {
        return (*this)[layout_.index2offset(index)];
    }

    template <typename _Mode>
    MATAZURE_GENERAL enable_if_t<is_same<_Mode, array_index>::value, reference> offset_imp(
        int_t i) const {
        return (*this)[layout_.offset2index(i)];
    }

    template <typename _Mode>
    MATAZURE_GENERAL enable_if_t<is_same<_Mode, linear_index>::value, reference> offset_imp(
        int_t i) const {
        return functor_(i);
    }

   private:
    const pointi<rank> shape_;
    const layout_type layout_;
    const _Func functor_;
};

template <int_t _Rank, typename _Func, typename _Layout = first_major_layout<_Rank>>
using device_lambda_tensor = general_lambda_tensor<_Rank, _Func, _Layout>;

template <int_t _Rank, typename _Func>
inline auto make_device_lambda(pointi<_Rank> ext, _Func fun)
    -> cuda::device_lambda_tensor<_Rank, _Func> {
    return cuda::device_lambda_tensor<_Rank, _Func>(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_general_lambda(pointi<_Rank> ext, _Func fun)
    -> general_lambda_tensor<_Rank, _Func> {
    return general_lambda_tensor<_Rank, _Func>(ext, fun);
}

}  // namespace cuda
}  // namespace matazure
