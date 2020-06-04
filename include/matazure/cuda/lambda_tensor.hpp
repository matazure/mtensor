#pragma once

#include <cuda_runtime.h>
#include <matazure/cuda/algorithm.hpp>
#include <matazure/cuda/lambda_tensor.hpp>
#include <matazure/cuda/runtime.hpp>
#include <matazure/cuda/tensor.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/type_traits.hpp>

namespace matazure {
namespace cuda {

template <int_t _Rank, typename _Fun, typename _Layout = row_major_layout<_Rank>>
class lambda_tensor : public tensor_expression<lambda_tensor<_Rank, _Fun, _Layout>> {
    typedef function_traits<_Fun> functor_traits;

   public:
    static const int_t rank = _Rank;
    typedef typename functor_traits::result_type reference;
    typedef remove_reference_t<reference> value_type;
    typedef typename matazure::internal::get_functor_accessor_type<_Rank, _Fun>::type index_type;
    typedef device_t runtime_type;
    typedef _Layout layout_type;

   public:
    lambda_tensor(const pointi<rank>& ext, _Fun fun) : shape_(ext), layout_(ext), functor_(fun) {}

    MATAZURE_GENERAL reference operator[](int_t i) const { return offset_imp<index_type>(i); }

    MATAZURE_GENERAL reference operator()(const pointi<rank>& idx) const {
        return index_imp<index_type>(idx);
    }

    template <typename... _Idx>
    MATAZURE_GENERAL reference operator()(_Idx... idx) const {
        return (*this)(pointi<rank>{idx...});
    }

    template <typename _ExecutionPolicy>
    tensor<decay_t<value_type>, rank> persist(_ExecutionPolicy policy) const {
        tensor<decay_t<value_type>, rank> re(this->shape());
        cuda::copy(policy, *this, re);
        return re;
    }

    tensor<decay_t<value_type>, rank> persist() const {
        for_index_execution_policy policy{};
        policy.total_size(this->size());
        return persist(policy);
    }

    MATAZURE_GENERAL pointi<rank> shape() const { return shape_; }

    MATAZURE_GENERAL int_t shape(int_t i) const { return shape()[i]; }

    MATAZURE_GENERAL int_t size() const { return layout_.size(); }

    MATAZURE_GENERAL layout_type layout() const { return layout_; }

    MATAZURE_GENERAL constexpr runtime_type runtime() const { return runtime_type{}; }

   public:
#pragma nv_exec_check_disable
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
    const _Fun functor_;
};

template <int_t _Rank, typename _Fun>
inline auto make_lambda(pointi<_Rank> ext, _Fun fun) -> lambda_tensor<_Rank, _Fun> {
    return lambda_tensor<_Rank, _Fun>(ext, fun);
}

template <int_t _Rank, typename _Fun, typename _Layout>
inline auto make_lambda(pointi<_Rank> ext, _Fun fun, _Layout)
    -> lambda_tensor<_Rank, _Fun, _Layout> {
    return lambda_tensor<_Rank, _Fun, _Layout>(ext, fun);
}

}  // namespace cuda

template <int_t _Rank, typename _Fun>
inline auto make_lambda(pointi<_Rank> ext, _Fun fun, device_t)
    -> decltype(cuda::make_lambda(ext, fun)) {
    return cuda::make_lambda(ext, fun);
}

template <int_t _Rank, typename _Fun, typename _Layout>
inline auto make_lambda(pointi<_Rank> ext, _Fun fun, device_t, _Layout layout)
    -> decltype(cuda::make_lambda(ext, fun, layout)) {
    return cuda::make_lambda(ext, fun, layout);
}

template <int_t _Rank, typename _Fun, typename _Layout>
inline auto make_lambda(pointi<_Rank> ext, _Fun fun, _Layout layout, device_t)
    -> decltype(cuda::make_lambda(ext, fun, layout)) {
    return cuda::make_lambda(ext, fun, layout);
}

}  // namespace matazure
