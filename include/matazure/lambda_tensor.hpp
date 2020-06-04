
#pragma once

#include <matazure/layout.hpp>
#include <matazure/local_tensor.hpp>
#include <matazure/tensor.hpp>
#include <matazure/type_traits.hpp>

namespace matazure {

namespace internal {

template <int_t _Rank, typename _Fun>
struct get_functor_accessor_type {
   private:
    typedef function_traits<_Fun> function_traits_t;
    static_assert(function_traits_t::arguments_size == 1, "functor must be unary");
    typedef decay_t<typename function_traits_t::template arguments<0>::type> _tmp_type;

   public:
    typedef conditional_t<
        is_same<int_t, _tmp_type>::value, linear_index,
        conditional_t<is_same<_tmp_type, pointi<_Rank>>::value, array_index, void>>
        type;
};

}  // namespace internal

/**
 * @brief a tensor without memory defined by the shape and lambda(functor)
 * @tparam _Rank the rank of tensor
 * @tparam _Fun the functor type of tensor, should be Index -> Value pattern
 * @see tensor
 */
template <int_t _Rank, typename _Fun, typename _Layout = row_major_layout<_Rank>>
class lambda_tensor : public tensor_expression<lambda_tensor<_Rank, _Fun, _Layout>> {
    typedef function_traits<_Fun> function_traits_t;

   public:
    static const int_t rank = _Rank;
    typedef _Fun functor_type;
    typedef typename function_traits_t::result_type reference;
    /// the value type of lambdd_tensor, it's the result type of functor_type
    typedef remove_reference_t<reference> value_type;
    /**
     * @brief the access mode of lambdd_tensor, it's decided by the argument pattern.
     *
     * when the functor is int_t -> value pattern, the access mode is linear access.
     * when the functor is pointi<rank> -> value pattern, the access mode is array access.
     */
    typedef typename internal::get_functor_accessor_type<_Rank, _Fun>::type index_type;

    typedef _Layout layout_type;
    typedef host_t runtime_type;

   public:
    /**
     * @brief constructs a lambdd_tensor by the shape and fun
     * @param ext the shape of tensor
     * @param fun the functor of lambdd_tensor, should be Index -> Value pattern
     */
    lambda_tensor(const pointi<rank>& ext, _Fun fun) : shape_(ext), layout_(ext), functor_(fun) {}

    /**
     * @brief copy constructor
     */
    lambda_tensor(const lambda_tensor& rhs)
        : shape_(rhs.shape_), layout_(rhs.layout_), functor_(rhs.functor_) {}

    /**
     * @brief accesses element by linear access mode
     * @param i linear index
     * @return element referece
     */
    reference operator[](int_t i) const { return offset_imp<index_type>(i); }

    /**
     * @brief accesses element by array access mode
     * @param idx array index
     * @return element const reference
     */
    reference operator[](pointi<rank> idx) const { return index_imp<index_type>(idx); }

    /**
     * @brief accesses element by array access mode
     * @param idx array index
     * @return element const reference
     */
    reference operator()(pointi<rank> idx) const { return index_imp<index_type>(idx); }

    /**
     * @brief accesses element by array access mode
     * @param idx packed array index parameters
     * @return element const reference
     */
    template <typename... _Idx>
    reference operator()(_Idx... idx) const {
        return (*this)(pointi<rank>{idx...});
    }

    /// @return the shape of lambed_tensor
    pointi<rank> shape() const { return shape_; }

    int_t shape(int_t i) const { return shape_[i]; }

    /// return the total size of lambda_tensor elements
    int_t size() const { return layout_.size(); }

    /**
     * @brief perisits a lambdd_tensor to a tensor with memory
     * @param policy the execution policy
     * @return a tensor which copys elements value from lambdd_tensor
     */
    template <typename _ExecutionPolicy>
    MATAZURE_GENERAL tensor<decay_t<value_type>, rank> persist(_ExecutionPolicy policy) const {
        tensor<decay_t<value_type>, rank> re(this->shape());
        copy(policy, *this, re);
        return re;
    }

    /// persists a lambdd_tensor to a tensor by sequence policy
    MATAZURE_GENERAL tensor<decay_t<value_type>, rank> persist() const {
        sequence_policy policy{};
        return persist(policy);
    }

    layout_type layout() const { return layout_; }

    constexpr runtime_type runtime() const { return runtime_type{}; }

    functor_type functor() const { return functor_; }

   private:
    template <typename _Mode>
    enable_if_t<is_same<_Mode, array_index>::value, reference> index_imp(pointi<rank> idx) const {
        return functor_(idx);
    }

    template <typename _Mode>
    enable_if_t<is_same<_Mode, linear_index>::value, reference> index_imp(pointi<rank> idx) const {
        return (*this)[layout_.index2offset(idx)];
    }

    template <typename _Mode>
    enable_if_t<is_same<_Mode, array_index>::value, reference> offset_imp(int_t i) const {
        return (*this)[layout_.offset2index(i)];
    }

    template <typename _Mode>
    enable_if_t<is_same<_Mode, linear_index>::value, reference> offset_imp(int_t i) const {
        return functor_(i);
    }

   private:
    const pointi<rank> shape_;
    const layout_type layout_;
    const _Fun functor_;
};

/**
 * @brief make a lambda_tensor
 * @param the shape
 * @param the functor, a index -> value pattern
 */
template <int_t _Rank, typename _Fun>
inline auto make_lambda(pointi<_Rank> extent, _Fun fun) -> lambda_tensor<_Rank, _Fun> {
    return lambda_tensor<_Rank, _Fun>(extent, fun);
}

template <int_t _Rank, typename _Fun, typename _Layout>
inline auto make_lambda(pointi<_Rank> extent, _Fun fun, _Layout) -> lambda_tensor<_Rank, _Fun> {
    return lambda_tensor<_Rank, _Fun, _Layout>(extent, fun);
}

/**
 * @brief make a lambda_tensor
 * @param the shape
 * @param the functor, a index -> value pattern
 */
template <int_t _Rank, typename _Fun>
inline auto make_lambda(pointi<_Rank> extent, _Fun fun, host_t) -> lambda_tensor<_Rank, _Fun> {
    return lambda_tensor<_Rank, _Fun>(extent, fun);
}

template <int_t _Rank, typename _Fun, typename _Layout>
inline auto make_lambda(pointi<_Rank> extent, _Fun fun, host_t, _Layout)
    -> lambda_tensor<_Rank, _Fun, _Layout> {
    return lambda_tensor<_Rank, _Fun, _Layout>(extent, fun);
}

template <int_t _Rank, typename _Fun, typename _Layout>
inline auto make_lambda(pointi<_Rank> extent, _Fun fun, _Layout, host_t)
    -> lambda_tensor<_Rank, _Fun, _Layout> {
    return lambda_tensor<_Rank, _Fun, _Layout>(extent, fun);
}

}  // namespace matazure
