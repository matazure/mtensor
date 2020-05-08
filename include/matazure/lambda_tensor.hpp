
#pragma once

#include <matazure/layout.hpp>
#include <matazure/local_tensor.hpp>
#include <matazure/tensor.hpp>
#include <matazure/type_traits.hpp>

namespace matazure {

namespace internal {

template <int_t _Rank, typename _Func>
struct get_functor_accessor_type {
   private:
    typedef function_traits<_Func> functor_traits;
    static_assert(functor_traits::arguments_size == 1, "functor must be unary");
    typedef decay_t<typename functor_traits::template arguments<0>::type> _tmp_type;

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
 * @tparam _Func the functor type of tensor, should be Index -> Value pattern
 * @see tensor
 */
template <int_t _Rank, typename _Func, typename _Layout = column_major_layout<_Rank>>
class lambda_tensor : public tensor_expression<lambda_tensor<_Rank, _Func, _Layout>> {
    typedef function_traits<_Func> functor_traits;

   public:
    static const int_t rank = _Rank;
    typedef _Func functor_type;
    typedef typename functor_traits::result_type reference;
    /// the value type of lambdd_tensor, it's the result type of functor_type
    typedef remove_reference_t<reference> value_type;
    /**
     * @brief the access mode of lambdd_tensor, it's decided by the argument pattern.
     *
     * when the functor is int_t -> value pattern, the access mode is linear access.
     * when the functor is pointi<rank> -> value pattern, the access mode is array access.
     */
    typedef typename internal::get_functor_accessor_type<_Rank, _Func>::type index_type;

    typedef _Layout layout_type;
    typedef host_tag memory_type;

   public:
    /**
     * @brief constructs a lambdd_tensor by the shape and fun
     * @param ext the shape of tensor
     * @param fun the functor of lambdd_tensor, should be Index -> Value pattern
     */
    lambda_tensor(const pointi<rank>& ext, _Func fun) : shape_(ext), layout_(ext), functor_(fun) {}

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
        return (*this)[pointi<rank>{idx...}];
    }

    /// prevents operator() const matching with pointi<rank> argument
    template <typename _Idx>
    reference operator()(_Idx idx) const {
        static_assert(std::is_same<_Idx, int_t>::value && rank == 1,
                      "only operator [] support access data by pointi");
        return (*this)[pointi<1>{idx}];
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
    const _Func functor_;
};

/**
 * @brief make a lambda_tensor
 * @param the shape
 * @param the functor, a index -> value pattern
 */
template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> extent, _Func fun) -> lambda_tensor<_Rank, _Func> {
    return lambda_tensor<_Rank, _Func>(extent, fun);
}

/**
 * @brief make a lambda_tensor
 * @param the shape
 * @param the functor, a index -> value pattern
 */
template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> extent, _Func fun, host_tag) -> lambda_tensor<_Rank, _Func> {
    return lambda_tensor<_Rank, _Func>(extent, fun);
}

}  // namespace matazure
