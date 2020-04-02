template <typename _Tensor, typename _Func>
struct linear_map_op {
   private:
    const _Tensor ts_;
    const _Func functor_;

   public:
    linear_map_op(_Tensor ts, _Func fun) : ts_(ts), functor_(fun) {}

    MATAZURE_GENERAL auto operator()(int_t i) const -> decltype(this->functor_(this->ts_[i])) {
        return functor_(ts_[i]);
    }
};

template <typename _Tensor, typename _Func>
struct array_map_op {
   private:
    const _Tensor ts_;
    const _Func functor_;

   public:
    array_map_op(_Tensor ts, _Func fun) : ts_(ts), functor_(fun) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const
        -> decltype(this->functor_(this->ts_[idx])) {
        return functor_(ts_[idx]);
    }
};

/**
 * @brief map the functor for each element of a linear indexing tensor
 * @param ts the source tensor
 * @param fun the functor, element -> value  pattern
 */
template <typename _Tensor, typename _Func>
inline auto map(_Tensor ts, _Func fun,
                enable_if_t<is_same<linear_index, typename _Tensor::index_type>::value>* = 0)
    -> decltype(make_lambda(ts.shape(), linear_map_op<_Tensor, _Func>(ts, fun),
                            typename _Tensor::memory_type{})) {
    return make_lambda(ts.shape(), linear_map_op<_Tensor, _Func>(ts, fun),
                       typename _Tensor::memory_type{});
}

/**
 * @brief map the functor for each element of a array indexing tensor
 * @param ts the source tensor
 * @param fun the functor, element -> value  pattern
 */
template <typename _Tensor, typename _Func>
inline auto map(_Tensor ts, _Func fun,
                enable_if_t<is_same<array_index, typename _Tensor::index_type>::value>* = 0)
    -> decltype(make_lambda(ts.shape(), array_map_op<_Tensor, _Func>(ts, fun),
                            typename _Tensor::memory_type{})) {
    return make_lambda(ts.shape(), array_map_op<_Tensor, _Func>(ts, fun),
                       typename _Tensor::memory_type{});
}
