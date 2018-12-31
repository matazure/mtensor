#pragma once

#include <matazure/point.hpp>
#include <matazure/execution.hpp>
#include <matazure/type_traits.hpp>

#define MATAZURE_STRINGIFY(a) #a

#if defined(_MSC_VER)
	#define MATAZURE_AUTO_VECTORISED __pragma(loop(ivdep))
#elif defined(__GNUC__)
	///@todo： auto vectorized does not support gcc now, to fix it
	#define MATAZURE_AUTO_VECTORISED /*_Pragma("ivdep")*/
#else
	#define MATAZURE_AUTO_VECTORISED _Pragma("ivdep")
#endif

#ifdef MATAZURE_OPENMP
	#if defined(_MSC_VER)
		#define MATAZURE_OPENMP_PARALLEL_FOR(n) __pragma(omp parallel for)
	#else
		#if _OPENMP >= 200805
			// #define PRIVATE_MATAZURE_PARALLEL_FOR(n) "omp parallel for schedule(dynamic, 1) collapse(" #n ")"
		 	#define MATAZURE_OPENMP_PARALLEL_FOR(n) _Pragma(MATAZURE_STRINGIFY(omp parallel for schedule(dynamic, 1) collapse(n)))
		#else
		 	#define MATAZURE_OPENMP_PARALLEL_FOR(n) _Pragma("omp parallel for")
		#endif
	#endif

	#if defined(_MSC_VER)
		#define MATAZURE_OPENMP_PARALLEL_FOR_VECTORISED(n) \
		 	__pragma(omp parallel for)
	#else
		 #if _OPENMP >= 200805
		 	#define MATAZURE_OPENMP_PARALLEL_FOR_VECTORISED(n) _Pragma(MATAZURE_STRINGIFY(omp parallel for schedule(dynamic, 1) collapse(n)))
		 #else
		 	#define MATAZURE_OPENMP_PARALLEL_FOR_VECTORISED(n) _Pragma("omp parallel for simd")
		#endif
	#endif
#endif

namespace matazure {

/// special is_linear_array for point
template <typename _Type, int_t _Rank>
struct _Is_linear_array<point<_Type, _Rank>>: bool_constant<true> {};

/**
* @brief for each linear index, apply fun by the sequence policy
* @param first the first index
* @param last the last index
* @param fun the functor,  int_t -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(sequence_policy, int_t first, int_t last, _Func fun) {
	for (int_t i = first; i < last; ++i) {
		fun(i);
	}
}

/**
* @brief for each linear index, apply fun by the sequence vectorized policy
* @param first the first index
* @param last the last index
* @param fun the functor,  int_t -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(sequence_vectorized_policy policy, int_t first, int_t last, _Func fun) {
	MATAZURE_AUTO_VECTORISED
	for (int_t i = first; i < last; ++i) {
		fun(i);
	}
}

#ifdef MATAZURE_OPENMP

/**
* @brief for each linear index, apply fun by the openmp parallel policy
* @param first the first index
* @param last the last index
* @param fun the functor,  int_t -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(omp_policy policy, int_t first, int_t last, _Func fun){
	MATAZURE_OPENMP_PARALLEL_FOR(1)
	for (int_t i = first; i < last; ++i) {
		fun(i);
	}
}

/**
* @brief for each linear index, apply fun by the openmp parallel vectorized policy
* @param first the first index
* @param last the last index
* @param fun the functor,  int_t -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(omp_vectorized_policy policy, int_t first, int_t last, _Func fun){
	#pragma omp parallel for
	for (int_t i = first; i < last; ++i) {
		fun(i);
	}
}

#endif

/**
* @brief for each linear index, apply fun by the sequence policy
* @param first the first index
* @param last the last index
* @param fun the functor,  int_t -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(int_t first, int_t last, _Func fun) {
	sequence_policy seq{};
	for_index(seq, first, last, fun);
}

/**

*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(int_t last, _Func fun){
	for_index(0, last, fun);
}

/**
* @brief for each 1-dim array index, apply fun by the sequence policy
* @param origin the origin index of the 1-dim range
* @param end the end index
* @param fun the functor,  pointi<1> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(sequence_policy, pointi<1> origin, pointi<1> end, _Func fun) {
	for (int_t i = origin[0]; i < end[0]; ++i) {
		fun(pointi<1>{ { i } });
	}
}

/**
* @brief for each 2-dim array index, apply fun by the sequence policy
* @param origin the origin index of the 2-dim range
* @param end the end index
* @param fun the functor,  pointi<2> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(sequence_policy, pointi<2> origin, pointi<2> end, _Func fun) {
	for (int_t j = origin[1]; j < end[1]; ++j) {
		for (int_t i = origin[0]; i < end[0]; ++i) {
			fun(pointi<2>{ { i, j } });
		}
	}
}

/**
* @brief for each 3-dim array index, apply fun by the sequence policy
* @param origin the origin index of the 3-dim range
* @param end the end index
* @param fun the functor,  pointi<3> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(sequence_policy, pointi<3> origin, pointi<3> end, _Func fun) {
	for (int_t k = origin[2]; k < end[2]; ++k) {
		for (int_t j = origin[1]; j < end[1]; ++j) {
			for (int_t i = origin[0]; i < end[0]; ++i) {
				fun(pointi<3>{ { i, j, k } });
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
template <typename _Func>
inline MATAZURE_GENERAL void for_index(sequence_policy, pointi<4> origin, pointi<4> end, _Func fun) {
	for (int_t l = origin[3]; l < end[3]; ++l) {
		for (int_t k = origin[2]; k < end[2]; ++k) {
			for (int_t j = origin[1]; j < end[1]; ++j) {
				for (int_t i = origin[0]; i < end[0]; ++i) {
					fun(pointi<4>{ {i, j, k, l} });
				}
			}
		}
	}
}

/**
* @brief for each 1-dim array index, apply fun by the sequence vectorized policy
* @param origin the origin index of the 1-dim range
* @param end the end index
* @param fun the functor,  pointi<1> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(sequence_vectorized_policy, pointi<1> origin, pointi<1> end, _Func fun) {
	MATAZURE_AUTO_VECTORISED
	for (int_t i = origin[0]; i < end[0]; ++i) {
		fun(pointi<1>{ { i } });
	}
}

/**
* @brief for each 2-dim array index, apply fun by the sequence vectorized policy
* @param origin the origin index of the 2-dim range
* @param end the end index
* @param fun the functor,  pointi<2> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(sequence_vectorized_policy, pointi<2> origin, pointi<2> end, _Func fun) {
	for (int_t j = origin[1]; j < end[1]; ++j) {
		MATAZURE_AUTO_VECTORISED
		for (int_t i = origin[0]; i < end[0]; ++i) {
			fun(pointi<2>{ { i, j } });
		}
	}
}

/**
* @brief for each 3-dim array index, apply fun by the sequence vectorized policy
* @param origin the origin index of the 3-dim range
* @param end the end index
* @param fun the functor,  pointi<3> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(sequence_vectorized_policy, pointi<3> origin, pointi<3> end, _Func fun) {
	for (int_t k = origin[2]; k < end[2]; ++k) {
		for (int_t j = origin[1]; j < end[1]; ++j) {
			MATAZURE_AUTO_VECTORISED
			for (int_t i = origin[0]; i < end[0]; ++i) {
				fun(pointi<3>{ { i, j, k } });
			}
		}
	}
}

/**
* @brief for each 4-dim array index, apply fun by the sequence vectorized policy
* @param origin the origin index of the 4-dim range
* @param end the end index
* @param fun the functor,  pointi<4> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(sequence_vectorized_policy, pointi<4> origin, pointi<4> end, _Func fun) {
	for (int_t l = origin[3]; l < end[3]; ++l) {
		for (int_t k = origin[2]; k < end[2]; ++k) {
			for (int_t j = origin[1]; j < end[1]; ++j) {
				MATAZURE_AUTO_VECTORISED
				for (int_t i = origin[0]; i < end[0]; ++i) {
					fun(pointi<4>{ {i, j, k, l} });
				}
			}
		}
	}
}

#ifdef _OPENMP

/**
* @brief for each 1-dim array index, apply fun by the openmp parallel policy
* @param origin the origin index of the 1-dim range
* @param end the end index
* @param fun the functor,  pointi<1> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(omp_policy, pointi<1> origin, pointi<1> end, _Func fun) {
	MATAZURE_OPENMP_PARALLEL_FOR(1)
	for (int_t i = origin[0]; i < end[0]; ++i) {
		fun(pointi<1>{ { i } });
	}
}

/**
* @brief for each 2-dim array index, apply fun by the openmp parallel policy
* @param origin the origin index of the 2-dim range
* @param end the end index
* @param fun the functor,  pointi<2> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(omp_policy, pointi<2> origin, pointi<2> end, _Func fun) {
	MATAZURE_OPENMP_PARALLEL_FOR(2)
	for (int_t j = origin[1]; j < end[1]; ++j) {
		for (int_t i = origin[0]; i < end[0]; ++i) {
			fun(pointi<2>{ { i, j } });
		}
	}
}

/**
* @brief for each 3-dim array index, apply fun by the openmp parallel policy
* @param origin the origin index of the 3-dim range
* @param end the end index
* @param fun the functor,  pointi<3> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(omp_policy, pointi<3> origin, pointi<3> end, _Func fun) {
	MATAZURE_OPENMP_PARALLEL_FOR(3)
	for (int_t k = origin[2]; k < end[2]; ++k) {
		for (int_t j = origin[1]; j < end[1]; ++j) {
			for (int_t i = origin[0]; i < end[0]; ++i) {
				fun(pointi<3>{ { i, j, k } });
			}
		}
	}
}

/**
* @brief for each 4-dim array index, apply fun by the openmp parallel policy
* @param origin the origin index of the 4-dim range
* @param end the end index
* @param fun the functor,  pointi<4> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(omp_policy, pointi<4> origin, pointi<4> end, _Func fun) {
	MATAZURE_OPENMP_PARALLEL_FOR(4)
	for (int_t l = origin[3]; l < end[3]; ++l) {
		for (int_t k = origin[2]; k < end[2]; ++k) {
			for (int_t j = origin[1]; j < end[1]; ++j) {
				for (int_t i = origin[0]; i < end[0]; ++i) {
					fun(pointi<4>{ {i, j, k, l} });
				}
			}
		}
	}
}

/**
* @brief for each 1-dim array index, apply fun by the openmp parallel vectorized policy
* @param origin the origin index of the 1-dim range
* @param end the end index
* @param fun the functor,  pointi<1> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(omp_vectorized_policy, pointi<1> origin, pointi<1> end, _Func fun) {
	MATAZURE_OPENMP_PARALLEL_FOR_VECTORISED(1)
	for (int_t i = origin[0]; i < end[0]; ++i) {
		fun(pointi<1>{ { i } });
	}
}

/**
* @brief for each 2-dim array index, apply fun by the openmp parallel vectorized policy
* @param origin the origin index of the 2-dim range
* @param end the end index
* @param fun the functor,  pointi<2> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(omp_vectorized_policy, pointi<2> origin, pointi<2> end, _Func fun) {
	MATAZURE_OPENMP_PARALLEL_FOR_VECTORISED(2)
	for (int_t j = origin[1]; j < end[1]; ++j) {
		for (int_t i = origin[0]; i < end[0]; ++i) {
			fun(pointi<2>{ { i, j } });
		}
	}
}

/**
* @brief for each 3-dim array index, apply fun by the openmp parallel vectorized policy
* @param origin the origin index of the 3-dim range
* @param end the end index
* @param fun the functor,  pointi<3> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(omp_vectorized_policy, pointi<3> origin, pointi<3> end, _Func fun) {
	MATAZURE_OPENMP_PARALLEL_FOR_VECTORISED(3)
	for (int_t k = origin[2]; k < end[2]; ++k) {
		for (int_t j = origin[1]; j < end[1]; ++j) {
			for (int_t i = origin[0]; i < end[0]; ++i) {
				fun(pointi<3>{ { i, j, k } });
			}
		}
	}
}

/**
* @brief for each 4-dim array index, apply fun by the openmp parallel vectorized policy
* @param origin the origin index of the 4-dim range
* @param end the end index
* @param fun the functor,  pointi<4> -> value pattern.
*/
template <typename _Func>
inline MATAZURE_GENERAL void for_index(omp_vectorized_policy, pointi<4> origin, pointi<4> end, _Func fun) {
	MATAZURE_OPENMP_PARALLEL_FOR_VECTORISED(4)
	for (int_t l = origin[3]; l < end[3]; ++l) {
		for (int_t k = origin[2]; k < end[2]; ++k) {
			for (int_t j = origin[1]; j < end[1]; ++j) {
				for (int_t i = origin[0]; i < end[0]; ++i) {
					fun(pointi<4>{ {i, j, k, l} });
				}
			}
		}
	}
}

#endif

/**
* @brief for each array index, apply fun by the sequence policy
* @param origin the origin index of the range
* @param end the end index
* @param fun the functor,  pointi -> value pattern.
*/
template <typename _Func, int_t _Rank>
inline MATAZURE_GENERAL void for_index(pointi<_Rank> origin, pointi<_Rank> end, _Func fun) {
	sequence_vectorized_policy policy{};
	for_index(policy, origin, end, fun);
}

template <typename _Func, int_t _Rank>
inline MATAZURE_GENERAL void for_index(pointi<_Rank> end, _Func fun){
	for_index(zero<pointi<_Rank>>::value(), end, fun);
}

/**
* @brief for each element of a linear indexing tensor, apply fun
* @param policy the execution policy
* @param ts the source tensor
* @param fun the functor, (element &) -> none pattern
*/
template <typename _ExectutionPolicy, typename _Tensor, typename _Fun>
inline MATAZURE_GENERAL void for_each(_ExectutionPolicy policy, _Tensor &&ts, _Fun fun, enable_if_t<are_linear_access<decay_t<_Tensor>>::value && none_device_memory<decay_t<_Tensor>>::value>* = 0) {
	for_index(policy, 0, ts.size(), [&](int_t i) {
		fun(ts[i]);
	});
}

/**
* @brief for each element of an array indexing tensor, apply fun
* @param policy the execution policy
* @param ts the source tensor
* @param fun the functor, (element &) -> none pattern
*/
template <typename _ExectutionPolicy, typename _Tensor, typename _Fun>
inline MATAZURE_GENERAL void for_each(_ExectutionPolicy policy, _Tensor &&ts, _Fun fun, enable_if_t<!are_linear_access<decay_t<_Tensor>>::value && none_device_memory<decay_t<_Tensor>>::value>* = 0) {
	for_index(policy, pointi<decay_t<_Tensor>::rank>::zeros(), ts.shape(), [&](pointi<decay_t<_Tensor>::rank> idx) {
		fun(ts[idx]);
	});
}

/**
* @brief for each element of a tensor, apply fun by the sequence policy
* @param ts the source tensor
* @param fun the functor, (element &) -> none pattern
*/
template <typename _Tensor, typename _Fun>
inline MATAZURE_GENERAL void for_each(_Tensor &&ts, _Fun fun,
	enable_if_t<
		none_device_memory<
			enable_if_t<
				is_linear_array<decay_t<_Tensor>>::value,
				decay_t<_Tensor>
			>
		>::value
	>* = 0)
{
	sequence_vectorized_policy policy{};
	for_each(policy, std::forward<_Tensor>(ts), fun);
}

/**
* @brief fill a tensor value elementwise
* @param policy the execution policy
* @param ts the source tensor
* @param v the filled value
*/
template <typename _ExectutionPolicy, typename _Tensor>
inline MATAZURE_GENERAL void fill(_ExectutionPolicy policy, _Tensor &&ts, typename decay_t<_Tensor>::value_type v, enable_if_t<none_device_memory<decay_t<_Tensor>>::value>* = 0) {
	for_each(policy, std::forward<_Tensor>(ts), [v](typename decay_t<_Tensor>::value_type &x) { x = v;});
}

/**
* @brief fill a tensor value elementwise by the sequence policy
* @param ts the source tensor
* @param v the filled value
*/
template <typename _Tensor>
inline MATAZURE_GENERAL void fill(_Tensor &&ts, typename decay_t<_Tensor>::value_type v, enable_if_t<
		none_device_memory<
			enable_if_t<
				is_linear_array<decay_t<_Tensor>>::value,
				decay_t<_Tensor>
			>
		>::value
	>* = 0) {
	sequence_vectorized_policy policy{};
	fill(policy, std::forward<_Tensor>(ts), v);
}

/**
* @brief elementwisely copy a linear indexing tensor to another one
* @param policy the execution policy
* @param ts_src the source tensor
* @param ts_dst the dest tensor
*/
template <typename _ExectutionPolicy, typename _TensorSrc, typename _TensorDst>
inline MATAZURE_GENERAL void copy(_ExectutionPolicy policy, const _TensorSrc &ts_src, _TensorDst &&ts_dst, enable_if_t<are_linear_access<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value && none_device_memory<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value>* = 0) {
	for_index(policy, 0, ts_src.size(), [&] (int_t i) {
		ts_dst[i] = ts_src[i];
	});
}

/**
* @brief elementwisely copy a array indexing tensor to another one
* @param policy the execution policy
* @param ts_src the source tensor
* @param ts_dst the dest tensor
*/
template <typename _ExectutionPolicy, typename _TensorSrc, typename _TensorDst>
inline MATAZURE_GENERAL void copy(_ExectutionPolicy policy, const _TensorSrc &ts_src, _TensorDst &&ts_dst, enable_if_t<!are_linear_access<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value && none_device_memory<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value>* = 0) {
	for_index(policy, pointi<_TensorSrc::rank>::zeros(), ts_src.shape(), [&] (pointi<_TensorSrc::rank> idx) {
		ts_dst[idx] = ts_src[idx];
	});
}

/**
* @brief elementwisely copy a tensor to another one by the sequence policy
* @param ts_src the source tensor
* @param ts_dst the dest tensor
*/
template <typename _TensorSrc, typename _TensorDst>
inline MATAZURE_GENERAL void copy(const _TensorSrc &ts_src, _TensorDst &&ts_dst, enable_if_t<none_device_memory<enable_if_t<is_linear_array<decay_t<_TensorSrc>>::value, decay_t<_TensorSrc>>, decay_t<_TensorDst>>::value>* = 0) {
	sequence_vectorized_policy policy;
	copy(policy, ts_src, std::forward<_TensorDst>(ts_dst));
}

/**
* @brief transform a linear indexing tensor to another by the fun
* @param policy the execution policy
* @param ts_src the source tensor
* @param ts_dst the destination tensor
* @param fun the functor, (e_src) -> e_dst pattern
*/
template <typename _ExectutionPolicy, typename _TensorSrc, typename _TensorDst, typename _Fun>
inline MATAZURE_GENERAL void transform(_ExectutionPolicy policy, const _TensorSrc &ts_src, _TensorDst &&ts_dst, _Fun fun, enable_if_t<are_linear_access<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value && none_device_memory<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value>* = 0) {
	for_index(policy, 0, ts_src.size(), [&](int_t i) {
		ts_dst[i] = fun(ts_src[i]);
	});
}

/**
* @brief transform a array indexing tensor to another by the fun
* @param policy the execution policy
* @param ts_src the source tensor
* @param ts_dst the destination tensor
* @param fun the functor, (e_src) -> e_dst pattern
*/
template <typename _ExectutionPolicy, typename _TensorSrc, typename _TensorDst, typename _Fun>
inline MATAZURE_GENERAL void transform(_ExectutionPolicy policy, const _TensorSrc &ts_src, _TensorDst &&ts_dst, _Fun fun, enable_if_t<!are_linear_access<decay_t<_TensorSrc>>::value && none_device_memory<decay_t<_TensorSrc>>::value>* = 0) {
	for_index(policy, pointi<_TensorSrc::rank>::zeros(), ts_src.shape(), [&] (pointi<_TensorSrc::rank> idx) {
		ts_dst[idx] = fun(ts_src[idx]);
	});
}

/**
* @brief transform a tensor to another by the fun
* @param ts_src the source tensor
* @param ts_dst the destination tensor
* @param fun the functor, (e_src) -> e_dst pattern
*/
template <typename _TensorSrc, typename _TensorDst, typename _Fun>
inline MATAZURE_GENERAL void transform(const _TensorSrc &ts_src, _TensorDst &&ts_dst, _Fun fun, enable_if_t<none_device_memory<enable_if_t<is_linear_array<decay_t<_TensorSrc>>::value, decay_t<_TensorSrc>>, decay_t<_TensorDst>>::value>* = 0)
{
	sequence_vectorized_policy policy{};
	transform(policy, ts_src, std::forward<_TensorDst>(ts_dst), fun);
}

/**
* @brief reduce the elements of a tensor
* @param the execution policy
* @param ts the source tensor
* @param init the initial value
* @param binary_fun the reduce functor, must be (element, element)-> value pattern
*/
///@todo
// template <typename _ExectutionPolicy, typename _Tensor, typename _VT, typename _BinaryFunc>
// inline MATAZURE_GENERAL _VT reduce(_ExectutionPolicy policy, _Tensor ts, _VT init, _BinaryFunc binary_fun, enable_if_t<none_device_memory<decay_t<_Tensor>>::value>* = 0) {
// 	auto re = init;
// 	for_each(policy, ts, [&re, binary_fun](decltype(ts[0]) x) {
// 		re = binary_fun(re, x);
// 	});
//
// 	return re;
// }

/**
* @brief reduce the elements of a tensor by the sequence policy
* @param ts the source tensor
* @param init the initial value
* @param binary_fun the reduce functor, must be (element, element)-> value pattern
*/
template <typename _Tensor, typename _VT, typename _BinaryFunc>
inline MATAZURE_GENERAL _VT reduce(const _Tensor &ts, _VT init, _BinaryFunc binary_fun) {
	sequence_policy policy{};
	auto re = init;
	for_each(policy, ts, [&re, binary_fun] (decltype(ts[0]) x) {
		re = binary_fun(re, x);
	});

	return re;
}

template <typename _Tensor>
inline MATAZURE_GENERAL auto sum(const _Tensor &ts) {
	return reduce(ts, zero<typename _Tensor::value_type>::value(), [](auto lhs, auto rhs) {
		return lhs + rhs;
	});
}

}
