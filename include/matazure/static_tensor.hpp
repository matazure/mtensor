#pragma once

#include <matazure/meta.hpp>

namespace matazure {

	/**
	* @brief convert array index to linear index by first marjor
	* @param id array index
	* @param stride tensor stride
	* @param first_major
	* @return linear index
	*/
	template <int_t _Rank>
	inline MATAZURE_GENERAL typename pointi<_Rank>::value_type index2offset(const pointi<_Rank> &id, const pointi<_Rank> &stride, first_major) {
		typename pointi<_Rank>::value_type offset = id[0];
		for (int_t i = 1; i < _Rank; ++i) {
			offset += id[i] * stride[i - 1];
		}

		return offset;
	};

	/**
	* @brief convert array index to linear index by first marjor
	* @param offset linear index
	* @param stride the stride of tensor
	* @param first_major
	* @return array index
	*/
	template <int_t _Rank>
	inline MATAZURE_GENERAL pointi<_Rank> offset2index(typename pointi<_Rank>::value_type offset, const pointi<_Rank> &stride, first_major) {
		pointi<_Rank> id;
		for (int_t i = _Rank - 1; i > 0; --i) {
			id[i] = offset / stride[i - 1];
			offset = offset % stride[i - 1];
		}
		id[0] = offset;

		return id;
	}

	template <int_t... _Values>
	using dim = meta::array<_Values ...>;

	/**
	* @brief a compile time tensor which uses static memory
	* @tparam _ValueType element value type, must be pod
	* @tparam _Ext a fixed shape, must be dim(meta::array) type
	*/
	template <typename _ValueType, typename _Ext>
	class static_tensor {
	private:
		template <int_t ..._Values>
		struct traits;

		template <int_t _S0>
		struct traits<_S0> {
			MATAZURE_GENERAL static constexpr int_t size() {
				return _S0;
			}

			MATAZURE_GENERAL static constexpr pointi<1> stride() {
				return pointi<1>{ { _S0 } };
			}
		};

		template <int_t _S0, int_t _S1>
		struct traits<_S0, _S1> {
			MATAZURE_GENERAL static constexpr int_t size() {
				return _S0 * _S1;
			}

			MATAZURE_GENERAL static constexpr pointi<2> stride() {
				return{ { _S0, _S0 * _S1 } };
			}
		};

		template <int_t _S0, int_t _S1, int_t _S2>
		struct traits<_S0, _S1, _S2> {
			MATAZURE_GENERAL static constexpr int_t size() {
				return _S0 * _S1 * _S2;
			}

			MATAZURE_GENERAL static constexpr pointi<3> stride() {
				return{ { _S0, _S0 * _S1, _S0 * _S1 * _S2 } };
			}
		};

		template <int_t _S0, int_t _S1, int_t _S2, int_t _S3>
		struct traits<_S0, _S1, _S2, _S3> {
			MATAZURE_GENERAL static constexpr int_t size() {
				return _S0 * _S1 * _S2 * _S3;
			}

			MATAZURE_GENERAL static constexpr pointi<4> stride() {
				return{ { _S0, _S0 * _S1, _S0 * _S1 * _S2, _S0 * _S1 * _S2 * _S3 } };
			}
		};

		template <typename _T>
		struct traits_helper;

		template <int_t ..._Values>
		struct traits_helper<dim<_Values...>> {
			typedef traits<_Values...> type;
		};

		typedef typename traits_helper<_Ext>::type traits_t;

		/// @todo should check each dim
		static_assert(traits_t::size() > 0, "");

	public:
		/// the meta shape type which has compile time ext
		typedef _Ext					meta_shape_type;
		static	const int_t				rank = meta_shape_type::size();
		typedef _ValueType				value_type;
		typedef value_type *			pointer;
		typedef const pointer			const_pointer;
		typedef value_type &			reference;
		typedef const value_type &		const_reference;
		typedef linear_index			index_type;
		typedef local_tag				memory_type;

		/**
		* @brief accesses element by linear access mode
		* @param i linear index
		* @return element const referece
		*/
		MATAZURE_GENERAL reference operator[](int_t i) {
			 return elements_[i];
		}

		/**
		* @brief accesses element by linear access mode
		* @param i linear index
		* @return element referece
		*/
		MATAZURE_GENERAL constexpr const_reference operator[](int_t i) const {
			return elements_[i];
		}

		/**
		* @brief accesses element by array access mode
		* @param idx array index
		* @return element const reference
		*/
		MATAZURE_GENERAL constexpr const_reference operator()(const pointi<rank> &idx) const {
			return (*this)[index2offset(idx, stride(), first_major{})];
		}

		/**
		* @brief accesses element by array access mode
		* @param idx array index
		* @return element reference
		*/
		MATAZURE_GENERAL reference operator()(const pointi<rank> &idx) {
			return (*this)[index2offset(idx, stride(), first_major{})];
		}

		/**
		* @brief accesses element by array access mode
		* @param idx packed array index parameters
		* @return element const reference
		*/
		template <typename ..._Idx>
		MATAZURE_GENERAL reference operator()(_Idx... idx) {
			return (*this)(pointi<rank>{ idx... });
		}


		/**
		* @brief accesses element by array access mode
		* @param idx packed array index parameters
		* @return element reference
		*/
		template <typename ..._Idx>
		MATAZURE_GENERAL constexpr const_reference operator()(_Idx... idx) const {
			return (*this)(pointi<rank>{ idx... });
		}


		/// @return the meta shape instance of tensor
		MATAZURE_GENERAL static constexpr meta_shape_type meta_shape() {
			return meta_shape_type();
		}

		/// @return the shape of tensor
		MATAZURE_GENERAL constexpr pointi<rank> shape() const {
			return meta_shape_type::value();
		}

		/// @return the stride of tensor
		MATAZURE_GENERAL constexpr pointi<rank> stride() const {
			return traits_t::stride();
		}

		/// return the total size of tensor elements
		MATAZURE_GENERAL constexpr int_t size() const {
			return traits_t::size();
		}

		/// return the const pointer of tensor elements
		MATAZURE_GENERAL const_pointer data() const {
			return elements_;
		}

		/// return the pointer of tensor elements
		MATAZURE_GENERAL pointer data() {
			return elements_;
		}

		MATAZURE_GENERAL  constexpr int_t element_size() const {
			return sizeof(value_type);
		}

	public:
		value_type			elements_[traits_t::size()];
	};

	static_assert(std::is_pod<static_tensor<float, dim<3,3>>>::value, "static_tensor should be pod type");

	/// alias of static_tensor<_ValueType, 2>
	template <typename _ValueType, typename _Ext, typename _Tmp = enable_if_t<_Ext::size() == 2>>
	using static_matrix = static_tensor<_ValueType, _Ext>;

	/// alias of static_tensor<_ValueType, 1>
	template <typename _ValueType, typename _Ext, typename _Tmp = enable_if_t<_Ext::size() == 1>>
	using static_vector = static_tensor<_ValueType, _Ext>;

	/// special for static_tensor
	template <typename _ValueType, typename _Ext>
	struct is_tensor<static_tensor<_ValueType, _Ext>> : bool_constant<true> {};

}
