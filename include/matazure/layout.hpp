#pragma once

#include <matazure/config.hpp>

namespace matazure {

	/**
	* \defgroup Tensor Memory Layout
	* @{
	*/

	template <int_t _Rank>
	class first_major_layout{
	public:
		const static int_t rank = _Rank;

		#pragma hd_warning_disable
		MATAZURE_GENERAL first_major_layout(const pointi<rank> &shape) :
			shape_(shape)
		{
			stride_[0] = shape[0];
			for (int_t i = 1; i < rank; ++i) {
				stride_[i] = shape[i] * stride_[i - 1];
			}
		}

		MATAZURE_GENERAL first_major_layout(const first_major_layout &rhs) :
			first_major_layout(rhs.shape())
		{ }

		#pragma hd_warning_disable
		MATAZURE_GENERAL first_major_layout & operator=(const first_major_layout &rhs) {
			shape_ = rhs.shape();
			stride_ = rhs.stride();
			return *this;
		}

		MATAZURE_GENERAL int_t index2offset(const pointi<rank> &id) const {
			int_t offset = id[0];
			for (int_t i = 1; i < rank; ++i) {
				offset += id[i] * stride_[i - 1];
			}

			return offset;
		};

		MATAZURE_GENERAL pointi<rank> offset2index(int_t offset) const {
			pointi<rank> id{};
			for (int_t i = rank - 1; i > 0; --i) {
				id[i] = offset / stride_[i - 1];
				offset = offset % stride_[i - 1];
			}
			id[0] = offset;

			return id;
		}

		MATAZURE_GENERAL pointi<rank> shape() const{
			return shape_;
		}

		MATAZURE_GENERAL pointi<rank> stride() const{
			return stride_;
		}

		MATAZURE_GENERAL ~first_major_layout() { }

	private:
		pointi<rank> shape_;
		pointi<rank> stride_;
	};

	// template <int_t _Rank>
	// class last_major_layout{
	// public:
	// 	const static int_t rank = _Rank;

	// 	last_major_layout(const pointi<rank> &shape) :
	// 		shape_(shape)
	// 	{
	// 		stride_[0] = shape[0];
	// 		for (int_t i = 1; i < rank; ++i) {
	// 			stride_[i] = shape[i] * stride_[i - 1];
	// 		}
	// 	}

	// 	MATAZURE_GENERAL last_major_layout(const last_major_layout &rhs) :
	// 		last_major_layout(rhs.shape())
	// 	{ }

	// 	#pragma hd_warning_disable
	// 	MATAZURE_GENERAL last_major_layout & operator=(const last_major_layout &rhs) {
	// 		shape_ = rhs.shape();
	// 		stride_ = rhs.stride();
	// 		return *this;
	// 	}

	// 	MATAZURE_GENERAL int_t index2offset(const pointi<rank> &id) const {
	// 		typename pointi<rank>::value_type offset = id[rank - 1];
	// 		for (int_t i = 1; i < rank; ++i) {
	// 			offset += id[rank - 1 - i] * stride_[i - 1];
	// 		}

	// 		return offset;
	// 	};

	// 	MATAZURE_GENERAL pointi<rank> offset2index(int_t offset) const {
	// 		pointi<rank> id{};
	// 		for (int_t i = rank - 1; i > 0; --i) {
	// 			id[rank - 1 - i] = offset / stride_[i - 1];
	// 			offset = offset % stride_[i - 1];
	// 		}
	// 		id[rank - 1] = offset;

	// 		return id;
	// 	}

	// 	pointi<rank> shape() const{
	// 		return shape_;
	// 	}

	// 	pointi<rank> stride() const{
	// 		return stride_;
	// 	}

	// private:
	// 	pointi<rank> shape_;
	// 	pointi<rank> stride_;
	// };


	template <int_t _Rank>
	class padding_layout{
	public:
		const static int_t rank = _Rank;

		/**
		 * @brief
		 * @param shape
		 * @param step each memory dim length size
		 * @param padding each memory dim padding from origin
		 */
		#pragma hd_warning_disable
		MATAZURE_GENERAL padding_layout(const pointi<rank> &shape, const pointi<rank> & origin_padding, const pointi<rank> & end_padding) :
			shape_(shape),
			origin_padding_(origin_padding),
			end_padding_(end_padding)
		{
			auto step = shape + origin_padding + end_padding_;
			stride_[0] = step[0];
			for (int_t i = 1; i < rank; ++i) {
				stride_[i] = step[i] * stride_[i - 1];
			}
		}

		MATAZURE_GENERAL padding_layout(const padding_layout &rhs) :
			shape_(rhs.shape_),
			origin_padding_(rhs.origin_padding_),
			end_padding_(rhs.end_padding_),
			stride_(rhs.stride_)
		{ }

		#pragma hd_warning_disable
		MATAZURE_GENERAL padding_layout & operator=(const padding_layout &rhs) {
			shape_ = rhs.shape_;
			origin_padding_ = rhs.origin_padding_;
			end_padding_ = rhs.end_padding_;
			stride_ = rhs.stride_;
			return *this;
		}

		MATAZURE_GENERAL int_t index2offset(const pointi<rank> &id) const {
			int_t offset = id[0] + origin_padding_[0];
			for (int_t i = 1; i < rank; ++i) {
				offset += (id[i] + origin_padding_[i]) * stride_[i - 1];
			}

			return offset;
		};

		MATAZURE_GENERAL pointi<rank> offset2index(int_t offset) const {
			pointi<rank> id{};
			for (int_t i = rank - 1; i > 0; --i) {
				id[i] = offset / stride_[i - 1] - origin_padding_[i];
				offset = offset % stride_[i - 1];
			}
			id[0] = offset - origin_padding_[0];

			return id;
		}

		MATAZURE_GENERAL pointi<rank> shape() const{
			return shape_;
		}

		MATAZURE_GENERAL pointi<rank> stride() const{
			return stride_;
		}

		MATAZURE_GENERAL pointi<rank> origin_padding() const {
			return origin_padding_;
		}

		MATAZURE_GENERAL pointi<rank> end_padding() const {
			return end_padding_;
		}

		MATAZURE_GENERAL ~padding_layout() { }

	private:
		pointi<rank> shape_;
		pointi<rank> origin_padding_;
		pointi<rank> end_padding_;
		pointi<rank> stride_;
	};

}
