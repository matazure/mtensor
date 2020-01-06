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
			shape_(shape),
			stride_(get_stride(shape))
		{
			// for_each(shape_, [](int_t b){
			// 	if (b < 0) throw invalid_shape{};
			// });
		}

		MATAZURE_GENERAL first_major_layout(const first_major_layout &rhs) :
			first_major_layout(rhs.shape())
		{ }

		#pragma hd_warning_disable
		MATAZURE_GENERAL first_major_layout & operator=(const first_major_layout &rhs) {
			shape_ = rhs.shape();
			stride_ = get_stride(shape_);

			// matazure::for_each(shape_, [](int_t b) {
			// 	if (b < 0) throw invalid_shape{};
			// });

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
		static pointi<rank> get_stride(pointi<rank> ext) {
			pointi<rank>  stride{};
			stride[0] = ext[0];
			for (int_t i = 1; i < rank; ++i) {
				stride[i] = ext[i] * stride[i - 1];
			}

			return stride;
		}

	private:
		pointi<rank> shape_;
		pointi<rank> stride_;
	};

	template <int_t _Rank>
	class last_major_layout{
	public:
		const static int_t rank = _Rank;

		last_major_layout(const pointi<rank> &shape) :
			shape_(shape),
			stride_(get_stride(shape))
		{ }

		MATAZURE_GENERAL int_t index2offset(const pointi<rank> &id) const {
			typename pointi<rank>::value_type offset = id[rank - 1];
			for (int_t i = 1; i < rank; ++i) {
				offset += id[rank - 1 - i] * stride_[i - 1];
			}

			return offset;
		};

		MATAZURE_GENERAL pointi<rank> offset2index(int_t offset) const {
			pointi<rank> id{};
			for (int_t i = rank - 1; i > 0; --i) {
				id[rank - 1 - i] = offset / stride_[i - 1];
				offset = offset % stride_[i - 1];
			}
			id[rank - 1] = offset;

			return id;
		}

		pointi<rank> shape() const{
			return shape_;
		}

		pointi<rank> stride() const{
			return stride_;
		}


	private:
		static pointi<rank> get_stride(pointi<rank> ext) {
			pointi<rank>  stride{};
			stride[0] = ext[rank - 1];
			for (int_t i = 1; i < rank; ++i) {
				stride[i] = ext[rank - 1 -i] * stride[i - 1];
			}
			return stride;
		}

	private:
		pointi<rank> shape_;
		pointi<rank> stride_;
	};

}
