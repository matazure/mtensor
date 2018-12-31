#pragma once

#include <matazure/tensor.hpp>

namespace matazure {

	enum struct data_type {
		undefined = 0,
		dt_uint8,
		dt_uint16,
		dt_uint32,
		dt_uint64,
		dt_int8,
		dt_int16,
		dt_int32,
		dt_int64,
		dt_float16,
		dt_float32,
		dt_float64
	};

	template <typename  _T>
	struct get_data_type_traits;
	template <>	struct get_data_type_traits<std::uint8_t>	{ const static data_type value = data_type::dt_uint8; };
	template <> struct get_data_type_traits<std::uint16_t>	{ const static data_type value = data_type::dt_uint16; };
	template <> struct get_data_type_traits<std::uint32_t>	{ const static data_type value = data_type::dt_uint32; };
	template <> struct get_data_type_traits<std::uint64_t>	{ const static data_type value = data_type::dt_uint64; };
	template <> struct get_data_type_traits<std::int8_t>	{ const static data_type value = data_type::dt_int8; };
	template <> struct get_data_type_traits<std::int16_t>	{ const static data_type value = data_type::dt_int16; };
	template <> struct get_data_type_traits<std::int32_t>	{ const static data_type value = data_type::dt_int32; };
	template <> struct get_data_type_traits<std::int64_t>	{ const static data_type value = data_type::dt_int64; };

	//template <> struct get_data_type_traits<std::float> { const static data_type value = data_type::dt_float16; };
	template <> struct get_data_type_traits<float>	{ const static data_type value = data_type::dt_float32; };
	template <> struct get_data_type_traits<double>	{ const static data_type value = data_type::dt_float64; };

	inline int_t get_data_type_size(data_type type) {
		switch (type) {
		case data_type::dt_uint8: return 1;
			case data_type::dt_int8:	return 1;
			case data_type::dt_uint16:	return 2;
			case data_type::dt_int16:	return 2;
			case data_type::dt_uint32:	return 4;
			case data_type::dt_int32:	return 4;
			case data_type::dt_float16:	return 2;
			case data_type::dt_float32:	return 4;
			case data_type::dt_float64:	return 8;
			default: MATAZURE_ASSERT(false, "unreachable");
		}

		return 0;
	}

	class dynamic_tensor {
	public:

		using shape_type = tensor<int_t, 1>;

		dynamic_tensor() {}

		dynamic_tensor(data_type type, shape_type ts_shape) :
			type_(type),
			ts_shape_(ts_shape),
			size_(reduce(ts_shape_, 1, [](auto x0, auto x1){ return x0 * x1; }))
		{
			auto p_mem_ = new byte[size_ * element_size()];
			sp_mem_.reset(p_mem_, [](byte *p) { delete[] p; });
		}

		dynamic_tensor(data_type type, shape_type ts_shape, shared_ptr<void> sp_mem) :
			type_(type),
			ts_shape_(ts_shape),
			size_(reduce(ts_shape_, 1, [](auto x0, auto x1){ return x0 * x1; })),
			sp_mem_(std::static_pointer_cast<byte>(sp_mem))
		{ }

		dynamic_tensor(const dynamic_tensor &other) :
			type_(other.type_),
			ts_shape_(other.ts_shape_),
			size_(other.size_),
			sp_mem_(other.sp_mem_)
		{}

		dynamic_tensor & operator=(const dynamic_tensor &other) {
			type_ = other.type_;
			ts_shape_ = other.ts_shape_;
			size_ = other.size_;
			sp_mem_ = other.sp_mem_;
			return *this;
		}

		data_type type() const {
			return type_;
		}

		shape_type shape() const {
			return ts_shape_;
		}

		int_t rank() const {
			return ts_shape_.size();
		}

		int_t size() const {
			return size_;
		}

		template <typename _Type = byte>
		shared_ptr<_Type> shared_data() {
			shared_ptr<_Type> sp_tmp(data<_Type>(), [sp_mem = sp_mem_](auto p) {});
			return sp_tmp;
		}

		template <typename _Type = byte>
		shared_ptr<const _Type> shared_data() const {
			shared_ptr<const _Type> sp_tmp(data<_Type>(), [sp_mem = sp_mem_](auto p) {});
			return sp_tmp;
		}

		template <typename _Type = byte>
		_Type* data() {
			return reinterpret_cast<_Type*>(sp_mem_.get());
		}

		template <typename _Type = byte>
		const _Type* data() const {
			return reinterpret_cast<_Type *>(sp_mem_.get());
		}

		int_t element_size() const {
			return  get_data_type_size(type_);
		}

	private:
		data_type type_;
		shared_ptr<byte> sp_mem_ = nullptr;
		shape_type ts_shape_;
		int_t size_;
	};

	template <typename _Tensor>
	dynamic_tensor dynamic_tensor_wrap(_Tensor ts){
		auto rank = _Tensor::rank;
		dynamic_tensor::shape_type shape(rank);
		copy(ts.shape(), shape);
		shared_ptr<byte> sp_tmp(reinterpret_cast<byte*>(ts.data()), [ts](auto p) {});
		return dynamic_tensor(get_data_type_traits<typename _Tensor::value_type>::value, shape, sp_tmp);
	}

}
