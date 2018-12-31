#pragma once

#include <cereal/cereal.hpp>
#include <matazure/tensor>

namespace cereal {
	//! Saving for matazure::point primitive types
	//! using binary serialization, if supported
	template <class Archive, class T, matazure::int_t N> inline
		typename std::enable_if<traits::is_output_serializable<BinaryData<T>, Archive>::value
		&& std::is_arithmetic<T>::value, void>::type
		CEREAL_SAVE_FUNCTION_NAME(Archive & ar, matazure::point<T, N> const & point)
	{
		ar(binary_data(point.data(), sizeof(point)));
	}

	//! Loading for matazure::point primitive types
	//! using binary serialization, if supported
	template <class Archive, class T, matazure::int_t N> inline
		typename std::enable_if<traits::is_input_serializable<BinaryData<T>, Archive>::value
		&& std::is_arithmetic<T>::value, void>::type
		CEREAL_LOAD_FUNCTION_NAME(Archive & ar, matazure::point<T, N> & point)
	{
		ar(binary_data(point.data(), sizeof(point)));
	}

	//! Saving for matazure::point all other types
	template <class Archive, class T, matazure::int_t N> inline
		typename std::enable_if<!traits::is_output_serializable<BinaryData<T>, Archive>::value
		|| !std::is_arithmetic<T>::value, void>::type
		CEREAL_SAVE_FUNCTION_NAME(Archive & ar, matazure::point<T, N> const & point)
	{
		matazure::for_each(point, [&ar](auto &e) {
			ar(e);
		});
	}

	//! Loading for matazure::point all other types
	template <class Archive, class T, matazure::int_t N> inline
		typename std::enable_if<!traits::is_input_serializable<BinaryData<T>, Archive>::value
		|| !std::is_arithmetic<T>::value, void>::type
		CEREAL_LOAD_FUNCTION_NAME(Archive & ar, matazure::point<T, N> & point)
	{
		matazure::for_each(point, [&ar](auto &e) {
			ar(e);
		});
	}

	//! Serialization for std::vectors of arithmetic (but not bool) using binary serialization, if supported
	template <class Archive, class T, matazure::int_t N, class Layout> inline
		typename std::enable_if<traits::is_output_serializable<BinaryData<T>, Archive>::value
		&& std::is_arithmetic<T>::value && !std::is_same<T, bool>::value, void>::type
		CEREAL_SAVE_FUNCTION_NAME(Archive & ar, matazure::tensor<T, N, Layout> const & ts)
	{
		ar(make_nvp("shape", ts.shape())); // number of elements
		ar(binary_data(ts.data(), static_cast<size_t>(ts.size()) * sizeof(T)));
	}

	//! Serialization for std::vectors of arithmetic (but not bool) using binary serialization, if supported
	template <class Archive, class T, matazure::int_t N, class Layout> inline
		typename std::enable_if<traits::is_input_serializable<BinaryData<T>, Archive>::value
		&& std::is_arithmetic<T>::value && !std::is_same<T, bool>::value, void>::type
		CEREAL_LOAD_FUNCTION_NAME(Archive & ar, matazure::tensor<T, N, Layout> & ts)
	{
		matazure::pointi<N> shape;
		ar(shape);
		ts = matazure::tensor<T, N, Layout>(shape);
		ar(binary_data(ts.data(), static_cast<size_t>(ts.size()) * sizeof(T)));
	}

	//! Serialization for non-arithmetic ts types
	template <class Archive, class T, matazure::int_t N, class Layout> inline
		typename std::enable_if<!traits::is_output_serializable<BinaryData<T>, Archive>::value
		|| !std::is_arithmetic<T>::value, void>::type
		CEREAL_SAVE_FUNCTION_NAME(Archive & ar, matazure::tensor<T, N, Layout> const & ts) {
		ar(make_nvp("shape", ts.shape()));
		for (matazure::int_t i = 0; i < ts.size(); ++i) {
			ar(ts[i]);
		}
	}

	//! Serialization for non-arithmetic ts types
	template <class Archive, class T, matazure::int_t N, class Layout> inline
		typename std::enable_if<!traits::is_input_serializable<BinaryData<T>, Archive>::value
		|| !std::is_arithmetic<T>::value, void>::type
		CEREAL_LOAD_FUNCTION_NAME(Archive & ar, matazure::tensor<T, N, Layout> & ts)
	{
		matazure::pointi<N> shape;
		ar(shape);
		ts = matazure::tensor<T, N, Layout>(shape);
		for (matazure::int_t i = 0; i < ts.size(); ++i) {
			ar(ts[i]);
		}
	}

	template <class Archive>
	inline std::string save_minimal(const Archive &, const matazure::data_type & dt) {
		using matazure::data_type;
		switch (dt) {
		case data_type::dt_uint8 :	return "uint8";
		case data_type::dt_uint16:	return "uint16";
		case data_type::dt_uint32:	return "uint32";
		case data_type::dt_uint64:	return "uint64";
		case data_type::dt_int8 :	return "int8";
		case data_type::dt_int16:	return "int16";
		case data_type::dt_int32:	return "int32";
		case data_type::dt_int64:	return "int64";
		case data_type::dt_float16:	return "float16";
		case data_type::dt_float32:	return "float32";
		case data_type::dt_float64:	return "float64";
		default: return "undefined";
		}

		return "undefined";
	}

	template <class Archive>
	inline void load_minimal(const Archive &, matazure::data_type & t, const std::string & value) {
		using matazure::data_type;
		if (value == "uint8")	t = data_type::dt_uint8;	return;
		if (value == "uint16")	t = data_type::dt_uint16;	return;
		if (value == "uint32")	t = data_type::dt_uint32;	return;
		if (value == "uint64")	t = data_type::dt_uint64;	return;
		if (value == "int8")	t = data_type::dt_int8;		return;
		if (value == "int16")	t = data_type::dt_int16;	return;
		if (value == "int32")	t = data_type::dt_int32;	return;
		if (value == "int64")	t = data_type::dt_int64;	return;
		if (value == "float16") t = data_type::dt_float16;	return;
		if (value == "float32") t = data_type::dt_float32;	return;
		if (value == "float64") t = data_type::dt_float64;	return;

		throw std::runtime_error("invalid data_type value");
	}

} // namespace cereal
