#pragma once

#include <matazure/config.hpp>

namespace matazure {

struct first_major_t {};
struct last_major_t {};

struct linear_access_t {};
struct array_access_t {};

struct host_t {};
struct device_t {};
struct local_t {};

struct pinned_t{};
struct unpinned_t{};

struct static_t{};
struct dynamic_t{};
struct saturate_t{};

template <typename _T>
struct zero;

template <typename _T>
struct zero{
	MATAZURE_GENERAL static constexpr _T value(){
		return 0;
	};
};

template <typename _Tensor>
class tensor_expression;

template <typename T>
struct function_traits
	: public function_traits<decltype(&T::operator())>
{
	typedef double type;
};


template <typename _ClassType, typename _ReturnType, typename... _Args>
struct function_traits<_ReturnType(_ClassType::*)(_Args...) const> {
	enum { arguments_size = sizeof...(_Args) };

	typedef _ReturnType result_type;

	template <int_t _index>
	struct arguments
	{
		typedef typename std::tuple_element<_index, std::tuple<_Args...>>::type type;
	};
};

//are host memory
template <typename ..._Tensor>
struct are_host_memory;
template <>

struct are_host_memory<> : bool_constant<true> {};

template <typename _Tensor, typename ..._OtherTensors>
struct are_host_memory<_Tensor, _OtherTensors...> : bool_constant<
	is_same<typename _Tensor::memory_type, host_t>::value
	&& are_host_memory<_OtherTensors...>::value> {};

//are device memory
template <typename ..._Tensor>
struct are_device_memory;

template <>
struct are_device_memory<> : bool_constant<true> {};

template <typename _Tensor, typename ..._OtherTensors>
struct are_device_memory<_Tensor, _OtherTensors...> : bool_constant<
	is_same<typename _Tensor::memory_type, device_t>::value
	&& are_device_memory<_OtherTensors...>::value> {};

//are linear access
template <typename ..._Tensor>
struct are_linear_access;

template <>
struct are_linear_access<> : bool_constant<true> {};

template <typename _Tensor, typename ..._OtherTensors>
struct are_linear_access<_Tensor, _OtherTensors...> : bool_constant<
	is_same<typename _Tensor::access_type, linear_access_t>::value
	&& are_linear_access<_OtherTensors...>::value> {};

//are array access
template <typename ..._Tensor>
struct are_array_access;

template <>
struct are_array_access<> : bool_constant<true> {};

template <typename _Tensor, typename ..._OtherTensors>
struct are_array_access<_Tensor, _OtherTensors...> : bool_constant<
	is_same<typename _Tensor::access_type, array_access_t>::value
	&& are_array_access<_OtherTensors...>::value> {};

//none host memory
template <typename ..._Tensor>
struct none_host_memory;
template <>

struct none_host_memory<> : bool_constant<true> {};

template <typename _Tensor, typename ..._OtherTensors>
struct none_host_memory<_Tensor, _OtherTensors...> : bool_constant<
	!is_same<typename _Tensor::memory_type, host_t>::value
	&& none_host_memory<_OtherTensors...>::value> {};

//none device memory
template <typename ..._Tensor>
struct none_device_memory;

template <>
struct none_device_memory<> : bool_constant<true> {};

template <typename _Tensor, typename ..._OtherTensors>
struct none_device_memory<_Tensor, _OtherTensors...> : bool_constant<
	!is_same<typename _Tensor::memory_type, device_t>::value
	&& none_device_memory<_OtherTensors...>::value> {};

//none linear access
template <typename ..._Tensor>
struct none_linear_access;

template <>
struct none_linear_access<> : bool_constant<true> {};

template <typename _Tensor, typename ..._OtherTensors>
struct none_linear_access<_Tensor, _OtherTensors...> : bool_constant<
	!is_same<typename _Tensor::access_type, linear_access_t>::value
	&& none_linear_access<_OtherTensors...>::value> {};

//none array access
template <typename ..._Tensor>
struct none_array_access;

template <>
struct none_array_access<> : bool_constant<true> {};

template <typename _Tensor, typename ..._OtherTensors>
struct none_array_access<_Tensor, _OtherTensors...> : bool_constant<
	!is_same<typename _Tensor::access_type, array_access_t>::value
	&& none_array_access<_OtherTensors...>::value> {};

}
