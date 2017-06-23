#pragma once

#include <matazure/config.hpp>

namespace matazure {

struct first_major_t {};
struct last_major_t {};
typedef first_major_t col_major_t;
typedef last_major_t row_major_t;

struct linear_access_t {};
struct array_access_t {};

struct host_t {};
struct device_t {};
struct local_t {};

struct pinned_t {};
struct unpinned_t {};

/// define a generical compile time zero
template <typename _T>
struct zero;

/// special for most type, value() return 0 directly;
template <typename _T>
struct zero {
	MATAZURE_GENERAL static constexpr _T value() {
		return 0;
	};
};

/// forward declare of tensor_expression
template <typename _Tensor>
class tensor_expression;

/// a type traits to get the argument type and result type of a functor
template <typename _Func>
struct function_traits
	: public function_traits<decltype(&_Func::operator())>
{ };

/// implements
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

/// special for the tensor_expression models, they are tensor.
template <typename _Type>
struct is_tensor : bool_constant<std::is_base_of<tensor_expression<_Type>, _Type>::value> {};

/// specail for the tensor liked type, they are linear array tensor.
template <typename _Type>
struct is_linear_array : bool_constant<is_tensor<_Type>::value> {};

//are tag
#define MATAZURE_DEFINE_ARE_TAG(name, tag_name, tag)						\
template <typename ..._Tensor>												\
struct name;																\
template <>																	\
																			\
struct name<> : bool_constant<true> {};										\
																			\
template <typename _Tensor, typename ..._OtherTensors>						\
struct name<_Tensor, _OtherTensors...> : bool_constant<						\
	is_same<typename _Tensor::tag_name, tag>::value							\
	&& name<_OtherTensors...>::value> {};

MATAZURE_DEFINE_ARE_TAG(are_host_memory, memory_type, host_t)
MATAZURE_DEFINE_ARE_TAG(are_device_memory, memory_type, device_t)
MATAZURE_DEFINE_ARE_TAG(are_linear_access, access_type, linear_access_t)
MATAZURE_DEFINE_ARE_TAG(are_array_access, access_type, array_access_t)

//none tag
#define MATAZURE_DEFINE_NONE_TAG(name, tag_name, tag)		\
template <typename ..._Tensor>								\
struct name;												\
template <>													\
															\
struct name<> : bool_constant<true> {};						\
															\
template <typename _Tensor, typename ..._OtherTensors>		\
struct name<_Tensor, _OtherTensors...> : bool_constant<		\
	!is_same<typename _Tensor::tag_name, tag>::value		\
	&& name<_OtherTensors...>::value> {};

MATAZURE_DEFINE_NONE_TAG(none_host_memory, memory_type, host_t)
MATAZURE_DEFINE_NONE_TAG(none_device_memory, memory_type, device_t)
MATAZURE_DEFINE_NONE_TAG(none_linear_access, access_type, linear_access_t)
MATAZURE_DEFINE_NONE_TAG(none_array_access, access_type, array_access_t)

}
