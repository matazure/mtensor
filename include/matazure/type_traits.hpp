#pragma once

#include <matazure/config.hpp>

namespace matazure {

struct first_major {};
struct last_major {};
typedef first_major col_major_t;
typedef last_major row_major_t;

struct linear_index {};
struct array_index {};

struct host_tag {};
struct device_tag {};
struct local_tag {};

struct pinned {};
struct unpinned {};

struct aligned{};

/// define a generical compile time zero
template <typename _T>
struct zero;

/// @todo: define one
template <typename _T>
struct one;

/// special for most type, value() return 0 directly;
template <typename _Type>
struct zero {
	MATAZURE_GENERAL static constexpr _Type value() {
		return _Type(0);
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

template <typename _Type>
struct _Is_linear_array;

///
template <typename _Type>
struct is_linear_array : public _Is_linear_array<remove_cv_t<_Type>> {};

/// specail for the tensor liked type, they are linear array tensor.
template <typename _Type>
struct _Is_linear_array : bool_constant<is_tensor<_Type>::value> {};

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

MATAZURE_DEFINE_ARE_TAG(are_host_memory, memory_type, host_tag)
MATAZURE_DEFINE_ARE_TAG(are_device_memory, memory_type, device_tag)
MATAZURE_DEFINE_ARE_TAG(are_linear_access, index_type, linear_index)
MATAZURE_DEFINE_ARE_TAG(are_array_access, index_type, array_index)

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

MATAZURE_DEFINE_NONE_TAG(none_host_memory, memory_type, host_tag)
MATAZURE_DEFINE_NONE_TAG(none_device_memory, memory_type, device_tag)
MATAZURE_DEFINE_NONE_TAG(none_linear_access, index_type, linear_index)
MATAZURE_DEFINE_NONE_TAG(none_array_access, index_type, array_index)

}
