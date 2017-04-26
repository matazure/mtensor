#pragma once

#include <matazure/config.hpp>
#include <array>

namespace matazure{
namespace meta{

	template <int_t ..._Values0>
	struct array{
		static constexpr int_t size(){
			return sizeof...(_Values0);
		}
	};

	template <int_t _Idx, int_t ..._Values>
	constexpr int_t at_c(array<_Values...> a) {		
		return std::array<int_t, sizeof...(_Values)>{_Values...}[_Idx];
	}

	template <int_t ..._Values0, int_t ..._Values1>
	constexpr auto add_c(array<_Values0...> a0, array<_Values1...> a1)->array<_Values0+_Values1...>{
		return array<_Values0 + _Values1...>{};
	}

}
}
