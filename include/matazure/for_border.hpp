#pragma once

#include <matazure/algorithm.hpp>

namespace matazure {

	template <typename _Func>
	inline void for_border(const pointi<2> & extent, const pointi<2> & origin_padding, const pointi<2> & end_padding, _Func fun) {
		//top
		for (int j = -origin_padding[1]; j < 0; ++j) {
			for (int i = -origin_padding[0]; i < extent[0] + end_padding[0]; ++i) {
				fun(pointi<2>{i, j});
			}
		}

		//center
		for (int j = 0; j < extent[1]; ++j) {
			for (int i = -origin_padding[0]; i < 0; ++i) {
				fun(pointi<2>{i, j});
			}
			for (int i = extent[0]; i < extent[0] + end_padding[0]; ++i) {
				fun(pointi<2>{i, j});
			}
		}

		//bottome
		for (int j = extent[1]; j < extent[1] + end_padding[1]; ++j) {
			for (int i = -origin_padding[0]; i < extent[0] + end_padding[0]; ++i) {
				fun(pointi<2>{i, j});
			}
		}
	}

}
