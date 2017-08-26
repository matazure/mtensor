#pragma once

#include <matazure/algorithm.hpp>

namespace matazure { namespace expr {

	template <typename _Kenel>
	inline void conv_kenel3x3(tensor<const float, 2> ts_input, _Kenel sts_kenel, tensor<float, 2> ts_output){
		auto kenel_radius = sts_kenel.shape() / 2;

		auto width = ts_input.shape()[0];
		auto height = ts_input.shape()[1];

		for(int_t j = 0; j < ts_input.shape()[1]; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0]; ++i) {
				if (i > 0 && j > 0 && i < width && j < height){
					float sum = 0.0f;
					for (int_t n = 0; n < sts_kenel.shape()[1]; ++n) {
						for (int_t m = 0; m < sts_kenel.shape()[0]; ++m) {
							sum += ts_input[pointi<2>{i, j} +pointi<2>{m, n} -kenel_radius] * sts_kenel[pointi<2>{m, n}];
						}
					}
					ts_output[pointi<2>{i, j}] = sum;
				}else{
					ts_output[pointi<2>{i, j}] = 0.0f;
				}
			}
		}
	}

} }
