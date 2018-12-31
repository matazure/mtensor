#pragma once

#include <matazure/simd.hpp>

namespace matazure { namespace puzzle {


#ifdef MATAZURE_NEON

	inline const point<neon_vector<float, 4>, 4> transpose(const point<neon_vector<float, 4>, 4> &p4sv4f) {
		auto row0 = reinterpret_cast<const float32x4_t*>(&p4sv4f[0]);
		auto row1 = reinterpret_cast<const float32x4_t*>(&p4sv4f[0]);
		auto row2 = reinterpret_cast<const float32x4_t*>(&p4sv4f[0]);
		auto row3 = reinterpret_cast<const float32x4_t*>(&p4sv4f[0]);

		point<neon_vector<float, 4>, 4> re;
		auto p_re = reinterpret_cast<float32x4_t *>(re.data());

		auto tmp0 = vtrnq_f32(*row0, *row1);
		auto tmp1 = vtrnq_f32(*row2, *row3);

		auto re0_low = vget_low_f32(tmp0.val[0]);
		auto re0_high = vget_low_f32(tmp1.val[0]);
		auto re0 = vcombine_f32(re0_low, re0_high);
		p_re[0] = re0;

		auto re2_low = vget_high_f32(tmp0.val[0]);
		auto re2_high = vget_high_f32(tmp1.val[0]);
		auto re2 = vcombine_f32(re2_low, re2_high);
		p_re[2] = re2;

		auto re1_low = vget_low_f32(tmp0.val[1]);
		auto re1_high = vget_low_f32(tmp1.val[1]);
		auto re1 = vcombine_f32(re1_low, re1_high);
		p_re[1] = re1;

		auto re3_low = vget_high_f32(tmp0.val[1]);
		auto re3_high = vget_high_f32(tmp1.val[1]);
		auto re3 = vcombine_f32(re3_low, re3_high);
		p_re[3] = re3;

		return re;
	}

#endif

#ifdef MATAZURE_SSE

	inline const point<sse_vector<float, 4>, 4> transpose(const point<sse_vector<float, 4>, 4> &p4sv4f) {
		auto row0 = *reinterpret_cast<const __m128*>(&p4sv4f[0]);
		auto row1 = *reinterpret_cast<const __m128*>(&p4sv4f[0]);
		auto row2 = *reinterpret_cast<const __m128*>(&p4sv4f[0]);
		auto row3 = *reinterpret_cast<const __m128*>(&p4sv4f[0]);

		point<sse_vector<float, 4>, 4> re;
		auto p_re = reinterpret_cast<__m128*>(&re[0]);

		auto _Tmp0 = _mm_shuffle_ps((row0), (row1), 0x44);	
		auto _Tmp2 = _mm_shuffle_ps((row0), (row1), 0xEE);
		auto _Tmp1 = _mm_shuffle_ps((row2), (row3), 0x44);
		auto _Tmp3 = _mm_shuffle_ps((row2), (row3), 0xEE);
		
		p_re[0] = _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88);
		p_re[1] = _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD);
		p_re[2] = _mm_shuffle_ps(_Tmp2, _Tmp3, 0x88);
		p_re[3] = _mm_shuffle_ps(_Tmp2, _Tmp3, 0xDD);

		return re;
	}

#endif

	inline const static_tensor<float, dim<4, 4>> transpose(const static_tensor<float, dim<4, 4>> &sts_src) {
		static_tensor<float, dim<4, 4>> sts_re;
		sts_re(0, 0) = sts_src(0, 0);
		sts_re(0, 1) = sts_src(1, 0);
		sts_re(0, 2) = sts_src(2, 0);
		sts_re(0, 3) = sts_src(3, 0);

		sts_re(1, 0) = sts_src(0, 1);
		sts_re(1, 1) = sts_src(1, 1);
		sts_re(1, 2) = sts_src(2, 1);
		sts_re(1, 3) = sts_src(3, 1);

		sts_re(2, 0) = sts_src(0, 2);
		sts_re(2, 1) = sts_src(1, 2);
		sts_re(2, 2) = sts_src(2, 2);
		sts_re(2, 3) = sts_src(3, 2);

		sts_re(3, 0) = sts_src(0, 3);
		sts_re(3, 1) = sts_src(1, 3);
		sts_re(3, 2) = sts_src(2, 3);
		sts_re(3, 3) = sts_src(3, 3);

		return sts_re;
	}


} }
