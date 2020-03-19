#include "bm_config.hpp"

#ifdef __aarch64__

#include <arm_neon.h>


typedef float32x4_t value_type3;

void bm_conv3(benchmark::State & state) {
    auto width = state.range(0);
    auto height = width;
    pointi<2> padding = {1, 1};
    tensor<value_type3, 2> img_in(pointi<2>{width + 2, height + 2});
    tensor<value_type3, 2> kernel(3,3);
    tensor<value_type3, 2> img_out(img_in.shape());

    while (state.KeepRunning()) {
        // benchmark::DoNotOptimize(conv3x3(img_in, kernel, img_out));
     
        auto p_r0 = &img_in(0, 0);
        auto p_r1 = &img_in(0, 1);
        auto p_r2 = &img_in(0, 2);
        auto p_r3 = &img_in(0, 3);
        auto p_out0 = &img_out(0, 0);
        auto p_out1 = &img_out(0, 1);
        for (int r = 0; r + 1< height; r += 2) {
          for (int c = 0; c + 3 < width; c += 4) {
              value_type3 out10 ={0};
              value_type3 out11 ={0};
              value_type3 out12 ={0};
              value_type3 out13 ={0};

              
              value_type3 out00 ={0};
              value_type3 out01 ={0};
              value_type3 out02 ={0};
              value_type3 out03 ={0};

              value_type3 r10 =  *(p_r1 + 0);
              value_type3 r11 =  *(p_r1 + 1);
              value_type3 r12 =  *(p_r1 + 2);
              value_type3 r13 =  *(p_r1 + 3);
              value_type3 r14 =  *(p_r1 + 4);
              value_type3 r15 =  *(p_r1 + 5);

              out10 = vfmaq_f32(out10, r10, kernel[0]);
              out11 = vfmaq_f32(out11, r11, kernel[0]);
              out12 = vfmaq_f32(out12, r12, kernel[0]);
              out13 = vfmaq_f32(out13, r13, kernel[0]);
              
              out00 = vfmaq_f32(out00, r10, kernel[3]);
              out01 = vfmaq_f32(out01, r11, kernel[3]);
              out02 = vfmaq_f32(out02, r12, kernel[3]);
              out03 = vfmaq_f32(out03, r13, kernel[3]);

              out10 = vfmaq_f32(out10, r11, kernel[1]);
              out11 = vfmaq_f32(out11, r12, kernel[1]);
              out12 = vfmaq_f32(out12, r13, kernel[1]);
              out13 = vfmaq_f32(out13, r14, kernel[1]);

              out00 = vfmaq_f32(out00, r11, kernel[4]);
              out01 = vfmaq_f32(out01, r12, kernel[4]);
              out02 = vfmaq_f32(out02, r13, kernel[4]);
              out03 = vfmaq_f32(out03, r14, kernel[4]);

              out10 = vfmaq_f32(out10, r12, kernel[2]);
              out11 = vfmaq_f32(out11, r13, kernel[2]);
              out12 = vfmaq_f32(out12, r14, kernel[2]);
              out13 = vfmaq_f32(out13, r15, kernel[2]);

              out00 = vfmaq_f32(out00, r12, kernel[5]);
              out01 = vfmaq_f32(out01, r13, kernel[5]);
              out02 = vfmaq_f32(out02, r14, kernel[5]);
              out03 = vfmaq_f32(out03, r15, kernel[5]);

              value_type3 r20 = *(p_r2 + 0);
              value_type3 r21 = *(p_r2 + 1);
              value_type3 r22 = *(p_r2 + 2);
              value_type3 r23 = *(p_r2 + 3);
              value_type3 r24 = *(p_r2 + 4);
              value_type3 r25 = *(p_r2 + 5);

              out10 = vfmaq_f32(out10, r20, kernel[3]);
              out11 = vfmaq_f32(out11, r21, kernel[3]);
              out12 = vfmaq_f32(out12, r22, kernel[3]);
              out13 = vfmaq_f32(out13, r23, kernel[3]);
              
              out00 = vfmaq_f32(out00, r20, kernel[6]);
              out01 = vfmaq_f32(out01, r21, kernel[6]);
              out02 = vfmaq_f32(out02, r22, kernel[6]);
              out03 = vfmaq_f32(out03, r23, kernel[6]);

              out10 = vfmaq_f32(out10, r21, kernel[4]);
              out11 = vfmaq_f32(out11, r22, kernel[4]);
              out12 = vfmaq_f32(out12, r23, kernel[4]);
              out13 = vfmaq_f32(out13, r24, kernel[4]);
              
              out00 = vfmaq_f32(out00, r21, kernel[7]);
              out01 = vfmaq_f32(out01, r22, kernel[7]);
              out02 = vfmaq_f32(out02, r23, kernel[7]);
              out03 = vfmaq_f32(out03, r24, kernel[7]);

              out10 = vfmaq_f32(out10, r22, kernel[5]);
              out11 = vfmaq_f32(out11, r23, kernel[5]);
              out12 = vfmaq_f32(out12, r24, kernel[5]);
              out13 = vfmaq_f32(out13, r25, kernel[5]);

              out00 = vfmaq_f32(out00, r22, kernel[8]);
              out01 = vfmaq_f32(out01, r23, kernel[8]);
              out02 = vfmaq_f32(out02, r24, kernel[8]);
              out03 = vfmaq_f32(out03, r25, kernel[8]);


              value_type3 r00 = *(p_r0 + 0);
              value_type3 r01 = *(p_r0 + 1);
              value_type3 r02 = *(p_r0 + 2);
              value_type3 r03 = *(p_r0 + 3);
              value_type3 r04 = *(p_r0 + 4);
              value_type3 r05 = *(p_r0 + 5);


              out00 = vfmaq_f32(out00, r00, kernel[0]);
              out01 = vfmaq_f32(out01, r01, kernel[0]);
              out02 = vfmaq_f32(out02, r02, kernel[0]);
              out03 = vfmaq_f32(out03, r03, kernel[0]);

              out00 = vfmaq_f32(out00, r01, kernel[1]);
              out01 = vfmaq_f32(out01, r02, kernel[1]);
              out02 = vfmaq_f32(out02, r03, kernel[1]);
              out03 = vfmaq_f32(out03, r04, kernel[1]);

              out00 = vfmaq_f32(out00, r02, kernel[2]);
              out01 = vfmaq_f32(out01, r03, kernel[2]);
              out02 = vfmaq_f32(out02, r04, kernel[2]);
              out03 = vfmaq_f32(out03, r05, kernel[2]);

              value_type3 r30 = *(p_r3 + 0);
              value_type3 r31 = *(p_r3 + 1);
              value_type3 r32 = *(p_r3 + 2);
              value_type3 r33 = *(p_r3 + 3);
              value_type3 r34 = *(p_r3 + 4);
              value_type3 r35 = *(p_r3 + 5);

              out10 = vfmaq_f32(out10, r30, kernel[6]);
              out11 = vfmaq_f32(out11, r31, kernel[6]);
              out12 = vfmaq_f32(out12, r32, kernel[6]);
              out13 = vfmaq_f32(out13, r33, kernel[6]);

              out10 = vfmaq_f32(out10, r31, kernel[7]);
              out11 = vfmaq_f32(out11, r32, kernel[7]);
              out12 = vfmaq_f32(out12, r33, kernel[7]);
              out13 = vfmaq_f32(out13, r34, kernel[7]);

              out10 = vfmaq_f32(out10, r32, kernel[8]);
              out11 = vfmaq_f32(out11, r33, kernel[8]);
              out12 = vfmaq_f32(out12, r34, kernel[8]);
              out13 = vfmaq_f32(out13, r35, kernel[8]);

              *(p_out0 + 0) = out00;
              *(p_out0 + 1) = out01;
              *(p_out0 + 2) = out02;
              *(p_out0 + 3) = out03;
              *(p_out1 + 0) = out10;
              *(p_out1 + 1) = out11;
              *(p_out1 + 2) = out12;
              *(p_out1 + 3) = out13;

              p_r0 += 4;
              p_r1 += 4;
              p_r2 += 4;
              p_r3 += 4;
              p_out0 += 4;
              p_out1 += 4;
          }
          p_out0 += width;
          p_out1 += width;
          p_r0 += 2 + width + 2;
          p_r1 += 2 + width + 2;
          p_r2 += 2 + width + 2;
          p_r3 += 2 + width + 2;
        }
    }

    auto byte_size = img_out.size() * sizeof(img_out[0]);
    auto item_size = img_out.size() * 4 * kernel.size() * 2;
    state.SetBytesProcessed(state.iterations() * byte_size);
    state.SetItemsProcessed(state.iterations() * item_size);
}

BENCHMARK(bm_conv3)->RangeMultiplier(2)->Range(16, 2048);



typedef float32x4_t value_type2;
static void convdw3x3s1_pack4_neon(tensor<value_type2, 2> ts_bottom, tensor<value_type2, 2> ts_top, tensor<value_type2, 2> ts_kernel)
{
    // int w = bottom_blob.w;
    int w = ts_bottom.shape()[0];

    int outw = ts_top.shape()[0];
    int outh = ts_top.shape()[1];

    // int outw = top_blob.w;
    // int outh = top_blob.h;

    // const int group = bottom_blob.c;
    const int group = 1;

    const float* bias = nullptr;

    #pragma omp parallel for num_threads(1)
    for (int g=0; g<group; g++)
    {
        // Mat out = top_blob.channel(g);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

        // const float* k0 = kernel.row(g);
        const float * k0 = reinterpret_cast<float *>(ts_kernel.data());

        // float* outptr0 = out.row(0);
        // float* outptr1 = out.row(1);

     

        // const Mat img0 = bottom_blob.channel(g);

        // const float* r0 = img0.row(0);
        // const float* r1 = img0.row(1);
        // const float* r2 = img0.row(2);
        // const float* r3 = img0.row(3);



        float32x4_t _k00 = vld1q_f32(k0);
        float32x4_t _k01 = vld1q_f32(k0+4);
        float32x4_t _k02 = vld1q_f32(k0+8);
        float32x4_t _k10 = vld1q_f32(k0+12);
        float32x4_t _k11 = vld1q_f32(k0+16);
        float32x4_t _k12 = vld1q_f32(k0+20);
        float32x4_t _k20 = vld1q_f32(k0+24);
        float32x4_t _k21 = vld1q_f32(k0+28);
        float32x4_t _k22 = vld1q_f32(k0+32);

        int i = 0;

        for (; i+1 < outh; i+=2)
        {
            int j = 0;

            for (; j+3 < outw; j+=4)
            {
                float * outptr0 = reinterpret_cast<float *>(&ts_top(0, i + 0));
                float * outptr1 = reinterpret_cast<float *>(&ts_top(0, i + 1));
                const float* r0 = reinterpret_cast<float *>(&ts_bottom(0, i + 0));
                const float* r1 = reinterpret_cast<float *>(&ts_bottom(0, i + 1));
                const float* r2 = reinterpret_cast<float *>(&ts_bottom(0, i + 2));
                const float* r3 = reinterpret_cast<float *>(&ts_bottom(0, i + 3));

                asm volatile(
                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v10.4s, v11.4s}, [%3], #32 \n"// r10 r11

                    "mov    v16.16b, %21.16b            \n"// sum00
                    "mov    v17.16b, %21.16b            \n"// sum01
                    "mov    v18.16b, %21.16b            \n"// sum02
                    "mov    v19.16b, %21.16b            \n"// sum03

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%3] \n"// r12 r13 r14 r15

                    "mov    v20.16b, %21.16b            \n"// sum10
                    "mov    v21.16b, %21.16b            \n"// sum11
                    "mov    v22.16b, %21.16b            \n"// sum12
                    "mov    v23.16b, %21.16b            \n"// sum13

                    "fmla   v16.4s, %15.4s, v10.4s      \n"
                    "fmla   v17.4s, %15.4s, v11.4s      \n"
                    "fmla   v18.4s, %15.4s, v12.4s      \n"
                    "fmla   v19.4s, %15.4s, v13.4s      \n"
                    "fmla   v20.4s, %12.4s, v10.4s      \n"
                    "fmla   v21.4s, %12.4s, v11.4s      \n"
                    "fmla   v22.4s, %12.4s, v12.4s      \n"
                    "fmla   v23.4s, %12.4s, v13.4s      \n"

                    "add    %3, %3, #32                 \n"

                    "fmla   v16.4s, %16.4s, v11.4s      \n"
                    "fmla   v17.4s, %16.4s, v12.4s      \n"
                    "fmla   v18.4s, %16.4s, v13.4s      \n"
                    "fmla   v19.4s, %16.4s, v14.4s      \n"
                    "fmla   v20.4s, %13.4s, v11.4s      \n"
                    "fmla   v21.4s, %13.4s, v12.4s      \n"
                    "fmla   v22.4s, %13.4s, v13.4s      \n"
                    "fmla   v23.4s, %13.4s, v14.4s      \n"

                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v10.4s, v11.4s}, [%4], #32 \n"// r20 r21

                    "fmla   v16.4s, %17.4s, v12.4s      \n"
                    "fmla   v17.4s, %17.4s, v13.4s      \n"
                    "fmla   v18.4s, %17.4s, v14.4s      \n"
                    "fmla   v19.4s, %17.4s, v15.4s      \n"
                    "fmla   v20.4s, %14.4s, v12.4s      \n"
                    "fmla   v21.4s, %14.4s, v13.4s      \n"
                    "fmla   v22.4s, %14.4s, v14.4s      \n"
                    "fmla   v23.4s, %14.4s, v15.4s      \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4] \n"// r22 r23 r24 r25

                    "fmla   v16.4s, %18.4s, v10.4s      \n"
                    "fmla   v17.4s, %18.4s, v11.4s      \n"
                    "fmla   v18.4s, %18.4s, v12.4s      \n"
                    "fmla   v19.4s, %18.4s, v13.4s      \n"
                    "fmla   v20.4s, %15.4s, v10.4s      \n"
                    "fmla   v21.4s, %15.4s, v11.4s      \n"
                    "fmla   v22.4s, %15.4s, v12.4s      \n"
                    "fmla   v23.4s, %15.4s, v13.4s      \n"

                    "add    %4, %4, #32                 \n"

                    "fmla   v16.4s, %19.4s, v11.4s      \n"
                    "fmla   v17.4s, %19.4s, v12.4s      \n"
                    "fmla   v18.4s, %19.4s, v13.4s      \n"
                    "fmla   v19.4s, %19.4s, v14.4s      \n"
                    "fmla   v20.4s, %16.4s, v11.4s      \n"
                    "fmla   v21.4s, %16.4s, v12.4s      \n"
                    "fmla   v22.4s, %16.4s, v13.4s      \n"
                    "fmla   v23.4s, %16.4s, v14.4s      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v10.4s, v11.4s}, [%2], #32 \n"// r00 r01

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v24.4s, v25.4s}, [%5], #32 \n"// r30 r31

                    "fmla   v16.4s, %20.4s, v12.4s      \n"
                    "fmla   v17.4s, %20.4s, v13.4s      \n"
                    "fmla   v18.4s, %20.4s, v14.4s      \n"
                    "fmla   v19.4s, %20.4s, v15.4s      \n"
                    "fmla   v20.4s, %17.4s, v12.4s      \n"
                    "fmla   v21.4s, %17.4s, v13.4s      \n"
                    "fmla   v22.4s, %17.4s, v14.4s      \n"
                    "fmla   v23.4s, %17.4s, v15.4s      \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2] \n"// r02 r03 r04 r05

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v26.4s, v27.4s, v28.4s, v29.4s}, [%5] \n"// r32 r33 r34 r35

                    "fmla   v16.4s, %12.4s, v10.4s      \n"
                    "fmla   v17.4s, %12.4s, v11.4s      \n"
                    "fmla   v18.4s, %12.4s, v12.4s      \n"
                    "fmla   v19.4s, %12.4s, v13.4s      \n"
                    "fmla   v20.4s, %18.4s, v24.4s      \n"
                    "fmla   v21.4s, %18.4s, v25.4s      \n"
                    "fmla   v22.4s, %18.4s, v26.4s      \n"
                    "fmla   v23.4s, %18.4s, v27.4s      \n"

                    "add    %2, %2, #32                 \n"

                    "fmla   v16.4s, %13.4s, v11.4s      \n"
                    "fmla   v17.4s, %13.4s, v12.4s      \n"
                    "fmla   v18.4s, %13.4s, v13.4s      \n"
                    "fmla   v19.4s, %13.4s, v14.4s      \n"
                    "fmla   v20.4s, %19.4s, v25.4s      \n"
                    "fmla   v21.4s, %19.4s, v26.4s      \n"
                    "fmla   v22.4s, %19.4s, v27.4s      \n"
                    "fmla   v23.4s, %19.4s, v28.4s      \n"

                    "add    %5, %5, #32                 \n"

                    "fmla   v16.4s, %14.4s, v12.4s      \n"
                    "fmla   v17.4s, %14.4s, v13.4s      \n"
                    "fmla   v18.4s, %14.4s, v14.4s      \n"
                    "fmla   v19.4s, %14.4s, v15.4s      \n"
                    "fmla   v20.4s, %20.4s, v26.4s      \n"
                    "fmla   v21.4s, %20.4s, v27.4s      \n"
                    "fmla   v22.4s, %20.4s, v28.4s      \n"
                    "fmla   v23.4s, %20.4s, v29.4s      \n"

                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"

                    : "=r"(outptr0),    // %0
                      "=r"(outptr1),    // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3)          // %5
                    : "0"(outptr0),
                      "1"(outptr1),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "w"(_k00),        // %12
                      "w"(_k01),        // %13
                      "w"(_k02),        // %14
                      "w"(_k10),        // %15
                      "w"(_k11),        // %16
                      "w"(_k12),        // %17
                      "w"(_k20),        // %18
                      "w"(_k21),        // %19
                      "w"(_k22),        // %20
                      "w"(_bias0)       // %21
                    : "memory", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"
                );
            }

            // r0 += 2 * 4 + w * 4;
            // r1 += 2 * 4 + w * 4;
            // r2 += 2 * 4 + w * 4;
            // r3 += 2 * 4 + w * 4;

            // outptr0 += outw * 4;
            // outptr1 += outw * 4;
        }
    }
}

void bm_conv_ncnn(benchmark::State & state) {
    auto width = state.range(0);
    auto height = width;
    pointi<2> padding = {1, 1};
    tensor<value_type2, 2> img_in(pointi<2>{width + 2, height + 2});
    tensor<value_type2, 2> kernel(3, 3);
    tensor<value_type2, 2> img_out(img_in.shape());

    while (state.KeepRunning()) {
        // benchmark::DoNotOptimize(conv3x3(img_in, kernel, img_out));
        convdw3x3s1_pack4_neon(img_in, img_out, kernel);
    }

    auto byte_size = img_out.size() * sizeof(img_out[0]);
    auto item_size = img_out.size() * 4 * kernel.size() * 2;
    state.SetBytesProcessed(state.iterations() * byte_size);
    state.SetItemsProcessed(state.iterations() * item_size);
}

BENCHMARK(bm_conv_ncnn)->RangeMultiplier(2)->Range(16, 2048);

#endif

BENCHMARK_MAIN();