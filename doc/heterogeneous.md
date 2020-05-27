# Heterogeneouse

gnu compiler向量化扩展是不支持的， 因为在cu文件里MATAZURE_GENERAL会被激活， 所以得确保gcc得特性是被device代码所支持， 显然向量化拓展的语法是不支持的， 需要自定义simd类型， 并用__CUDA_ARCH__来切换才行
