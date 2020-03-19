#include <chrono>
#include <vector>
#include <matazure/tensor>
#include <xmmintrin.h>

using namespace std;
using namespace matazure;

typedef float v4sf __attribute__((vector_size(16)));


int main(int argc, char *argv[]) {

    int mem_size = 1024 * 1024 * 200;

    int item_size = mem_size / sizeof(v4sf);
    int iter_num = 1000;

    v4sf _add;

    // static_assert(std::is_same<v4sf, std::decay_t<v4sf>>::value);

    auto t = new v4sf[3];
    std::cout << "size: " << sizeof(*t) << std::endl;

	tensor<v4sf, 1> ts_a(pointi<1>{item_size});
	tensor<v4sf, 1> ts_b(pointi<1>{item_size});
	// vector<byte> ts_a(item_size);
    // vector<float> ts_b(item_size);
    // tmp_vector ts_a(item_size);
    // tmp_vector ts_b(item_size);
    for(int i = 0; i < item_size; ++i) {
        ts_a[i][0] = i % 256;
    }
    // auto a = ts_a.data();
    // auto b = ts_b.data();
    // ts_a.sp_data = shared_ptr<float>(ts_a.data, [](float *p) { delete [] p; });
    auto t0 = chrono::high_resolution_clock::now();

    auto b = ts_b[0];

    for (int i = 0; i < iter_num; ++i) {
        /*
        for (size_t j = 0; j +256 < item_size; j+=64) {
            // _mm_prefetch(&(ts_a[j + 4] ), _MM_HINT_NTA);
            // _mm_prefetch(&(ts_a[j + 8]), _MM_HINT_NTA);
            // _mm_prefetch(&(ts_a[j + 12]), _MM_HINT_NTA);
            // _mm_prefetch(&(ts_a[j + 16]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 20]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 24]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 28]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 32]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 36]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 40]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 44]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 48]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 52]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 56]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_a[j + 60]), _MM_HINT_NTA);
			_mm_prefetch(&(ts_a[j + 64]), _MM_HINT_NTA);

			// _mm_prefetch(&(ts_b[j + 4]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 8]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 12]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 16]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 20]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 24]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 28]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 32]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 36]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 40]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 44]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 48]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 52]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 56]), _MM_HINT_NTA);
			// _mm_prefetch(&(ts_b[j + 60]), _MM_HINT_NTA);
			_mm_prefetch(&(ts_b[j + 64]), _MM_HINT_NTA);

			for (size_t k = 0; k < 64; ++k) {
                ts_b[j+k] = ts_a[j+k];
            }

        }
        */
        memset(ts_b.data(), 0, item_size * sizeof(ts_a[0]));
    }
    auto t1 = chrono::high_resolution_clock::now();

    float item =  item_size * 1.0f * iter_num  / (t1-t0).count();

    std::cout << "mem " << item *  sizeof(ts_a[0]) << " GB/s" << "item: " << item << "Ghz"  << std::endl;

    int array[132] = {213};

    return array[0];
}
