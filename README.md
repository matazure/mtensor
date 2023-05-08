# mtensor

mtensor is a C++/CUDA template library that supports lazy evaluation of Tensor. Unlike Eigen, mtensor focus the tensor structure and supports cuda lazy evalution. mtensor does not provide linear algebra and numerical features.

## How to use

### tensor

We use the template class **tensor** to manage basic data structures, and we use it as the following

```c++
pointi<2> shape = {10, 20};
tensor<int, 2> cts(shape, runtime::cuda);
tensor<int, 2> ts(shape, runtime::host);
```

**tensor** can change value_type and dimensions through template parameters, and specify the memory type o by runtime host or cuda.

### lambda_tensor

**lambda_tensor** is a functional tensor which has strong representatility.
we can define a lambda_tensor by a "index->value" function with shape(domain).

```c++
pointi<2> shape = {10, 10};
tensor<int, 2> ts(shape);
auto fun = [=](pointi<2> idx) {
    return ts(idx) + ts(idx);
};
lambda_tensor<decltype(fun)> lts(shape, fun);
// will evaluate fun(pointi<2>{2,2}), and return it;
auto value = lts(2, 2);
```

### for_index

**for_index** is a functional loop, which supports host or cuda execution mode.
note that in the cuda runtime, we can only access the cuda tensor.

```c++
pointi<2> shape = {10, 10};
tensor<int, 2> cts(shape, runtime::cuda);
auto lts = make_lambda_tensor(shape, [](pointi<2> idx)->int{
    return idx[0] * 10 + idx[1];
});

for_index(shape, [=](pointi<2> idx){
    cts(idx) = lts(idx);
}, cts.runtime()); // in cuda runtime, we can only access cuda runtime tensor
```

We can extend for_index that implement different execution policy, for example we can extend an openmp for_index to use the CPU's multi-core.

## samples build

```bash
git submodule update --init .
cmake -B build
```

## How to integrate

Download [mtensor.hpp](mtensor.hpp) single header file, include it in your project.

```
#include "mtensor.hpp"
```

mtensor project is lite. when you find a bug, please fix it and pull a request.
