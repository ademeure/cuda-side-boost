// https://github.com/ademeure/cuda_side_boost
//
// Example of how to define a custom kernel with sideaware_alloc_only.cuh
// We define a custom kernel name then include sideaware_kernel.cuh
// This is different to the way sideaware.cu works with NVRTC
// (see example_llmc_rope.cuh for an example of the latter)
#define KERNEL_NAME side_aware_memcpy

typedef int o0;
typedef int i0;

struct unused {};
typedef unused o1, o2, o3, i1, i2, i3;

constexpr bool reverse_order = false; // maximise L2 hits with normal->reverse->normal->...
constexpr bool input_evict[4] = {1,0,0,0}; // do not keep inputs in L2 (more space for outputs)

__device__ void elementwise_op(size_t element_idx, int sideband,
                               o0 &out0, o1 &out1, o2 &out2, o3 &out3,
                               const i0 &in0, const i1 &in1, const i2 &in2, const i3 &in3) {
    out0 = in0;
}

#include "sideaware_kernel.cuh"
