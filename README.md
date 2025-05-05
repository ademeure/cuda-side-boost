# cuda-side-boost

CUDA L2 Side Boost makes it possible for the first time to optimize your CUDA kernels to reduce the amount of traffic between the two "sides" on NVIDIA Hopper & Blackwell GPUs.

- Allocate memory in a "L2 Side Aware" way (by microbenchmarking physical pages and remapping their virtual memory).
- Super Optimized & Easy "Elementwise Kernel Builder" to make your own custom kernels.
- The most efficient Hopper & Blackwell memcpy in existence (~10% lower power)!
- Easy to integrate in both PyTorch and raw CUDA C++

Kernel fusion is so last year! (okay, not really, but this is great for non-fused kernels)

## Custom Allocator for CUDA & PyTorch (Side Aware Virtual Memory)

CUDA C++
```cuda
sideaware_malloc_async(&inout, num_bytes); // side aware with 2MiB alignment

rope_kernel = sideaware_create_kernel(R"_(#include "kernel_rope.cuh")_");
sideaware_elementwise(rope_kernel, num_bytes,
                      inout, nullptr, nullptr, nullptr,    // 1 output
                      inout, freqs_cis, nullptr, nullptr); // 2 inputs

sideaware_free_async(d_inout) // caching allocator, reused if possible
```
PyTorch

```python
# Initialize and run a simple memcpy (see test.py for custom kernels)
sideaware_alloc = torch.cuda.memory.CUDAPluggableAllocator(path, 'sideaware_malloc_auto', 'sideaware_free_auto')
torch.cuda.memory.change_current_allocator(sideaware_alloc)
_lib = ctypes.CDLL(path)

# Define C-style function signatures
_lib.sideaware_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
_lib.sideaware_memcpy.restype = None

# Define PyTorch custom operations for memcpy:
def direct_register_custom_op(op_lib, op_name, op_func, mutates_args):
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    op_lib.define(op_name + schema_str)
    op_lib.impl(op_name, op_func, "CUDA")

# Allocate tensors
src_tensor = torch.arange(256*1024*1024, device="cuda", dtype=torch.float32)
dst_tensor = torch.zeros_like(in0)

# Run kernel (also compatible with torch.compatible)
torch.ops.sideaware.memcpy(dst_tensor, src_tensor)
```


## Elementwise Kernel Builder with L2 Side Optimization

For maximum performance, try to keep all pointers 2MiB aligned for the inputs & outputs of the optimized kernels.

Hard requirement of 16 byte alignment (still very fast if not aligned to 2MiB, it's just better)

Side Aware memcpy:
```cuda
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
```

RoPE:
```cuda
typedef float2 o0; // output pair
typedef float2 i0; // input pair
typedef float2 i1; // precomputed freqs

struct unused {};
typedef unused o1, o2, o3, i2, i3;

#define UNROLLED 2
constexpr bool reverse_order = false;
constexpr bool input_evict[4] = {1,0,0,0};

// ----------------------------------------------------------------------------

constexpr int T = 1024;
constexpr int n_head = 32;
constexpr int head_dim = 128;
constexpr int head_dim_half = head_dim / 2;

constexpr int query_heads = n_head;
constexpr int kv_heads = n_head;
constexpr int total_heads = query_heads + 2*kv_heads;

__device__ void elementwise_op(size_t element_idx, int sideband,
                               o0 &out0, o1 &out1, o2 &out2, o3 &out3,
                               const i0 &in0, const i1 &in1, const i2 &in2, const i3 &in3) {
    float x_real = in0.x;
    float x_imag = in0.y;
    float freqs_cos = in1.x;
    float freqs_sin = in1.y;

    out0.x = x_real * freqs_cos - x_imag * freqs_sin;
    out0.y = x_real * freqs_sin + x_imag * freqs_cos;
}

#define CUSTOM_IDX_FUNC
__device__ bool indexing(size_type vec_idx, int vec_size,
                         size_type &idx_i0, size_type &idx_i1,
                         size_type &idx_i2, size_type &idx_i3, int* _mem, int _val) {
    size_type idx_pair = vec_idx * vec_size;

    int head = (idx_pair / head_dim_half) % total_heads;
    bool skip = (head >= query_heads + kv_heads); // skip value head (inplace so don't need load/store)
    head -= (head >= query_heads) ? query_heads : 0; // adjust head index for key head

    int token = (idx_pair / (total_heads * head_dim_half)) % T;
    int head_pair_idx = idx_pair % head_dim_half;
    int freqs_pair_idx = token * head_dim_half + head_pair_idx;

    idx_i0 = vec_idx;
    idx_i1 = freqs_pair_idx / vec_size;

    return skip; // return here to help the compiler (early 'return true' results in worse code)
}
```

## Background & Timelines

- August 2024: llm.c experiments (never completely finished, mostly worked)
- September 2024: Briefly talked about it with NVIDIA engineers at CUDA MODE IRL
- February 2025: Discussions with Georg Kolling (mentioned in his GTC 2025 presentation)
- March 2025: Semianalysis Blackwell Hackathon (won 1st Place)
- March 2025: Fork of Deepseek's DeepGEMM with Side Aware B: https://github.com/ademeure/DeeperGEMM
- April 2025: Side Aware Reduction - https://github.com/ademeure/QuickRunCUDA/blob/main/tests/side_aware.cu
- May 2025: Release of "CUDA L2 Side Boost" - https://github.com/ademeure/cuda-side-boost

## Super Optimized SASS assembly

The code was aggressively optimized to achieve ~100% optimal assembly with minimal overhead for the L2 side calculation. This is the output for the inner loop of our memcpy kernel (4x unrolled with 32-bit indices and 2MiB aligned pointers).

There's literally not a single wasted instruction, it's a thing of beauty:
```sass
VIADD R29, R35.reuse, 0x200
LOP3.LUT R8, R35.reuse, 0x2b300, RZ, 0xc0, !PT
VIADD R31, R35.reuse, 0x400
VIADD R33, R35, 0x600
LOP3.LUT R9, R29, 0x2b300, RZ, 0xc0, !PT
LOP3.LUT R11, R31, 0x2b300, RZ, 0xc0, !PT
POPC R8, R8
LOP3.LUT R15, R33, 0x2b300, RZ, 0xc0, !PT
POPC R12, R9
LOP3.LUT R10, R5, 0x1, R8, 0x78, !PT
POPC R14, R11
IMAD R27, R10, 0x100, R35
LOP3.LUT R12, R5, 0x1, R12, 0x78, !PT
POPC R16, R15
IMAD.WIDE R8, R27, 0x10, R6
IMAD R29, R12, 0x100, R29
LOP3.LUT R14, R5, 0x1, R14, 0x78, !PT
LDG.E.EF.128 R8, desc[UR4][R8.64]
IMAD.WIDE R12, R29, 0x10, R6
IMAD R31, R14, 0x100, R31
LOP3.LUT R16, R5, 0x1, R16, 0x78, !PT
LDG.E.EF.128 R12, desc[UR4][R12.64]
IMAD R33, R16, 0x100, R33
IMAD.WIDE R16, R31, 0x10, R6
IMAD.WIDE R20, R33, 0x10, R6
LDG.E.EF.128 R16, desc[UR4][R16.64]
LDG.E.EF.128 R20, desc[UR4][R20.64]
IMAD.WIDE R26, R27, 0x10, R24
IMAD.WIDE R28, R29, 0x10, R24
IMAD.WIDE R30, R31, 0x10, R24
IMAD.WIDE R32, R33, 0x10, R24
IMAD.IADD R37, R36, 0x1, R37
ISETP.GE.AND P1, PT, R37, R4, PT
IMAD R35, R36, 0x800, R35
STG.E.128 desc[UR4][R26.64], R8
STG.E.128 desc[UR4][R28.64], R12
STG.E.128 desc[UR4][R30.64], R16
STG.E.128 desc[UR4][R32.64], R20
@!P1 BRA 0x1d0
```

This is the actual output of the NVIDIA compiler without any custom PTX or SASS: the idea was to structure the CUDA code to help the CUDA 12.8 compiler does a good job.

Needless to say, some extremely inefficient SASS was generated along the way, and this required an obscene amount of back-and-forth to achieve. It doesn't matter much for DRAM bandwidth limited kernels, but I still think it's pretty cool :)