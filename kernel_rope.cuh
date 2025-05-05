// Forward RoPE kernel (in-place)
// Powered by "CUDA L2 Side Boost"
// https://github.com/ademeure/cuda_side_boost

// This is an example of how to use the NVRTC path of sideaware.cu
// See example_llmc_rope.cu for the CPU code to run this

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
