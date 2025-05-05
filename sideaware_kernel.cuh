// ============================================================================
// L2 Side Aware Multi-Input / Multi-Output Elementwise kernel
// 16B alignment is required (even when it works, it's slower, so why bother?)
// Most efficient with 2MiB aligned addresses and array sizes of less than 2GiB
// ============================================================================
#include <cuda_runtime.h>

#ifndef KERNEL_NAME
#define KERNEL_NAME sideaware_elementwise
#endif

#ifndef UNROLLED
#define UNROLLED 4 // unrolled loop iterations (increases register pressure especially with multiple inputs)
#endif
#ifndef LAUNCH_BOUNDS
#define LAUNCH_BOUNDS __launch_bounds__(1024, 1) // (1024,1) ==>> compiler must use less than 64 registers
#endif
#ifndef DISCARD_INPUTS
constexpr bool input_discard[4] = {0,0,0,0}; // danger: discards from L2 *before* data is written to DRAM
#endif

#ifndef FORCE_WRONG_SIDE
#define FORCE_WRONG_SIDE false
#endif
#ifndef FORCE_RANDOM_SIDE
#define FORCE_RANDOM_SIDE false
#endif

//#define DYNAMIC_INDEX_FOR_CONCURRENT_KERNELS // use atomics to dynamically assign SM side index
//#define CUSTOM_VECTOR_OP // custom vector_op() instead of elementwise_op()
//#define CUSTOM_IDX_FUNC

// Typically only need custom elementwise_op, but support custom vector_op if needed
// e.g. for BF16 -> MXFP8 with microscaling where we need the absmax over 32 elements
// (in theory, we have access to 512 bytes per warp and 4096 bytes per group with barriers)
#ifndef CUSTOM_VECTOR_OP
__device__ void vector_op(size_type vec_idx, int vec_size,
                          o0 *out0, o1 *out1, o2 *out2, o3 *out3,
                          const i0 *in0, const i1 *in1, const i2 *in2, const i3 *in3,
                          int* sideband_memory, int sideband_value) {
    for (int i = 0; i < vec_size; i++) {
        elementwise_op(vec_idx * vec_size + i, sideband_value,
                       out0[i], out1[i], out2[i], out3[i],
                       in0[i], in1[i], in2[i], in3[i]);
    }
}
#endif

// This allows complex indexing (not 1:1) as long it can be determinally statically
// e.g. RoPE will have a dynamic non-linear index for i1 based on vec_idx/vec_size only
#ifndef CUSTOM_IDX_FUNC
__device__ bool indexing(size_type vec_idx, int vec_size,
                         size_type &idx_i0, size_type &idx_i1, size_type &idx_i2, size_type &idx_i3,
                         int* sideband_memory, int sideband_value) {
    idx_i0 = vec_idx;
    idx_i1 = vec_idx;
    idx_i2 = vec_idx;
    idx_i3 = vec_idx;
    return false; // do not skip
}
#endif

// Packed structure of arrays to turn the user's scalar types into vectors
// The largest vector will automatically be 16 bytes (i0 and/or o1) and others will be 16B/8B/4B
// This allows us to automatically use the GPU's vector loads (e.g. LDG.128) and stores (STG.128)
template<typename T, size_t N>
struct packed {
    T data[N];
    __device__ __forceinline__ T& operator[](int index) { return data[index]; }
    __device__ __forceinline__ const T& operator[](int index) const { return data[index]; }
    __device__ __forceinline__ T* operator+() { return data; }
    __device__ __forceinline__ const T* operator+() const { return data; }
};

// The kernel is "side aware" for i0 by default unless o1 is larger than i0 (more bytes per element)
// All inputs and outputs of that same size will be processed "side aware"
// assuming (ptr0 % 4096*1024*1024) == (ptr1 % 4096*1024*1024)
// This is guaranteed for memory allocations of the same size in our custom allocator (sideaware.cu)
constexpr size_t vector_bytes = 16;
constexpr bool sideaware_for_o0 = sizeof(o0) > sizeof(i0);
constexpr int vec_size = vector_bytes / (sideaware_for_o0 ? sizeof(o0) : sizeof(i0));

typedef packed<o0, vec_size> vo0;
typedef packed<o1, vec_size> vo1;
typedef packed<o2, vec_size> vo2;
typedef packed<o3, vec_size> vo3;
typedef packed<i0, vec_size> vi0;
typedef packed<i1, vec_size> vi1;
typedef packed<i2, vec_size> vi2;
typedef packed<i3, vec_size> vi3;

// Automatically use vector loads if the type is 16B/8B/4B
// Assumes pointer is aligned (otherwise it's an illegal memory access)
// Supports cache streaming to evict the data from L2 (leaving more space for the write data)
template<bool evict = false, typename T>
__device__ __forceinline__ T load(const T * __restrict__ src) {
    if constexpr (sizeof(T) == 16) {
        int4 data = evict ? __ldcs((const int4*)src) : __ldcg((const int4*)src);
        return *(T*)&data;
    } else if constexpr (sizeof(T) == 8) {
        int2 data = evict ? __ldcs((const int2*)src) : __ldcg((const int2*)src);
        return *(T*)&data;
    } else if constexpr (sizeof(T) == 4) {
        int data = evict ? __ldcs((const int*)src) : __ldcg((const int*)src);
        return *(T*)&data;
    }
    return *src;
}

// Automatically use vector stores (assumes pointer is aligned, otherwise it's an illegal memory access)
template<bool writeback = true, typename T> // TODO: cache streaming support for outputs
__device__ __forceinline__ void store(T* __restrict__ ptr, const T &value) {
    T* aligned_ptr = (T*)__builtin_assume_aligned(ptr, sizeof(T));
    aligned_ptr[0] = value;
}

template<typename T, typename U>
struct is_same { static constexpr bool value = false; };
template<typename T>
struct is_same<T, T> { static constexpr bool value = true; };

// Advanced and extremely dangerous: potentially discards inputs from L2 *and* DRAM
// Useful if the inputs were written by the previous kernel and will nevber be used again
// If the inputs and outputs are the same size, reuse the same memory for inputs and outputs instead
// But this is useful if the sizes are different, e.g. converting from BF16 to FP8 after a matrix multiplication
__device__ __forceinline__ void discard_inputs(
    const vi0 * input0, const vi1 * input1, const vi2 * input2, const vi3 * input3) {
    if (input_discard[0] && !is_same<i0, unused>::value && (unsigned long long)input0 % 128 == 0)
        asm volatile("discard.global.L2 [%0], 128;\n" : : "l"(input0));
    if (input_discard[1] && !is_same<i1, unused>::value && (unsigned long long)input1 % 128 == 0)
        asm volatile("discard.global.L2 [%0], 128;\n" : : "l"(input1));
    if (input_discard[2] && !is_same<i2, unused>::value && (unsigned long long)input2 % 128 == 0)
        asm volatile("discard.global.L2 [%0], 128;\n" : : "l"(input2));
    if (input_discard[3] && !is_same<i3, unused>::value && (unsigned long long)input3 % 128 == 0)
        asm volatile("discard.global.L2 [%0], 128;\n" : : "l"(input3));
}

// MAX_SM is set dynamically at compile time to avoid passing more parameters than necessary
typedef struct {
    unsigned char side_index[MAX_SM];
} param_sm_side_t;

// ---------------------------------------------------------------------------
// Kernel definition
// ---------------------------------------------------------------------------
extern "C" __global__ LAUNCH_BOUNDS void KERNEL_NAME(
        size_type num_vectors_16B,
        vo0* __restrict__ output0, vo1* __restrict__ output1,
        vo2* __restrict__ output2, vo3* __restrict__ output3,
        const vi0 * __restrict__ input0, const vi1 * __restrict__ input1,
        const vi2 * __restrict__ input2, const vi3 * __restrict__ input3,
        int* mem, int val,
        unsigned int num_sm_per_side,
        unsigned int* zeroed_scratch_buffer,
        const __grid_constant__ param_sm_side_t params) {

    // Decode side and index from a single byte (assumes less than 256 SMs for static side index case)
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
    unsigned int sm_side = params.side_index[smid] & 1;
    unsigned int sm_side_index;

    // Static side index (determined at init time at the same time as detecting the side of each SM)
    // Only works if all SMs are available and gridDim == SM count, otherwise some data might be skipped
    sm_side_index = params.side_index[smid] >> 1;

    #ifdef DYNAMIC_INDEX_FOR_CONCURRENT_KERNELS
    // Dynamic side index using atomics (always works and only significantly slower when static would fail)
    __shared__ unsigned int shared_sm_side_index;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sm_side_index = atomicInc(&zeroed_scratch_buffer[sm_side], 999);
        if (sm_side_index >= num_sm_per_side) {
            sm_side = 1 - sm_side; // need to process the 'wrong side' or no one will (slower but safer)
            sm_side_index = atomicInc(&zeroed_scratch_buffer[sm_side], 999);
        }
        unsigned int total_both_sides = atomicInc(&zeroed_scratch_buffer[2], 999);
        if (total_both_sides >= gridDim.x - 1) {
            // Reset the counter for the next kernel launch
            // (cycle through multiple buffers to support kernels running in parallel)
            zeroed_scratch_buffer[0] = 0;
            zeroed_scratch_buffer[1] = 0;
            zeroed_scratch_buffer[2] = 0;
        }
        shared_sm_side_index = sm_side_index;
    }
    __syncthreads();
    sm_side_index = shared_sm_side_index;
    #endif

    if (sm_side_index >= num_sm_per_side) {
        return;
    }

    // Side calculation: if 2MiB aligned, we don't need a base if we adjust the side based on bit 21 (4MiB unaligned)
    unsigned int base = (sideaware_for_o0 ? (unsigned long long)output0 : (unsigned long long)input0) & 0xFFFFFFFFULL;
    if constexpr (aligned_2MiB) {
        if (base & (2*1024*1024)) {
            sm_side ^= 1;
        }
        // This is enough for the compiler to optimize everything else away
        base = 0;
    } else {
        base /= vector_bytes;
    }

    // Chunks
    constexpr int CHUNK_OFFSET = 4096 / vector_bytes;
    constexpr int DOUBLE_CHUNK_OFFSET = 2 * CHUNK_OFFSET;
    constexpr size_type MULTI_CHUNK_OFFSET = UNROLLED * DOUBLE_CHUNK_OFFSET;
    int num_double_chunks = num_vectors_16B / (size_type)DOUBLE_CHUNK_OFFSET;
    int multi_chunks = num_double_chunks / UNROLLED;

    // Groups and indices
    // Up to 4 groups (blockDim.y) of 256 threads each (blockDim.x)
    // Each group is responsible for processing  of the output
    int group = threadIdx.y;
    int group_tid_offset = threadIdx.x;
    int groups_per_side = num_sm_per_side * blockDim.y;
    int global_idx = (sm_side_index * blockDim.y) + group;
    int adjusted_global_idx = reverse_order ? (multi_chunks - global_idx - 1) : global_idx;

    // Offsets (vi0 and/or vo0 are 16B so +1 element corresponds to +16 bytes)
    size_type global_offset = adjusted_global_idx * MULTI_CHUNK_OFFSET + group_tid_offset;
    size_type offset_increment_outer = (MULTI_CHUNK_OFFSET * groups_per_side) * (reverse_order ? -1 : 1);

    // Loop variables (worst case register pressure is very roughly all of this live at the same time)
    size_type offsets[UNROLLED];
    bool skip[UNROLLED];
    vi0 in0[UNROLLED];
    vi1 in1[UNROLLED];
    vi2 in2[UNROLLED];
    vi3 in3[UNROLLED];
    vo0 out0;
    vo1 out1;
    vo2 out2;
    vo3 out3;

    #pragma unroll 1
    for (int i = global_idx; i < multi_chunks; i += groups_per_side, global_offset += offset_increment_outer) {
        #pragma unroll UNROLLED
        for (int j = 0; j < UNROLLED; j++) {
            size_type offset = global_offset + (j * DOUBLE_CHUNK_OFFSET);

            // Determine the side of the 1st 4KiB chunk in this 8KiB "double chunk"
            // In theory, all threads in a group will have the same result for the 2MiB aligned case
            // But the generated SASS is *extremely* efficient so probably not worth the synchronization or complexity
            unsigned int lsb_bits = base + (offset & 0xFFFFFFFF);
            unsigned int side = __popc(lsb_bits & HASH) & 1;
            if constexpr (FORCE_WRONG_SIDE) side ^= 1;
            if constexpr (FORCE_RANDOM_SIDE) side = 0;

            // Switch to the 2nd 4KiB chunk in a 8KiB "double chunk" if the 1st 4KiB is on the other side
            // The paired SM from the other side will be responsible for the opposite 4KiB chunk
            // (since these two 4KiB chunks are always on opposite sides)
            // This is an optimized version of: "if (side != sm_side) byte_offset += CHUNK_SIZE;"
            unsigned int use_second_chunk = sm_side ^ side;
            offset += (use_second_chunk * CHUNK_OFFSET);

            offsets[j] = offset;
            size_type offsets_i[4] = {0};
            skip[j] = indexing(offset, vec_size, offsets_i[0], offsets_i[1], offsets_i[2], offsets_i[3], mem, val);

            if (skip[j] == false) {
                in0[j] = load<input_evict[0]>(input0 + offsets_i[0]);
                in1[j] = load<input_evict[1]>(input1 + offsets_i[1]);
                in2[j] = load<input_evict[2]>(input2 + offsets_i[2]);
                in3[j] = load<input_evict[3]>(input3 + offsets_i[3]);
            }
        }

        // Execute operation and store results *after* the loads for every UNROLLED iteration
        // In theory, the compiler should be able to reorder things to maximise memory parallelism either way...
        // In practice, this is the only way to get it to reliably do the right thing (all indexing is optimized away)
        #pragma unroll UNROLLED
        for (int j = 0; j < UNROLLED; j++) {
            if (skip[j] == false) {
                size_type offset = offsets[j];
                vector_op(offset, vec_size, +out0, +out1, +out2, +out3, +in0[j], +in1[j], +in2[j], +in3[j], mem, val);
                store(output0 + offset, out0);
                store(output1 + offset, out1);
                store(output2 + offset, out2);
                store(output3 + offset, out3);
            }
        }

        // Optional discard needs to happen after the stores otherwise performance is extremely bad (due to fences?)
        #ifdef DISCARD_INPUTS
        #pragma unroll UNROLLED
        for (int j = 0; j < UNROLLED; j++) {
            if (skip[j] == false) {
                size_type offsets_i[4] = {0};
                indexing(offsets[j], vec_size, offsets_i[0], offsets_i[1], offsets_i[2], offsets_i[3], mem, val);
                discard_inputs(input0+offsets_i[0], input1+offsets_i[1], input2+offsets_i[2], input3+offsets_i[3]);
            }
        }
        #endif
    }

    // Handle everything that isn't a multiple of MULTI_CHUNK_OFFSET (i.e. 32KiB with UNROLLED = 4)
    // Try to process this on the 'last SMs' that are most likely to require one less 'full' iteration
    // Worst case we need 8 SMs (4 per side) to process 32KiB so we don't need more than 1 group per SM
    if (group == 0) {
        int max_remaining_double_chunks = (num_double_chunks + 1) - (multi_chunks * UNROLLED);
        int start_sm_side_idx = num_sm_per_side - max_remaining_double_chunks;
        int idx = sm_side_index - start_sm_side_idx;
        if (idx >= 0) {
            size_type offset = (size_type)(idx + multi_chunks * UNROLLED) * DOUBLE_CHUNK_OFFSET;
            offset += group_tid_offset;

            unsigned int lsb_bits = base + (offset & 0xFFFFFFFF);
            unsigned int side = __popc(lsb_bits & HASH) & 1;
            unsigned int use_second_chunk = sm_side ^ side;
            offset += (use_second_chunk * CHUNK_OFFSET);

            if (offset < num_vectors_16B) {
                size_type offsets_i[4] = {0};
                skip[0] = indexing(offset, vec_size, offsets_i[0], offsets_i[1], offsets_i[2], offsets_i[3], mem, val);
                if (skip[0]) {
                    return;
                }

                in0[0] = load<input_evict[0]>(input0 + offsets_i[0]);
                in1[0] = load<input_evict[1]>(input1 + offsets_i[1]);
                in2[0] = load<input_evict[2]>(input2 + offsets_i[2]);
                in3[0] = load<input_evict[3]>(input3 + offsets_i[3]);

                vector_op(offset, vec_size, +out0, +out1, +out2, +out3, +in0[0], +in1[0], +in2[0], +in3[0], mem, val);
                store(output0 + offset, out0);
                store(output1 + offset, out1);
                store(output2 + offset, out2);
                store(output3 + offset, out3);
                #ifdef DISCARD_INPUTS
                discard_inputs(input0+offsets_i[0], input1+offsets_i[1], input2+offsets_i[2], input3+offsets_i[3]);
                #endif
            }
        }
    }
}
