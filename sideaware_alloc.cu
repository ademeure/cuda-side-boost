// ============================================================================
// Subset of sideaware.cu memory allocation for a single GPU (not fully tested)
// This is all you need to integrate for your custom CUDA project (non-PyTorch)
// For kernels, look at sideaware_memcpy.cu which includes sideaware_kernel.cuh
// https://github.com/ademeure/cuda_side_boost
// ============================================================================
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>

// ---------------------------------------------------------------------------
// Compile Time Settings
// ---------------------------------------------------------------------------
//#define DEBUG_PRINTF
constexpr bool   SYNC_ON_EVERY_ALLOC = false;  // sync when (pre)allocating memory to avoid confusing async errors
constexpr bool   ZERO_ON_PREALLOC = false;     // zero out memory when preallocating physical memory

constexpr int    MAX_SM = 640;                 // 640 SMs ought to be enough for anybody (kernels use real SM count)
constexpr int    FORCED_HASH = 0;              // 0xB3000 for H100, 0xAB000 for GH200 96GiB, 0x1EF000 for GB200
constexpr int    L2_SIDE_TEST_ITERATIONS = 25; // increases warmup time (at init & per page) but improves accuracy
constexpr int    SIDE_HASH_BIT = 21;           // we force bit 21 of the virtual address to track the side of the page
constexpr size_t CHUNK_BYTES = 4096;           // 4KiB (= granularity of side switch on H100/GB200)
constexpr size_t PAGE_SIZE  = 2 * 1024 * 1024; // 2MiB (= granularity of NVIDIA MMU pages since Pascal)
constexpr size_t UNBACKED_VIRTUAL_PAGES = 2048ULL * 1024ULL * 1024ULL / PAGE_SIZE; // maximum we allocate in one go

// Offsets into the cpu_side_info array
constexpr int OFFSET_ZEROED_COUNTER    = MAX_SM;
constexpr int OFFSET_AVERAGE_LATENCY   = MAX_SM + 1;
constexpr int OFFSET_SIDE_HASH_MASK    = MAX_SM + 2;
constexpr int OFFSET_NUM_SM_SIDE0      = MAX_SM + 3;
constexpr int OFFSET_NUM_SM_SIDE1      = MAX_SM + 4;
constexpr int OFFSET_MIN_SM_PER_SIDE   = MAX_SM + 5;

// ---------------------------------------------------------------------------
// Global configuration variables
// ---------------------------------------------------------------------------
static bool g_use_compression = false; // Enable CUDA compressible memory (release unused memory to avoid mixed reuse)
static size_t g_free_mapped_start_threshold = 16ULL * 1024ULL * 1024ULL * 1024ULL; // auto unmap on malloc above this
static size_t g_free_mapped_end_threshold   = 2ULL  * 1024ULL * 1024ULL * 1024ULL; // stop auto unmapping at this point
static int g_prealloc_extra_required = 1, g_prealloc_extra_alloc = 10; // configured by set_prealloc_config()

// ---------------------------------------------------------------------------
// Everything needed for side-aware page-based allocation
// ---------------------------------------------------------------------------
struct LargeAllocInfo {
    size_t user_requested   = 0;
    void*  base_ptr         = nullptr;  // Points to the correctly aligned section for mapping
    void*  alloc_ptr        = nullptr;  // Original pointer from cuMemAddressReserve
    size_t aligned_size     = 0;
    bool   use_compression  = false;
    cudaStream_t last_use_stream = 0;
    std::vector<CUmemGenericAllocationHandle> handles; // One handle per 2MiB page
    // Track which side was actually used for each page so we know where to return it:
    std::vector<int> side_used; // side (0 or 1) used for each page (returns to that pool on free)
};

// ---------------------------------------------------------------------------
// Device Context structure to track per-device data
// ---------------------------------------------------------------------------
typedef struct {
    unsigned char side_index[MAX_SM]; // struct to pass this as a single argument to the GPU
} ParamSmSide;

struct DeviceContext {
    // Device properties
    bool initialized = false;
    int num_sms = 0;

    // CPU memory (used on CPU or passed as kernel parameter)
    int cpu_side_info[MAX_SM * 2];
    ParamSmSide param_sm_side;
    int side_summary[5];

    // GPU memory for side info
    unsigned int* gpu_allocator_metadata = nullptr;
    unsigned int* gpu_side_info = nullptr;
    unsigned char* gpu_side_index = nullptr;
    unsigned char* gpu_scratch_buffer = nullptr;
    unsigned char* gpu_scratch_zeroed_buffer = nullptr;
    // Unbacked virtual memory tracking
    unsigned char cpu_page_side[UNBACKED_VIRTUAL_PAGES];

    // Memory pools
    std::vector<CUmemGenericAllocationHandle> free_handles_side[2];
    std::unordered_map<size_t, std::vector<LargeAllocInfo>> large_alloc_cache;
    std::unordered_map<void*, LargeAllocInfo> large_alloc_registry;

    // State tracking
    size_t total_mapped_free = 0;
    bool cross_side_warning_issued = false;
    bool compression_available = false;
    std::unordered_map<size_t, int> size_side_map;

    // Initialize the context for a specific device
    void initialize();
};

// ---------------------------------------------------------------------------
// Helper Functions & Classes
// ---------------------------------------------------------------------------
template<class... Ts>
__device__ __host__ static void debugf(const char* fmt, Ts... args) {
#ifdef DEBUG_PRINTF
    std::printf(fmt, args...);
#endif
}

// ---------------------------------------------------------------------------
// Device functions & kernels for side info
// See the following example for simpler code with more comments:
// https://github.com/ademeure/QuickRunCUDA/blob/main/tests/side_aware.cu
// ---------------------------------------------------------------------------
__device__ __forceinline__ int test_latency_l2(unsigned int* data, size_t offset) {
    unsigned int old_value = atomicExch(&data[offset], 0); // also warms up the cache!
    long long int start_clock = clock64();
    for (int i = 0; i < L2_SIDE_TEST_ITERATIONS; i++) {
        int value = atomicInc(&data[offset], 99999999);
        offset += (value > L2_SIDE_TEST_ITERATIONS*10) ? 1 : 0;
    }
    int latency = clock64() - start_clock;
    data[offset] = old_value;
    return latency;
}

__global__ void init_side_info(unsigned int* base_page, unsigned int *side_info, unsigned char *side_index) {
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);

        int offset = 4 * smid;
        assert(offset * sizeof(int) < CHUNK_BYTES);

        __nanosleep(((smid+4)% 16) * 2000 + 100);
        int total_latency = test_latency_l2(base_page, offset);
        side_info[smid] = total_latency;
        atomicAdd(&side_info[OFFSET_AVERAGE_LATENCY], total_latency);

        int num_done = atomicInc(&side_info[OFFSET_ZEROED_COUNTER], gridDim.x - 1);
        if (num_done == gridDim.x - 1) {
            int average_latency = side_info[OFFSET_AVERAGE_LATENCY] / gridDim.x;
            debugf("Average L2-latency threshold: %.1f\n", (float)average_latency / (float)L2_SIDE_TEST_ITERATIONS);

            // SM0 is always side 0 (everything else is relative to it)
            int far_side =  (side_info[0] > average_latency) ? 0 : 1;
            int near_side = (side_info[0] > average_latency) ? 1 : 0;

            int side0_counter = 0, side1_counter = 0;
            for (int i = 0; i < gridDim.x; i++) {
                int latency = side_info[i];
                side_info[i] = (latency > average_latency) ? far_side : near_side;
                if ((side_info[i] & 1) == 0) {
                    side_info[i] |= (side0_counter++) << 1;
                } else {
                    side_info[i] |= (side1_counter++) << 1;
                }
                side_index[i] = (unsigned char)side_info[i];
                debugf("[SM %3d] L2-latency = %.1f -> side=%d idx=%d\n",
                       i, (float)latency / (float)L2_SIDE_TEST_ITERATIONS,
                       (side_info[i] & 1), (side_info[i] >> 1));
            }
            side_info[OFFSET_AVERAGE_LATENCY] = average_latency;
            side_info[OFFSET_NUM_SM_SIDE0]    = side0_counter;
            side_info[OFFSET_NUM_SM_SIDE1]    = side1_counter;
            side_info[OFFSET_MIN_SM_PER_SIDE] = min(side0_counter, side1_counter);

            if constexpr (FORCED_HASH == 0) {
                unsigned long long int addr_int = reinterpret_cast<unsigned long long int>(base_page);
                if (addr_int % ((size_t)PAGE_SIZE) != 0) {
                    debugf("ERROR: base_page not 2MiB-aligned\n");
                    assert(false);
                    return;
                }
                int base_side = side_info[smid] & 1;
                int check_start_bit = 4;
                int check_last_bit  = 20;
                int toggle_bits = 0;
                for (int i = check_start_bit; i <= check_last_bit; i++) {
                    int bitmask = 1 << i;
                    int offset2 = bitmask / sizeof(int);
                    int total_latency2 = test_latency_l2(base_page, offset2);
                    int offset_side = (total_latency2 > average_latency)? far_side : near_side;
                    if (offset_side != base_side) {
                        toggle_bits |= bitmask;
                    }
                }
                side_info[OFFSET_SIDE_HASH_MASK] = toggle_bits;
                debugf("Detected side-hash bits: 0x%X\n", toggle_bits);
                if (!(toggle_bits & CHUNK_BYTES) || (toggle_bits & (CHUNK_BYTES - 1))) {
                    printf("\nERROR: CHUNK_BYTES %d doesn't work with hash %x\n\n", (int)CHUNK_BYTES, toggle_bits);
                    assert(false);
                }
            } else {
                side_info[OFFSET_SIDE_HASH_MASK] = FORCED_HASH;
            }
        }
    } else if (threadIdx.x >= 32) {
        __nanosleep(10000);
    }
}

__global__ void test_page_latency(unsigned int* ptr, unsigned int *side_info, unsigned char* page_side, int num_pages)
{
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
        int near_side = side_info[smid] & 1;
        int far_side  = 1 - near_side;
        int average_latency = side_info[OFFSET_AVERAGE_LATENCY];

        debugf("Testing %u pages.\n", num_pages);
        for (int i = 0; i < num_pages; i++) {
            size_t offset = (size_t)i * (PAGE_SIZE / sizeof(unsigned int));
            int total_latency = test_latency_l2(ptr, offset);
            page_side[i] = (total_latency > average_latency) ? far_side : near_side;
            debugf("[SM %3d] Page %3d: L2-latency = %.1f (raw: %u) -> side=%d\n",
                   smid, i, (float)total_latency / (float)L2_SIDE_TEST_ITERATIONS, total_latency, page_side[i]);
        }
    }
}

// Device contexts array - indexed by device ID
static DeviceContext g_deviceContext;
static DeviceContext& get_device_context() {
    if (!g_deviceContext.initialized) {
        g_deviceContext.initialize();
    }
    return g_deviceContext;
}

static void init_allocation_constraints(DeviceContext& ctx)
{
    int comp_available;
    cuDeviceGetAttribute(&comp_available, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, 0);
    ctx.compression_available = (comp_available != 0);

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    size_t granularity;
    CUresult res = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(res == CUDA_SUCCESS && granularity == PAGE_SIZE); // Verify granularity equals PAGE_SIZE

    if (ctx.compression_available) {
        prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
        res = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        assert(res == CUDA_SUCCESS && granularity == PAGE_SIZE);
    }
}

void DeviceContext::initialize() {
    if (initialized) return;

    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    init_allocation_constraints(*this);
    cudaMalloc(&gpu_allocator_metadata, PAGE_SIZE);
    cudaMemset(gpu_allocator_metadata, 0, PAGE_SIZE);
    assert((uintptr_t)gpu_allocator_metadata % PAGE_SIZE == 0);

    gpu_side_info = &gpu_allocator_metadata[16 * 1024];
    gpu_side_index = (unsigned char*)&gpu_allocator_metadata[32 * 1024];
    gpu_scratch_buffer = (unsigned char*)&gpu_allocator_metadata[64 * 1024];
    gpu_scratch_zeroed_buffer = (unsigned char*)&gpu_allocator_metadata[128 * 1024];

    init_side_info<<<num_sms, 512, 0, 0>>>(gpu_allocator_metadata, gpu_side_info, gpu_side_index);

    unsigned char cpu_side_index[MAX_SM];
    cudaMemcpyAsync(cpu_side_index, gpu_side_index, MAX_SM * sizeof(unsigned char), cudaMemcpyDeviceToHost, 0);
    cudaMemcpy(cpu_side_info, gpu_side_info, MAX_SM * 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost); // syncs both

    for (int i = 0; i < MAX_SM; i++) {
        param_sm_side.side_index[i] = cpu_side_index[i];
    }

    initialized = true;
}

// ---------------------------------------------------------------------------
// Unmap free blocks if threshold is exceeded
// ---------------------------------------------------------------------------
static void unmap_free_allocations(DeviceContext& ctx, size_t start_threshold=0, size_t end_threshold=0) {
    if (ctx.total_mapped_free <= start_threshold) return;
    cudaDeviceSynchronize();

    for (auto &kv : ctx.large_alloc_cache) {
        auto &vec = kv.second;
        auto it = vec.begin();

        while (it != vec.end() && ctx.total_mapped_free > end_threshold) {
            LargeAllocInfo &blk = *it;
            size_t size = blk.aligned_size;
            size_t num_pages = blk.aligned_size / PAGE_SIZE;
            CUdeviceptr base = (CUdeviceptr)blk.base_ptr;

            for (size_t i = 0; i < num_pages; i++) {
                cuMemUnmap(base + i * PAGE_SIZE, PAGE_SIZE);
                // Return handle to the side it was allocated from:
                int side = blk.side_used[i];
                ctx.free_handles_side[side].push_back(blk.handles[i]);
            }
            // Use alloc_ptr for freeing address space, not base_ptr
            cuMemAddressFree((CUdeviceptr)blk.alloc_ptr, blk.aligned_size + 2 * PAGE_SIZE);

            assert(ctx.total_mapped_free >= size);
            ctx.total_mapped_free -= size;
            it = vec.erase(it);
        }
        if (ctx.total_mapped_free <= end_threshold)
            break;
    }
    cudaDeviceSynchronize();
}

static size_t release_unused_memory() {
    DeviceContext& ctx = get_device_context();

    // Unmap all cached blocks then release handles from side0/side1 pools
    unmap_free_allocations(ctx);

    size_t freed_memory = 0;
    for (int side = 0; side < 2; side++) {
        for (auto &h : ctx.free_handles_side[side]) {
            cuMemRelease(h);
            freed_memory += PAGE_SIZE;
        }
        ctx.free_handles_side[side].clear();
    }

    cudaDeviceSynchronize();
    return freed_memory;
}

// ---------------------------------------------------------------------------
// KEY FUNCTION: This is where the magic happens!
// Allocate physical memory and determine which side it's on and stores the
// handles in per-side "free pools" for future mapping to virtual memory
// (pre-allocate more than required because it causes a GPU sync)
// ---------------------------------------------------------------------------
static CUresult sideaware_preallocate(DeviceContext& ctx, int countNeeded, bool compression, cudaStream_t stream)
{
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.allocFlags.compressionType = compression ? CU_MEM_ALLOCATION_COMP_GENERIC : 0;

    CUresult lastError = CUDA_SUCCESS;
    int totalAllocated = 0;

    // We'll do it in batches so we don't exceed UNBACKED_VIRTUAL_PAGES at once.
    while (countNeeded > 0) {
        int batch = std::min<int>(countNeeded, UNBACKED_VIRTUAL_PAGES);
        int batchAllocated = 0;

        // Reserve a *temporary* VA space for up to 'batch' pages (freed before returning)
        CUdeviceptr dptr;
        lastError = cuMemAddressReserve(&dptr, batch * PAGE_SIZE, 0, 0, 0);
        if (lastError != CUDA_SUCCESS) {
            break; // Can't continue this batch, but process what we have so far
        }

        // Create + map each handle:
        std::vector<CUmemGenericAllocationHandle> tempHandles(batch);
        for (int i = 0; i < batch; i++) {
            lastError = cuMemCreate(&tempHandles[i], PAGE_SIZE, &prop, 0);
            if (lastError != CUDA_SUCCESS) {
                tempHandles.resize(i);
                batch = i; // adjust batch size
                break;
            }

            lastError = cuMemMap(dptr + i * PAGE_SIZE, PAGE_SIZE, 0, tempHandles[i], 0);
            if (lastError != CUDA_SUCCESS) {
                cuMemRelease(tempHandles[i]);
                tempHandles.resize(i);
                batch = i;
                break;
            }
            batchAllocated++;
        }

        // If failed to allocate anything in this iteration, free the VA space and continue
        if (batchAllocated == 0) {
            cuMemAddressFree(dptr, batch * PAGE_SIZE);
            break;
        }

        // Set read/write access for everything we allocated (in the VA space)
        CUmemAccessDesc accessDesc;
        accessDesc.location.id   = prop.location.id;
        accessDesc.location.type = prop.location.type;
        accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        lastError = cuMemSetAccess(dptr, batchAllocated * PAGE_SIZE, &accessDesc, 1);
        if (lastError != CUDA_SUCCESS) {
            for (int i = 0; i < batchAllocated; i++) {
                cuMemUnmap(dptr + i * PAGE_SIZE, PAGE_SIZE);
                cuMemRelease(tempHandles[i]);
            }
            cuMemAddressFree(dptr, batch * PAGE_SIZE);
            break;
        }

        // Zero out memory before using it
        if constexpr (ZERO_ON_PREALLOC) {
            lastError = cuMemsetD8(dptr, 0, batchAllocated * PAGE_SIZE);
            if (lastError != CUDA_SUCCESS) {
                printf("ERROR: Failed to zero out memory before test (this should never happen?!)\n");
                assert(false);
                return lastError;
            }
        }

        // Classify each page (side 0 or 1) by calling test_page_latency
        test_page_latency<<<1, 512, 0, stream>>>((unsigned int*)dptr, ctx.gpu_side_info,
                                                 ctx.gpu_scratch_buffer, batchAllocated);
        cudaMemcpy(ctx.cpu_page_side, ctx.gpu_scratch_buffer, batchAllocated, cudaMemcpyDeviceToHost);

        // Unmap + place each handle in the correct side's pool
        for (int i = 0; i < batchAllocated; i++) {
            cuMemUnmap(dptr + i * PAGE_SIZE, PAGE_SIZE);
            int side = ctx.cpu_page_side[i];
            ctx.free_handles_side[side].push_back(tempHandles[i]);
        }
        cuMemAddressFree(dptr, batch * PAGE_SIZE);

        totalAllocated += batchAllocated;
        countNeeded -= batchAllocated;

        if (batchAllocated < batch) {
            break; // probably out of GPU memory?
        }
    }

    // If we allocated anything at all, consider it at least a partial success
    if (totalAllocated > 0) {
        return CUDA_SUCCESS;
    }
    return lastError;
}

constexpr inline int side_from_virtual_address(uint64_t va)
{
    return (int)((va >> SIDE_HASH_BIT) & 1ULL);
}

// ---------------------------------------------------------------------------
// KEY FUNCTION: Allocate by mapping preallocated physical memory (which we already know the side of)
// We want bit 21 of the virtual address to fully determine the base side of that 2MiB page
// So we try to map a free physical page from the correct side's pool if any is available
// Also we try to have every allocation of the same size have the same 4MiB alignment
// (so if you do a memcpy between two buffers of the same size, read & write sides will match)
// ---------------------------------------------------------------------------
static CUresult sideaware_allocate(DeviceContext& ctx, LargeAllocInfo &info, size_t size, bool compression)
{
    info.aligned_size = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
    info.use_compression = (compression && ctx.compression_available);

    size_t num_pages = info.aligned_size / PAGE_SIZE;
    CUresult rc = CUDA_SUCCESS;

    // Optional sync to make sure we catch all async errors before allocation
    if constexpr (SYNC_ON_EVERY_ALLOC) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            printf("ERROR: cudaDeviceSynchronize() failed before allocation, unrelated async error?\n");
            assert(false);
        }
    }

    // 1) Pre-check we have enough free handles, otherwise allocate more
    // This heuristic is a bit complicated and could probably be improved
    size_t needed_per_side = (num_pages + g_prealloc_extra_required) / 2;
    size_t free0 = ctx.free_handles_side[0].size();
    size_t free1 = ctx.free_handles_side[1].size();
    size_t needed_side0 = (needed_per_side > free0) ? (needed_per_side - free0) : 0;
    size_t needed_side1 = (needed_per_side > free1) ? (needed_per_side - free1) : 0;
    size_t needed_worst_case = max(needed_side0, needed_side1);
    size_t needed_both_sides = 2 * needed_worst_case;

    if (needed_both_sides > 0) {
        needed_both_sides += g_prealloc_extra_alloc;
        rc = sideaware_preallocate(ctx, needed_both_sides, compression, info.last_use_stream);
        if (rc != CUDA_SUCCESS) {
            return rc;
        }
    }

    // Determine the desired "start side" for this allocation size
    // If we haven't seen this size before, choose the side with fewer handles
    if (ctx.size_side_map.find(info.aligned_size) == ctx.size_side_map.end()) {
        int side0_free = ctx.free_handles_side[0].size();
        int side1_free = ctx.free_handles_side[1].size();
        ctx.size_side_map[info.aligned_size] = (side0_free >= side1_free) ? 0 : 1;
    }
    int desiredStartSide = ctx.size_side_map[info.aligned_size];

    // 2) Reserve VA space with EXTRA space (extra page = 2MiB)
    // This ensures we'll find a properly-aligned section with the desired side
    CUdeviceptr allocPtr = 0;
    size_t extraSize = info.aligned_size + PAGE_SIZE;
    rc = cuMemAddressReserve(&allocPtr, extraSize, 0, 0, 0);
    if (rc != CUDA_SUCCESS) {
        return rc;
    }

    info.alloc_ptr = reinterpret_cast<void*>(allocPtr);
    CUdeviceptr basePtr = allocPtr;

    // If the first page is not on the desired side, try the second page
    int firstPageSide = side_from_virtual_address((uint64_t)basePtr);
    if (firstPageSide != desiredStartSide) {
        basePtr += PAGE_SIZE; // Skip to next 2MiB page
    }

    // Verify that this page has the desired side
    int basePageSide = side_from_virtual_address((uint64_t)basePtr);
    if (basePageSide != desiredStartSide) {
        // This should never happen with our hash function using bit 21?! (assuming SIDE_HASH_BIT = 21)
        printf("ERROR: Failed to find desired side in 2 consecutive 2MiB pages\n");
        cuMemAddressFree(allocPtr, extraSize);
        return CUDA_ERROR_UNKNOWN;
    }

    // Set the basePtr to point to the section with the desired side
    info.base_ptr = reinterpret_cast<void*>(basePtr);
    info.handles.resize(num_pages);
    info.side_used.resize(num_pages);

    // 3) Map each page from whichever side is indicated by bit 21
    for (size_t i = 0; i < num_pages; i++) {
        uint64_t thisVa = (uint64_t)(basePtr + i * PAGE_SIZE);
        int desiredSide = side_from_virtual_address(thisVa);

        // pop from the correct side if available
        std::vector<CUmemGenericAllocationHandle>* correctPool = &ctx.free_handles_side[desiredSide];
        std::vector<CUmemGenericAllocationHandle>* otherPool   = &ctx.free_handles_side[1 - desiredSide];

        if (!correctPool->empty()) {
            info.handles[i] = correctPool->back();
            correctPool->pop_back();
            info.side_used[i] = desiredSide;
        } else {
            // Try to fallback to the other side
            if (otherPool->empty()) {
                // We've run out of GPU memory on both sides
                cuMemAddressFree(allocPtr, extraSize);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            // Warn the first time this happens
            if (!ctx.cross_side_warning_issued) {
                printf("WARNING: Cross-side handle usage on device 0!\n");
                ctx.cross_side_warning_issued = true;
            }
            info.handles[i] = otherPool->back();
            otherPool->pop_back();
            info.side_used[i] = desiredSide;
        }

        rc = cuMemMap(basePtr + i * PAGE_SIZE, PAGE_SIZE, 0, info.handles[i], 0);
        if (rc != CUDA_SUCCESS) {
            // Return the handle to whichever side we took it from
            ctx.free_handles_side[info.side_used[i]].push_back(info.handles[i]);
            // Cleanup on allocation failure
            for (size_t j = 0; j < i; j++) {
                cuMemUnmap(basePtr + j * PAGE_SIZE, PAGE_SIZE);
                ctx.free_handles_side[info.side_used[j]].push_back(info.handles[j]);
            }
            cuMemAddressFree(allocPtr, extraSize);
            return rc;
        }
    }

    // 4) Enable read/write access for the entire allocation in the VA space
    CUmemAccessDesc accessDesc;
    accessDesc.location.id   = 0;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    rc = cuMemSetAccess(basePtr, info.aligned_size, &accessDesc, 1);
    if (rc != CUDA_SUCCESS) {
        // Cleanup on allocation failure
        for (size_t i = 0; i < num_pages; i++) {
            cuMemUnmap(basePtr + i * PAGE_SIZE, PAGE_SIZE);
            ctx.free_handles_side[info.side_used[i]].push_back(info.handles[i]);
        }
        cuMemAddressFree(allocPtr, extraSize);
        return rc;
    }

    // 5) Optional sync to make sure we catch all async errors
    if constexpr (SYNC_ON_EVERY_ALLOC) {
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            // Cleanup on allocation failure
            for (size_t i = 0; i < num_pages; i++) {
                cuMemUnmap(basePtr + i * PAGE_SIZE, PAGE_SIZE);
                ctx.free_handles_side[info.side_used[i]].push_back(info.handles[i]);
            }
            cuMemAddressFree(allocPtr, extraSize);
            return CUDA_ERROR_UNKNOWN;
        }
    }
    return CUDA_SUCCESS;
}

static void* sideaware_reuse_allocation(DeviceContext& ctx, size_t alignedSize, cudaStream_t currentStream)
{
    auto &vec = ctx.large_alloc_cache[alignedSize + (int)g_use_compression];
    if (vec.empty()) return nullptr;

    LargeAllocInfo info = vec.back();
    assert(ctx.total_mapped_free >= info.aligned_size);
    ctx.total_mapped_free -= info.aligned_size;
    vec.pop_back();

    // If it's a different stream, we might need to synchronize
    // TODO: is this safe or do we need a full device synchronization?
    if (info.last_use_stream != currentStream) {
        cudaStreamSynchronize(info.last_use_stream);
        info.last_use_stream = currentStream;
    }

    // Insert back into active registry
    ctx.large_alloc_registry[info.base_ptr] = info;
    return info.base_ptr;
}

static void* sideaware_try_allocate(DeviceContext& ctx, size_t userSize, cudaStream_t stream)
{
    LargeAllocInfo info;
    info.user_requested = userSize;
    info.last_use_stream = stream;
    CUresult rc = sideaware_allocate(ctx, info, userSize, g_use_compression);
    if (rc != CUDA_SUCCESS)
        return nullptr;
    ctx.large_alloc_registry[info.base_ptr] = info;
    return info.base_ptr;
}

static void* sideaware_malloc_large(DeviceContext& ctx, size_t size, cudaStream_t stream)
{
    size_t alignedSize = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;

    // 1) Try exact-size reuse
    void* p = sideaware_reuse_allocation(ctx, alignedSize, stream);
    if (p) return p;

    // 2) Unmap free pages over configurable threshold (then try allocating new pages for the 1st time)
    unmap_free_allocations(ctx, g_free_mapped_start_threshold, g_free_mapped_end_threshold);
    if (p = sideaware_try_allocate(ctx, size, stream)) return p;

    // 3) Unmap all free pages & try again
    unmap_free_allocations(ctx);
    if (p = sideaware_try_allocate(ctx, size, stream)) return p;

    // 4) Try fully releasing all unused memory & try again (this shouldn't be required?)
    release_unused_memory();
    return sideaware_try_allocate(ctx, size, stream);
}

static cudaError_t sideaware_free_large(DeviceContext& ctx, void* ptr, cudaStream_t stream, bool try_cuda_free = false)
{
    if (!ptr) return cudaSuccess;
    auto it = ctx.large_alloc_registry.find(ptr);
    if (it == ctx.large_alloc_registry.end()) {
        return cudaErrorUnknown;
    }

    LargeAllocInfo info = it->second;
    ctx.large_alloc_registry.erase(it);
    ctx.large_alloc_cache[info.aligned_size + (int)info.use_compression].push_back(info);
    ctx.total_mapped_free += info.aligned_size;
    return cudaSuccess;
}


// ---------------------------------------------------------------------------
// PUBLIC API
// ---------------------------------------------------------------------------
extern "C" {

// Memory allocation functions
cudaError_t sideaware_malloc_async(void** devPtr, size_t size, cudaStream_t stream) {
    debugf("sideaware_malloc_async(%zu, %p)\n", size, stream);
    *devPtr = sideaware_malloc_large(get_device_context(), size, stream);
    return (*devPtr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t sideaware_free_async(void* ptr, cudaStream_t stream) {
    debugf("sideaware_free_async(%p, %p)\n", ptr, stream);
    return sideaware_free_large(get_device_context(), ptr, stream);
}

// Synchronous memory allocation (usually better to use async)
cudaError_t sideaware_malloc(void** devPtr, size_t size) {
    debugf("sideaware_malloc(%zu)\n", size);

    assert(cudaDeviceSynchronize() == cudaSuccess);
    *devPtr = sideaware_malloc_large(get_device_context(), size, 0);

    return (*devPtr) ? cudaSuccess : cudaErrorMemoryAllocation;
}
cudaError_t sideaware_free(void* ptr) {
    debugf("sideaware_free(%p)\n", ptr);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    return sideaware_free_large(get_device_context(), ptr, 0);
}
size_t sideaware_release_unused() {
    return release_unused_memory();
}

// Configuration functions
void sideaware_compression(bool enable) {
    g_use_compression = enable;
}
void sideaware_prealloc_config(int extra_required, int extra_alloc) {
    g_prealloc_extra_required = extra_required;
    g_prealloc_extra_alloc = extra_alloc;
}
void sideaware_free_mapped_thresholds(size_t start_threshold, size_t end_threshold) {
    g_free_mapped_start_threshold = start_threshold;
    g_free_mapped_end_threshold = end_threshold;
}

// Query functions
void sideaware_fill_side_index(unsigned char* gpu_array) {
    DeviceContext& ctx = get_device_context();
    assert(cudaMemcpy(gpu_array, ctx.gpu_side_index, ctx.num_sms, cudaMemcpyDeviceToDevice) == cudaSuccess);
}
const char* sideaware_gpu_side_index() {
    DeviceContext& ctx = get_device_context();
    return (const char*)ctx.gpu_side_index;
}
const char* sideaware_cpu_side_index() {
    DeviceContext& ctx = get_device_context();
    return (const char*)ctx.param_sm_side.side_index;
}
const int* sideaware_sm_side_summary()
{
    DeviceContext& ctx = get_device_context();
    ctx.side_summary[0] = ctx.num_sms;
    ctx.side_summary[1] = ctx.cpu_side_info[OFFSET_NUM_SM_SIDE0];
    ctx.side_summary[2] = ctx.cpu_side_info[OFFSET_NUM_SM_SIDE1];
    ctx.side_summary[3] = ctx.cpu_side_info[OFFSET_MIN_SM_PER_SIDE];
    ctx.side_summary[4] = ctx.cpu_side_info[OFFSET_SIDE_HASH_MASK] | (1 << SIDE_HASH_BIT);
    return ctx.side_summary;
}
int sideaware_num_sms() { return get_device_context().num_sms; }
int sideaware_num_sm_side0() { return get_device_context().cpu_side_info[OFFSET_NUM_SM_SIDE0]; }
int sideaware_num_sm_side1() { return get_device_context().cpu_side_info[OFFSET_NUM_SM_SIDE1]; }
int sideaware_min_sm_per_side() { return get_device_context().cpu_side_info[OFFSET_MIN_SM_PER_SIDE]; }
int sideaware_hash_mask() { return get_device_context().cpu_side_info[OFFSET_SIDE_HASH_MASK] | (1 << SIDE_HASH_BIT); }

} // extern "C"