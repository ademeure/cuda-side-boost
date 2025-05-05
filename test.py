# ---------------------------------------------------------------------------
# Extremely non-exhaustive test.py for sideaware.cu
# Somewhat useful as an example of how to use with PyTorch
# See DeeperGEMM and sideaware_wrapper.py for another example
# ---------------------------------------------------------------------------
# https://github.com/ademeure/cuda-side-boost
# ---------------------------------------------------------------------------
import torch
import ctypes

# ---------------------------------------------------------------------------
# Load sideaware.so library both directly and through CUDAPluggableAllocator
# ---------------------------------------------------------------------------
sideaware_alloc = torch.cuda.memory.CUDAPluggableAllocator('./sideaware.so', 'sideaware_malloc_auto', 'sideaware_free_auto')
torch.cuda.memory.change_current_allocator(sideaware_alloc)
lib = ctypes.CDLL('./sideaware.so')

lib.sideaware_create_kernel.argtypes = [ctypes.c_char_p]
lib.sideaware_create_kernel.restype = ctypes.c_int
lib.sideaware_sm_side_summary.argtypes = []
lib.sideaware_sm_side_summary.restype = ctypes.POINTER(ctypes.c_int * 5)
lib.sideaware_fill_side_index.argtypes = [ctypes.c_void_p]
lib.sideaware_fill_side_index.restype = None

lib.sideaware_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
lib.sideaware_memcpy.restype = None
lib.sideaware_one_to_one.argtypes = [ctypes.c_int, ctypes.c_size_t,     # kernel_id, num_bytes
                                     ctypes.c_void_p, ctypes.c_void_p,  # out0, in0
                                     ctypes.c_int, ctypes.c_void_p]     # device, stream
lib.sideaware_one_to_one.restype = None
lib.sideaware_elementwise.argtypes = [ctypes.c_int, ctypes.c_size_t,    # kernel_id, num_bytes
                                      ctypes.c_void_p, ctypes.c_void_p, # out0, out1
                                      ctypes.c_void_p, ctypes.c_void_p, # out2, out3
                                      ctypes.c_void_p, ctypes.c_void_p, # in0, in1
                                      ctypes.c_void_p, ctypes.c_void_p, # in2, in3
                                      ctypes.c_void_p, ctypes.c_void_p, # sideband_ptr, sideband_value
                                      ctypes.c_int, ctypes.c_int,       # parallel_chunks, forced_sm_per_side
                                      ctypes.c_int, ctypes.c_void_p]    # device, stream
lib.sideaware_elementwise.restype = None

# ---------------------------------------------------------------------------
# Wrappers for compiling sideaware kernels & fetching metadata
# ---------------------------------------------------------------------------
# Compile custom elementwise kernel (returns id for sideaware_elementwise)
def sideaware_create_kernel(header_code: bytes) -> int:
    return lib.sideaware_create_kernel(header_code)

# Returns (num_sms, num_side0, num_side1, min_sm_per_side, hash_mask)
def sideaware_summary():
    return tuple(lib.sideaware_sm_side_summary().contents)

# User friendly version of sideaware_summary()
def sideaware_summary_str():
    num_sms, num_side0, num_side1, min_side, hash = sideaware_summary()
    return { "num_sms": num_sms, "num_side0": num_side0, "num_side1": num_side1, "min_side": min_side, "hash": hash }

# Helper to get SM side & index as a PyTorch tensor
def sideaware_side_index_tensor():
    num_sms, _, _, _, _ = sideaware_summary()
    sides_tensor = torch.zeros(num_sms, dtype=torch.uint8, device="cuda")
    tensor_ptr = sides_tensor.data_ptr()
    lib.sideaware_fill_side_index(tensor_ptr)
    return sides_tensor

# ---------------------------------------------------------------------------
# PyTorch custom op wrappers for sideaware_[memcpy/one_to_one/elementwise]
# e.g. torch.ops.sideaware.memcpy(dst, src)
# ---------------------------------------------------------------------------
# Sideaware memcpy (i.e. "default kernel" when no custom kernel is provided via sideaware_compile)
def sideaware_memcpy(dst: torch.Tensor, src: torch.Tensor) -> None:
    # Validate inputs
    assert dst.device.type == "cuda" and src.device.type == "cuda", "Both tensors must be on CUDA"
    assert dst.dtype == src.dtype, "Source and destination must have the same dtype"
    assert dst.numel() >= src.numel(), "Destination tensor must be at least as large as source"

    # Get pointers and size
    dst_ptr = dst.data_ptr()
    src_ptr = src.data_ptr()
    num_bytes = src.numel() * src.element_size()

    # Make sure src and dst are contiguous and aligned
    assert src.is_contiguous(), "src must be contiguous"
    assert dst.is_contiguous(), "dst must be contiguous"
    assert (dst_ptr % 16 == 0) and (src_ptr % 16 == 0), "dst and src must be 16-byte aligned"

    device, stream = torch.cuda.current_device(), torch.cuda.current_stream()
    lib.sideaware_memcpy(dst_ptr, src_ptr, num_bytes, device, stream)

# Sideaware single-input / single-output elementwise API (simple version)
def sideaware_one_to_one(kernel_id: int, dst: torch.Tensor, src: torch.Tensor) -> None:
    # Validate inputs
    src_bytes = dst is not None and dst.numel() * dst.element_size() or 0
    dst_bytes = dst is not None and dst.numel() * dst.element_size() or 0
    num_bytes = max(src_bytes, dst_bytes)
    assert num_bytes > 0

    # Make sure src and dst are contiguous and aligned
    dst_ptr = dst is not None and dst.data_ptr() or 0
    src_ptr = src_bytes and src.data_ptr() or 0
    assert src is None or (src.is_contiguous() and src.device.type == "cuda"), "src must be contiguous"
    assert dst is None or (dst.is_contiguous() and dst.device.type == "cuda"), "dst must be contiguous"
    assert dst_ptr % 16 == 0 and src_ptr % 16 == 0, "dst and src must be 16-byte aligned"

    device, stream = torch.cuda.current_device(), torch.cuda.current_stream()
    lib.sideaware_one_to_one(kernel_id, num_bytes, dst_ptr, src_ptr, device, stream)

# Sideaware multi-input / multi-output elementwise API (advanced version)
def sideaware_elementwise(kernel_id: int,
                          out0: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor, out3: torch.Tensor,
                          in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, in3: torch.Tensor,
                          sideband_tensor: torch.Tensor = None, sideband_value: int = 0,
                          parallel_chunks: int = 0, forced_sm_per_side: int = 0) -> None:
    # Validate inputs
    src_bytes = out0 is not None and out0.numel() * out0.element_size() or 0
    dst_bytes = out0 is not None and out0.numel() * out0.element_size() or 0
    num_bytes = max(src_bytes, dst_bytes)
    assert num_bytes > 0

    # Make sure src and dst are contiguous and aligned
    out0_ptr = out0 is not None and out0.data_ptr() or 0
    out1_ptr = out1 is not None and out1.data_ptr() or 0
    out2_ptr = out2 is not None and out2.data_ptr() or 0
    out3_ptr = out3 is not None and out3.data_ptr() or 0
    in0_ptr = in0 is not None and in0.data_ptr() or 0
    in1_ptr = in1 is not None and in1.data_ptr() or 0
    in2_ptr = in2 is not None and in2.data_ptr() or 0
    in3_ptr = in3 is not None and in3.data_ptr() or 0
    sideband_ptr = sideband_tensor is not None and sideband_tensor.data_ptr() or 0

    assert in0 is None or (in0.is_contiguous() and in0.device.type == "cuda"), "in0 must be contiguous"
    assert in1 is None or (in1.is_contiguous() and in1.device.type == "cuda"), "in1 must be contiguous"
    assert in2 is None or (in2.is_contiguous() and in2.device.type == "cuda"), "in2 must be contiguous"
    assert in3 is None or (in3.is_contiguous() and in3.device.type == "cuda"), "in3 must be contiguous"
    assert out0 is None or (out0.is_contiguous() and out0.device.type == "cuda"), "out0 must be contiguous"
    assert out1 is None or (out1.is_contiguous() and out1.device.type == "cuda"), "out1 must be contiguous"
    assert out2 is None or (out2.is_contiguous() and out2.device.type == "cuda"), "out2 must be contiguous"
    assert out3 is None or (out3.is_contiguous() and out3.device.type == "cuda"), "out3 must be contiguous"
    assert out0_ptr % 16 == 0 and out1_ptr % 16 == 0 and out2_ptr % 16 == 0 and out3_ptr % 16 == 0, "16B alignment"
    assert in0_ptr % 16 == 0 and in1_ptr % 16 == 0 and in2_ptr % 16 == 0 and in3_ptr % 16 == 0, "16B alignment"

    device, stream = torch.cuda.current_device(), torch.cuda.current_stream()
    lib.sideaware_elementwise(kernel_id, num_bytes,
                              out0_ptr, out1_ptr, out2_ptr, out3_ptr,
                              in0_ptr, in1_ptr, in2_ptr, in3_ptr,
                              sideband_ptr, sideband_value, parallel_chunks, forced_sm_per_side, device, stream)

# Define PyTorch custom operations
def direct_register_custom_op(op_lib, op_name, op_func, mutates_args):
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    op_lib.define(op_name + schema_str)
    op_lib.impl(op_name, op_func, "CUDA")

sideaware_lib = torch.library.Library("sideaware", "FRAGMENT")
direct_register_custom_op(sideaware_lib, "memcpy", sideaware_memcpy, mutates_args=(["dst"]))
direct_register_custom_op(sideaware_lib, "one_to_one", sideaware_one_to_one, mutates_args=(["dst"]))
direct_register_custom_op(sideaware_lib, "elementwise", sideaware_elementwise,
                          mutates_args=(["out0", "out1", "out2", "out3", "sideband_tensor"]))

###############################################################################
##########                   Test Code Starts Here                   ##########
###############################################################################

print(f"SM sides metadata: {sideaware_summary_str()}")
print(f"SM sides tensor: {sideaware_side_index_tensor()}")

# Test sideaware memcpy
print("\nTesting sideaware memcpy:")
src_tensor = torch.ones((8, 8), dtype=torch.float32, device="cuda")
dst_tensor = torch.zeros((8, 8), dtype=torch.float32, device="cuda")
print(f"Before memcpy - src: {src_tensor[0, 0]} & {src_tensor[7, 7]}, dst: {dst_tensor[0, 0]} & {dst_tensor[7, 7]}")
torch.ops.sideaware.memcpy(dst_tensor, src_tensor)
print(f"After memcpy - src: {src_tensor[0, 0]} & {src_tensor[7, 7]}, dst: {dst_tensor[0, 0]} & {dst_tensor[7, 7]}")

# Test with torch.compile
@torch.compile(fullgraph=True)
def compiled_memcpy(dst, src):
    torch.ops.sideaware.memcpy(dst, src)
    return dst # We return dst here for convenience in the test function

src2 = torch.ones((8, 8), dtype=torch.float32, device="cuda") * 2.0
dst2 = torch.zeros((8, 8), dtype=torch.float32, device="cuda")
print(f"Before compiled memcpy - src: {src2[0, 0]} & {src2[7, 7]}, dst: {dst2[0, 0]} & {dst2[7, 7]}")
result2 = compiled_memcpy(dst2, src2)
print(f"After compiled memcpy - src: {src2[0, 0]} & {src2[7, 7]}, dst: {dst2[0, 0]} & {dst2[7, 7]}")

# ---------------------------------------------------------------------------
# Demonstrate dynamic header injection & recompilation via NVRTC
# ---------------------------------------------------------------------------
print("\nTesting NVRTC dynamic recompilation path (custom sideaware kernel)...")

header_code = b"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

typedef float o0;
typedef half o1;
typedef __nv_bfloat16 i0;
typedef float i1;

struct unused {};
typedef unused o2, o3, i2, i3;

__device__ void elementwise_op(size_t element_idx, int sideband,
                               o0 &out0, o1 &out1, o2 &out2, o3 &out3,
                               const i0 &in0, const i1 &in1, const i2 &in2, const i3 &in3)
{
    out0 = (o0)((float)in0 + (float)in1);
    out1 = (o1)((float)in1 * 10.0f);
}

// process end of array 1st (maximise L2 hits with normal->reverse->normal->...)
constexpr bool reverse_order = true;

// do not keep inputs in L2 (more space for outputs)
constexpr bool input_evict[4] = {1,1,1,1};
"""

in0_hdr = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8)
in1_hdr = torch.ones_like(in0_hdr) * 2
out0_hdr = torch.zeros_like(in1_hdr)
out1_hdr = torch.zeros_like(in1_hdr)

in0_hdr = in0_hdr.to(torch.bfloat16)
out1_hdr = out1_hdr.to(torch.float16)

kernel_id = sideaware_create_kernel(header_code)
torch.ops.sideaware.elementwise(kernel_id, out0_hdr, out1_hdr, None, None, in0_hdr, in1_hdr, None, None, None, 0, 0, 0)

# Verify correctness after recompilation
if not torch.all(out0_hdr == (in0_hdr+in1_hdr)) or not torch.all(out1_hdr == (in1_hdr*10.0)):
    print("!!!!! Incorrect output for custom elementwise kernel !!!!!")
    print(f"in0_hdr: {in0_hdr}")
    print(f"in1_hdr: {in1_hdr}")
    print(f"out0_hdr: {out0_hdr}")
    print(f"out1_hdr: {out1_hdr}")
else:
    print("Successfully compiled and executed the custom elementwise kernel")

# ---------------------------------------------------------------------------
# Benchmark and validate sideaware_memcpy
# TODO: more exhaustive tests including sideaware_elementwise
# ---------------------------------------------------------------------------
print("\n===== MANUAL MEMCPY TESTING =====")

def test_memcpy_correctness(shape, dtype=torch.float32, run_benchmark=False, start_idx=0):
    """Test correctness of sideaware_memcpy with tensors of given shape and dtype."""
    print(f"\nTesting shape={shape}, dtype={dtype}")

    # Calculate size in MB (not MiB, because bandwidth is typically GB/s rather than GiB/s)
    element_size = torch.empty(1, dtype=dtype).element_size()
    num_elements = 1
    for dim_size in shape:
        num_elements *= dim_size
    size_mb = num_elements * element_size / (1000 * 1000)
    print(f"Tensor size: {size_mb:.2f} MB")

    # Create source tensor with recognizable pattern
    src = torch.arange(num_elements, dtype=dtype, device="cuda").reshape(shape)

    # Create destination tensors for our method and PyTorch's built-in
    dst_manual = torch.zeros_like(src)
    dst_builtin = torch.zeros_like(src)

    # Warm-up CUDA
    torch.cuda.synchronize()

    # Copy using our manual memcpy
    if run_benchmark:
        for i in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            torch.ops.sideaware.memcpy(dst_manual[start_idx:], src[start_idx:])
            end.record()
            torch.cuda.synchronize()
            manual_time = start.elapsed_time(end)
            print(f"\tSideaware copy: {manual_time:.3f} ms")

            # Copy using PyTorch's built-in
            start.record()
            dst_builtin[start_idx:].copy_(src[start_idx:])
            end.record()
            torch.cuda.synchronize()
            builtin_time = start.elapsed_time(end)
            print(f"\tPyTorch's copy: {builtin_time:.3f} ms")
            print(f"\tRatio: {manual_time/builtin_time:.2f}x")
    else:
        # Just do the copies without timing
        torch.ops.sideaware.memcpy(dst_manual[start_idx:], src[start_idx:])
        dst_builtin[start_idx:].copy_(src[start_idx:])

    # Check correctness
    is_equal = torch.all(dst_manual == dst_builtin).item()
    print(f"Tensors equal: {is_equal}")

    if not is_equal:
        # Show up to 5 differences
        diff_indices = (dst_manual != dst_builtin).nonzero()
        next_diffs = diff_indices[0:5]
        for idx in next_diffs:
            print(f"--- Index: {idx}")
            print(f"!!! Output: {dst_manual[tuple(idx)]}")
            print(f"!!! Expected: {dst_builtin[tuple(idx)]}")


    return is_equal

# Test with increasing sizes
test_memcpy_correctness((4, 777))  # Tiny
test_memcpy_correctness((100000, 15000))  # Huge: 6 GB

# Test different dtypes
test_memcpy_correctness((1000, 1000), dtype=torch.int16)
test_memcpy_correctness((1000, 1000), dtype=torch.float64)
test_memcpy_correctness((10000, 1000), dtype=torch.float32, run_benchmark=True)  # ~40 MB

try:
    test_memcpy_correctness((20000, 20000), dtype=torch.float32, run_benchmark=True)  # ~1600 MB
except RuntimeError as e:
    print(f"Skipped ~1600MB test due to: {e}")

try:
    test_memcpy_correctness((200000, 20000), dtype=torch.float32, run_benchmark=True)  # ~16000 MB
except RuntimeError as e:
    print(f"Skipped ~16000MB test due to: {e}")

print("\n===== TESTING COMPLETE =====")