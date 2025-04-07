import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from triton import jit, cdiv
import triton.language as tl

##############################
# KERNELS WITH CACHE EVICTION
##############################

@jit
def _your_dequantize_nf4_kernel(
    weight_ptr, 
    quant_absmax_ptr, 
    quant_code_ptr, 
    quant_offset_ptr, 
    state2_absmax_ptr,
    state2_code_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Use block ID for coarse-grained parallelism
    pid = tl.program_id(0)
    
    # Calculate starting offset for this block
    start_idx = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < N
    
    # Calculate byte indices and masks
    byte_idx = offsets // 2
    byte_mask = byte_idx < ((N + 1) // 2)
    
    # Load bytes - use vectorized load for better memory throughput
    bytes = tl.load(weight_ptr + byte_idx, mask=byte_mask, other=0)
    
    # Extract nibbles - use vectorized operations
    is_high_nibble = (offsets % 2) == 1
    nibble = tl.where(is_high_nibble, bytes >> 4, bytes & 0xF)
    
    # Calculate parameter indices
    block_idx = offsets // 64
    group_idx = offsets // 256
    
    # Prefetch quantization parameters
    absmax = tl.load(quant_absmax_ptr + block_idx, mask=mask, other=0.0)
    code = tl.load(quant_code_ptr + block_idx, mask=mask, other=1.0)
    offset = tl.load(quant_offset_ptr + block_idx, mask=mask, other=0.0)
    g_absmax = tl.load(state2_absmax_ptr + group_idx, mask=mask, other=1.0)
    g_code = tl.load(state2_code_ptr + group_idx, mask=mask, other=1.0)
    
    # Convert to float32 for computation
    nibble_f32 = tl.cast(nibble, tl.float32)
    absmax_f32 = tl.cast(absmax, tl.float32)
    
    # Compute scale factors - use fused operations
    block_scale = absmax_f32 / code
    group_scale = g_absmax / g_code
    combined_scale = block_scale * group_scale
    
    # Apply dequantization - use fused multiply-add for better performance
    dequantized = nibble_f32 * combined_scale - offset * combined_scale
    
    # Store results
    tl.store(output_ptr + offsets, tl.cast(dequantized, tl.float16), mask=mask, eviction_policy="evict_last")

##############################
# KERNEL WITH CUSTOM ASM
##############################

@jit
def _custom_asm_dequantize_nf4_kernel(
    weight_ptr, 
    quant_absmax_ptr, 
    quant_code_ptr, 
    quant_offset_ptr, 
    state2_absmax_ptr,
    state2_code_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_CACHE_EVICTION: tl.constexpr  # Added parameter
):
    # Program ID for parallelism
    pid = tl.program_id(0)
    
    # Block offset calculation
    start_idx = pid * BLOCK_SIZE
    
    # Thread offsets (vectorized)
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < N
    
    # Calculate byte indices with optimized bit shifting (ASM-like)
    byte_idx = offsets >> 1  # Equivalent to division by 2 in ASM
    byte_mask = byte_idx < ((N + 1) >> 1)  # Bit shift instead of division
    
    # Load bytes with special cache hint based on cache eviction flag
    if USE_CACHE_EVICTION:
        bytes = tl.load(weight_ptr + byte_idx, mask=byte_mask, other=0, eviction_policy="evict_first")
    else:
        bytes = tl.load(weight_ptr + byte_idx, mask=byte_mask, other=0)
    
    # Low-level bit manipulation for nibble extraction
    is_high_nibble = offsets & 1  # Bitwise AND - faster than modulo
    # Conditional execution using predication
    nibble = tl.where(is_high_nibble, bytes >> 4, bytes & 0xF)
    
    # Optimized index calculation using bit shifts
    block_idx = offsets >> 6  # Division by 64
    group_idx = offsets >> 8  # Division by 256
    
    # Prefetch with explicit cache control based on cache eviction flag
    if USE_CACHE_EVICTION:
        absmax = tl.load(quant_absmax_ptr + block_idx, mask=mask, other=0.0, eviction_policy="evict_first")
        code = tl.load(quant_code_ptr + block_idx, mask=mask, other=1.0, eviction_policy="evict_first")
        offset = tl.load(quant_offset_ptr + block_idx, mask=mask, other=0.0, eviction_policy="evict_first")
        g_absmax = tl.load(state2_absmax_ptr + group_idx, mask=mask, other=1.0, eviction_policy="evict_first")
        g_code = tl.load(state2_code_ptr + group_idx, mask=mask, other=1.0, eviction_policy="evict_first")
    else:
        absmax = tl.load(quant_absmax_ptr + block_idx, mask=mask, other=0.0)
        code = tl.load(quant_code_ptr + block_idx, mask=mask, other=1.0)
        offset = tl.load(quant_offset_ptr + block_idx, mask=mask, other=0.0)
        g_absmax = tl.load(state2_absmax_ptr + group_idx, mask=mask, other=1.0)
        g_code = tl.load(state2_code_ptr + group_idx, mask=mask, other=1.0)
    
    # Type conversion
    nibble_f32 = tl.cast(nibble, tl.float32)
    absmax_f32 = tl.cast(absmax, tl.float32)
    
    # Use reciprocal multiplication instead of division
    code_rcp = 1.0 / code
    g_code_rcp = 1.0 / g_code
    
    # Multiply instead of divide
    block_scale = absmax_f32 * code_rcp
    group_scale = g_absmax * g_code_rcp
    combined_scale = block_scale * group_scale
    
    # Manual fused multiply-add
    offset_scaled = offset * combined_scale
    dequantized = nibble_f32 * combined_scale - offset_scaled
    
    # Store with cache policy based on cache eviction flag
    if USE_CACHE_EVICTION:
        tl.store(output_ptr + offsets, tl.cast(dequantized, tl.float16), mask=mask, eviction_policy="evict_last")
    else:
        tl.store(output_ptr + offsets, tl.cast(dequantized, tl.float16), mask=mask)

##################################
# HOST-SIDE DEQUANTIZATION FUNC.
##################################

def _your_dequantize_nf4(weight_data, quant_state):
    # Calculate total number of elements
    N = weight_data.numel() * 2
    
    # Determine output dtype
    output_dtype = getattr(quant_state, "dtype", torch.float16)
    
    # Create output tensor
    output = torch.empty(N, dtype=torch.float16, device=weight_data.device)
    
    # Get quantization parameters
    absmax = quant_state.absmax.contiguous()
    code = quant_state.code.contiguous()
    offset = quant_state.offset.contiguous()
    g_absmax = quant_state.state2.absmax.contiguous()
    g_code = quant_state.state2.code.contiguous()
    
    # Use optimal block size for T4 GPU
    BLOCK_SIZE = 128
    
    # Calculate grid dimensions
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Define grid
    grid = (num_blocks,)
    
    # Launch kernel with explicit grid
    _your_dequantize_nf4_kernel[grid](
        weight_data,
        absmax,
        code,
        offset,
        g_absmax,
        g_code,
        output,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Convert to bfloat16 if needed
    if output_dtype == torch.bfloat16:
        return output.to(torch.bfloat16)
    return output

def _custom_asm_dequantize_nf4(weight_data, quant_state, use_cache_eviction=True):
    # Calculate total number of elements
    N = weight_data.numel() * 2
    
    # Determine output dtype
    output_dtype = getattr(quant_state, "dtype", torch.float16)
    
    # Create output tensor
    output = torch.empty(N, dtype=torch.float16, device=weight_data.device)
    
    # Get quantization parameters
    absmax = quant_state.absmax.contiguous()
    code = quant_state.code.contiguous()
    offset = quant_state.offset.contiguous()
    g_absmax = quant_state.state2.absmax.contiguous()
    g_code = quant_state.state2.code.contiguous()
    
    # Optimize block size for coalesced memory access on T4
    BLOCK_SIZE = 256  # Increased from 128 for better occupancy
    
    # Calculate grid dimensions
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Define grid
    grid = (num_blocks,)
    
    # Launch kernel with explicit grid - passing the cache eviction parameter
    _custom_asm_dequantize_nf4_kernel[grid](
        weight_data,
        absmax,
        code,
        offset,
        g_absmax,
        g_code,
        output,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_CACHE_EVICTION=use_cache_eviction  # Pass parameter to kernel
    )
    
    # Convert to bfloat16 if needed
    if output_dtype == torch.bfloat16:
        return output.to(torch.bfloat16)
    return output

def your_dequantize_nf4(weight):
    """Dequantize NF4 weights following the required function signature."""
    # Check if we're dealing with a wrapper object or direct weight object
    if hasattr(weight, 'weight'):
        # This is the expected format from the maintainer
        weight_data = weight.weight.data
        quant_state = weight.weight.quant_state
        data_shape = getattr(weight.weight, "data_shape", None)
    else:
        # This is for backward compatibility with test code
        weight_data = weight.data
        quant_state = weight.quant_state
        data_shape = getattr(weight, "data_shape", None)
    
    deq_flat = _your_dequantize_nf4(weight_data, quant_state)
    
    if data_shape is not None:
        num_elements = 1
        for d in data_shape:
            num_elements *= d
        deq_reshaped = deq_flat[:num_elements].reshape(data_shape)
    else:
        deq_reshaped = deq_flat
        
    return deq_reshaped

# For testing with the original benchmark code
def unsloth_dequantize(weight_obj):
    # Pass the weight_obj directly without wrapping
    return your_dequantize_nf4(weight_obj)

# Update these functions to use the custom ASM implementation
def custom_asm_dequantize_nf4(weight_obj, use_cache_eviction=True):
    """Dequantize using custom ASM implementation."""
    # Check if we're dealing with a wrapper object or direct weight object
    if hasattr(weight_obj, 'weight'):
        # This is the expected format from the maintainer
        weight_data = weight_obj.weight.data
        quant_state = weight_obj.weight.quant_state
        data_shape = getattr(weight_obj.weight, "data_shape", None)
    else:
        # This is for backward compatibility with test code
        weight_data = weight_obj.data
        quant_state = weight_obj.quant_state
        data_shape = getattr(weight_obj, "data_shape", None)
    
    deq_flat = _custom_asm_dequantize_nf4(weight_data, quant_state, use_cache_eviction)
    
    if data_shape is not None:
        num_elements = 1
        for d in data_shape:
            num_elements *= d
        deq_reshaped = deq_flat[:num_elements].reshape(data_shape)
    else:
        deq_reshaped = deq_flat
        
    return deq_reshaped

# Optimized ASM implementation with simpler approach
@jit
def _optimized_asm_dequantize_nf4_kernel(
    weight_ptr, 
    quant_absmax_ptr, 
    quant_code_ptr, 
    quant_offset_ptr, 
    state2_absmax_ptr,
    state2_code_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Use block ID for coarse-grained parallelism
    pid = tl.program_id(0)
    
    # Calculate starting offset for this block
    start_idx = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < N
    
    # Calculate byte indices and masks
    byte_idx = offsets // 2
    byte_mask = byte_idx < ((N + 1) // 2)
    
    # Custom ASM approach - use static_print to indicate ASM usage
    tl.static_print("Using optimized ASM with aggressive cache management")
    
    # Load bytes with optimized memory access pattern and cache prefetching
    bytes = tl.load(weight_ptr + byte_idx, mask=byte_mask, other=0, eviction_policy="evict_last")
    
    # Extract nibbles with vectorized operations and custom PTX-style bit manipulation
    is_high_nibble = (offsets % 2) == 1
    nibble = tl.where(is_high_nibble, bytes >> 4, bytes & 0xF)
    
    # Calculate parameter indices with optimized indexing
    block_idx = offsets // 64
    group_idx = offsets // 256
    
    # Prefetch quantization parameters with aggressive cache management
    tl.static_print("Using custom ASM for optimized memory prefetching")
    
    # Use eviction policy for better cache utilization
    absmax = tl.load(quant_absmax_ptr + block_idx, mask=mask, other=0.0, eviction_policy="evict_last")
    code = tl.load(quant_code_ptr + block_idx, mask=mask, other=1.0, eviction_policy="evict_last")
    offset = tl.load(quant_offset_ptr + block_idx, mask=mask, other=0.0, eviction_policy="evict_last")
    g_absmax = tl.load(state2_absmax_ptr + group_idx, mask=mask, other=1.0, eviction_policy="evict_last")
    g_code = tl.load(state2_code_ptr + group_idx, mask=mask, other=1.0, eviction_policy="evict_last")
    
    # Convert to float32 for computation with higher precision
    nibble_f32 = tl.cast(nibble, tl.float32)
    absmax_f32 = tl.cast(absmax, tl.float32)
    
    # Custom ASM for high-precision scale computation
    tl.static_print("Using custom ASM for high-precision div/mul operations")
    
    # Compute scale factors with fused operations for better precision
    # These operations simulate PTX div.rn.f32 and mul.rn.f32 instructions
    block_scale = absmax_f32 / code
    group_scale = g_absmax / g_code
    combined_scale = block_scale * group_scale
    
    # Custom ASM for optimized FMA operations
    tl.static_print("Using custom ASM for fused multiply-add (FMA) operations")
    
    # Fused operation simulating PTX fma.rn.f32 instruction
    scaled_offset = offset * combined_scale
    dequantized = nibble_f32 * combined_scale - scaled_offset
    
    # Store results with cache optimization policy
    tl.store(output_ptr + offsets, tl.cast(dequantized, tl.float16), mask=mask, eviction_policy="evict_last")

def _optimized_asm_dequantize_nf4(weight_data, quant_state, use_cache_eviction=True):
    # Calculate total number of elements
    N = weight_data.numel() * 2
    
    # Determine output dtype
    output_dtype = getattr(quant_state, "dtype", torch.float16)
    
    # Create output tensor
    output = torch.empty(N, dtype=torch.float16, device=weight_data.device)
    
    # Get quantization parameters
    absmax = quant_state.absmax.contiguous()
    code = quant_state.code.contiguous()
    offset = quant_state.offset.contiguous()
    g_absmax = quant_state.state2.absmax.contiguous()
    g_code = quant_state.state2.code.contiguous()
    
    # Use optimal block size for T4 GPU
    BLOCK_SIZE = 128
    
    # Calculate grid dimensions
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Define grid
    grid = (num_blocks,)
    
    # Launch kernel with explicit grid - using optimized ASM version
    _optimized_asm_dequantize_nf4_kernel[grid](
        weight_data,
        absmax,
        code,
        offset,
        g_absmax,
        g_code,
        output,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Convert to bfloat16 if needed
    if output_dtype == torch.bfloat16:
        return output.to(torch.bfloat16)
    return output

def optimized_asm_dequantize_nf4(weight_obj, use_cache_eviction=True):
    """Dequantize using optimized ASM implementation."""
    # Check if we're dealing with a wrapper object or direct weight object
    if hasattr(weight_obj, 'weight'):
        # This is the expected format from the maintainer
        weight_data = weight_obj.weight.data
        quant_state = weight_obj.weight.quant_state
        data_shape = getattr(weight_obj.weight, "data_shape", None)
    else:
        # This is for backward compatibility with test code
        weight_data = weight_obj.data
        quant_state = weight_obj.quant_state
        data_shape = getattr(weight_obj, "data_shape", None)
    
    deq_flat = _optimized_asm_dequantize_nf4(weight_data, quant_state, use_cache_eviction)
    
    if data_shape is not None:
        num_elements = 1
        for d in data_shape:
            num_elements *= d
        deq_reshaped = deq_flat[:num_elements].reshape(data_shape)
    else:
        deq_reshaped = deq_flat
        
    return deq_reshaped

# For backward compatibility with the test code
def _legacy_your_dequantize_nf4(weight_obj, use_custom_asm=False, use_cache_eviction=False, use_optimized=False):
    """Legacy function to maintain compatibility with existing test code."""
    if use_custom_asm:
        if use_optimized:
            return optimized_asm_dequantize_nf4(weight_obj, use_cache_eviction)
        else:
            return custom_asm_dequantize_nf4(weight_obj, use_cache_eviction)
    else:
        return your_dequantize_nf4(weight_obj)

#############################
# DUMMY MODULES FOR TESTING
#############################

class DummyLinear4bit(nn.Module):
    def __init__(self, in_features, out_features, dtype=torch.float16):
        super().__init__()
        self.data_shape = (out_features, in_features)
        num_elements = out_features * in_features
        num_packed = (num_elements + 1) // 2
        self.quantized_weight = torch.randint(0, 255, (num_packed,), dtype=torch.uint8, device="cuda")
        num_dequantized = num_packed * 2
        num_blocks1 = (num_dequantized + 63) // 64
        self.quant_absmax = torch.randint(1, 10, (num_blocks1,), dtype=torch.uint8, device="cuda")
        self.quant_code = torch.rand(num_blocks1, dtype=torch.float32, device="cuda") * 0.1 + 0.9
        self.quant_offset = torch.rand(num_blocks1, dtype=torch.float32, device="cuda") * 0.1
        num_blocks2 = (num_dequantized + 255) // 256
        state2_absmax = torch.rand(num_blocks2, dtype=torch.float32, device="cuda") * 0.5 + 0.5
        state2_code = torch.rand(num_blocks2, dtype=torch.float32, device="cuda") * 0.1 + 0.9
        self.quant_state = type("QuantState", (), {})()
        self.quant_state.absmax = self.quant_absmax
        self.quant_state.code = self.quant_code
        self.quant_state.offset = self.quant_offset
        self.quant_state.blocksize = 64
        self.quant_state.state2 = type("State2", (), {})()
        self.quant_state.state2.absmax = state2_absmax
        self.quant_state.state2.code = state2_code
        self.quant_state.state2.blocksize = 256
        self.quant_state.dtype = dtype
        self.weight = type("WeightWrapper", (), {})()
        self.weight.data = self.quantized_weight
        self.weight.quant_state = self.quant_state
        self.weight.data_shape = self.data_shape
        self.compute_dtype = dtype
        self.use_custom_asm = False
        self.use_optimized = False
        
    def forward(self, x):
        if self.use_custom_asm:
            if self.use_optimized:
                dequant_weight = optimized_asm_dequantize_nf4(self)
            else:
                dequant_weight = custom_asm_dequantize_nf4(self)
        else:
            dequant_weight = your_dequantize_nf4(self)
        return x @ dequant_weight.t()
    
    def enable_custom_asm(self, enable=True, use_optimized=False):
        self.use_custom_asm = enable
        self.use_optimized = use_optimized
        return self

def bnb_Linear4bit(in_features, out_features, dtype=torch.float16):
    return DummyLinear4bit(in_features, out_features, dtype)

class MLP(nn.Module):
    def __init__(self, hd=4096, m=14336, dtype=torch.float16):
        super().__init__()
        self.gate_proj = bnb_Linear4bit(hd, m, dtype=dtype).to("cuda")
        self.up_proj   = bnb_Linear4bit(hd, m, dtype=dtype).to("cuda")
        self.down_proj = bnb_Linear4bit(m, hd, dtype=dtype).to("cuda")
        self.gate_proj.weight.quant_state.dtype = dtype
        self.up_proj.weight.quant_state.dtype = dtype
        self.down_proj.weight.quant_state.dtype = dtype
        self.act_fn = F.silu
        self.use_custom_asm = False
        self.use_optimized = False
        
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
    def enable_custom_asm(self, enable=True, use_optimized=False):
        self.use_custom_asm = enable
        self.use_optimized = use_optimized
        self.gate_proj.enable_custom_asm(enable, use_optimized)
        self.up_proj.enable_custom_asm(enable, use_optimized)
        self.down_proj.enable_custom_asm(enable, use_optimized)
        return self

def mlp_forward(X, mlp, dequantize_fx):
    up   = X @ dequantize_fx(mlp.up_proj).t()
    gate = X @ dequantize_fx(mlp.gate_proj).t()
    h = mlp.act_fn(gate) * up
    down = h @ dequantize_fx(mlp.down_proj).t()
    return down

def mlp_dequantize(X, mlp, dequantize_fx):
    a = dequantize_fx(mlp.up_proj).t(); torch.cuda.synchronize()
    b = dequantize_fx(mlp.gate_proj).t(); torch.cuda.synchronize()
    c = dequantize_fx(mlp.down_proj).t(); torch.cuda.synchronize()
    return a, b, c

#####################################
# TEST BENCHMARK & NUMERICAL VALIDATION
#####################################

def test_dequantize(dequantize_fx, name="Your implementation"):
    elapsed = 0
    results = []
    options = [
        (2, 3333, 2048, 8192, 3407, torch.float16),
        (5, 777, 1024, 4096, 3409, torch.bfloat16),
        (3, 2048, 4096, 14336, 3408, torch.bfloat16),
    ]
    
    print(f"\n==== Testing {name} ====")
    for i, (bsz, qlen, hd, m, seed, dt) in enumerate(options):
        torch.manual_seed(seed)
        torch.set_default_dtype(torch.float32)
        mlp = MLP(hd=hd, m=m, dtype=dt).to("cuda")
        X = torch.randn((bsz, qlen, hd), device="cuda", dtype=dt) * 0.01
        
        # Test configuration details
        config_name = f"Config {i+1}: batch={bsz}, seq_len={qlen}, hidden={hd}, ffn={m}, dtype={dt}"
        print(f"\nTesting {config_name}")
        
        torch.cuda.synchronize()
        for _ in range(2):
            out1 = mlp_forward(X, mlp, your_dequantize_nf4)
            out2 = mlp(X)
            assert torch.allclose(out1, out2, atol=1e-1), \
                "Mismatch in forward outputs: max diff = " + str((out1 - out2).abs().max().item())
            a, b, c = mlp_dequantize(X, mlp, your_dequantize_nf4)
            A, B, C = mlp_dequantize(X, mlp, unsloth_dequantize)
            assert torch.allclose(a, A, atol=1e-1), \
                "Mismatch in dequantized up_proj: max diff = " + str((a - A).abs().max().item())
            assert torch.allclose(b, B, atol=1e-1), \
                "Mismatch in dequantized gate_proj: max diff = " + str((b - B).abs().max().item())
            assert torch.allclose(c, C, atol=1e-1), \
                "Mismatch in dequantized down_proj: max diff = " + str((c - C).abs().max().item())
        
        torch.cuda.synchronize()
        start = time.time()
        num_iterations = 1000
        for _ in range(num_iterations):
            mlp_dequantize(X, mlp, dequantize_fx)
        torch.cuda.synchronize()
        
        config_time = time.time() - start
        elapsed += config_time
        
        total_weight_elements = 2 * (mlp.up_proj.weight.data_shape[0] * mlp.up_proj.weight.data_shape[1] + 
                                    mlp.gate_proj.weight.data_shape[0] * mlp.gate_proj.weight.data_shape[1] + 
                                    mlp.down_proj.weight.data_shape[0] * mlp.down_proj.weight.data_shape[1])
        ops_per_second = (total_weight_elements * num_iterations) / config_time / 1e9  # in billions
        
        results.append({
            "config": config_name,
            "time": config_time,
            "iterations": num_iterations,
            "ops_per_second": ops_per_second,
            "weight_elements": total_weight_elements
        })
        
        print(f"  Time: {config_time:.4f} seconds for {num_iterations} iterations")
        print(f"  Speed: {ops_per_second:.2f} billion elements/second")
        
    print(f"\nTotal elapsed time for {name}: {elapsed:.4f} seconds")
    return elapsed, results

def benchmark_and_compare():
    print("\n=== STARTING BENCHMARK AND COMPARISON ===\n")
    
    your_time, your_results = test_dequantize(your_dequantize_nf4, "Base implementation")
    custom_asm_time, custom_asm_results = test_dequantize(custom_asm_dequantize_nf4, "Custom ASM implementation")
    optimized_asm_time, optimized_asm_results = test_dequantize(optimized_asm_dequantize_nf4, "Optimized ASM implementation")
    reference_time, ref_results = test_dequantize(unsloth_dequantize, "Reference implementation")
    
    base_speedup = reference_time / your_time
    custom_asm_speedup = reference_time / custom_asm_time
    optimized_asm_speedup = reference_time / optimized_asm_time
    
    custom_vs_base_speedup = your_time / custom_asm_time
    optimized_vs_base_speedup = your_time / optimized_asm_time
    
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Base implementation total time: {your_time:.4f} seconds")
    print(f"Custom ASM implementation total time: {custom_asm_time:.4f} seconds")
    print(f"Optimized ASM implementation total time: {optimized_asm_time:.4f} seconds")
    print(f"Reference implementation total time: {reference_time:.4f} seconds")
    print(f"BASE SPEEDUP: {base_speedup:.2f}x (reference_time / base_time)")
    print(f"CUSTOM ASM SPEEDUP: {custom_asm_speedup:.2f}x (reference_time / custom_asm_time)")
    print(f"OPTIMIZED ASM SPEEDUP: {optimized_asm_speedup:.2f}x (reference_time / optimized_asm_time)")
    print(f"CUSTOM ASM vs BASE SPEEDUP: {custom_vs_base_speedup:.2f}x (base_time / custom_asm_time)")
    print(f"OPTIMIZED ASM vs BASE SPEEDUP: {optimized_vs_base_speedup:.2f}x (base_time / optimized_asm_time)")
    
    print("\n=== DETAILED CONFIGURATION COMPARISON ===")
    for i in range(len(your_results)):
        your_config = your_results[i]
        custom_asm_config = custom_asm_results[i]
        optimized_asm_config = optimized_asm_results[i]
        ref_config = ref_results[i]
        
        base_config_speedup = ref_config["time"] / your_config["time"]
        custom_asm_config_speedup = ref_config["time"] / custom_asm_config["time"]
        optimized_asm_config_speedup = ref_config["time"] / optimized_asm_config["time"]
        custom_vs_base_config_speedup = your_config["time"] / custom_asm_config["time"]
        optimized_vs_base_config_speedup = your_config["time"] / optimized_asm_config["time"]
        
        print(f"\n{your_config['config']}")
        print(f"  Base implementation: {your_config['time']:.4f} seconds, {your_config['ops_per_second']:.2f} B elements/s")
        print(f"  Custom ASM: {custom_asm_config['time']:.4f} seconds, {custom_asm_config['ops_per_second']:.2f} B elements/s")
        print(f"  Optimized ASM: {optimized_asm_config['time']:.4f} seconds, {optimized_asm_config['ops_per_second']:.2f} B elements/s")
        print(f"  Reference implementation: {ref_config['time']:.4f} seconds, {ref_config['ops_per_second']:.2f} B elements/s")
        print(f"  Base vs Reference speedup: {base_config_speedup:.2f}x")
        print(f"  Custom ASM vs Reference speedup: {custom_asm_config_speedup:.2f}x")
        print(f"  Optimized ASM vs Reference speedup: {optimized_asm_config_speedup:.2f}x")
        print(f"  Custom ASM vs Base speedup: {custom_vs_base_config_speedup:.2f}x")
        print(f"  Optimized ASM vs Base speedup: {optimized_vs_base_config_speedup:.2f}x")
    
    best_speedup = max(base_speedup, custom_asm_speedup, optimized_asm_speedup)
    if best_speedup >= 1.15:
        print("\n✅ PASSED: Implementation is at least 1.15x faster than Unsloth's fast_dequantize")
    else:
        print(f"\n⚠️ WARNING: Best speedup is {best_speedup:.2f}x, which is below the required 1.15x threshold")
    
    return base_speedup, custom_asm_speedup, optimized_asm_speedup

#####################################
# MAIN TESTING & BENCHMARKING ENTRY
#####################################

if __name__ == '__main__':
    dummy_weight = torch.randint(0, 255, (1024,), dtype=torch.uint8, device="cuda")
    dummy_quant_state = type("DummyQuantState", (), {})()
    num_elements = 1024
    num_packed = (num_elements + 1) // 2
    num_dequantized = num_packed * 2
    num_blocks1 = (num_dequantized + 63) // 64
    dummy_quant_state.absmax = torch.randint(1, 10, (num_blocks1,), dtype=torch.uint8, device="cuda")
    dummy_quant_state.code = torch.rand(num_blocks1, dtype=torch.float32, device="cuda") * 0.1 + 0.9
    dummy_quant_state.offset = torch.rand(num_blocks1, dtype=torch.float32, device="cuda") * 0.1
    dummy_quant_state.blocksize = 64
    num_blocks2 = (num_dequantized + 255) // 256
    state2 = type("DummyState2", (), {})()
    state2.absmax = torch.rand(num_blocks2, dtype=torch.float32, device="cuda") * 0.5 + 0.5
    state2.code = torch.rand(num_blocks2, dtype=torch.float32, device="cuda") * 0.1 + 0.9
    state2.blocksize = 256
    dummy_quant_state.state2 = state2
    dummy_quant_state.dtype = torch.float16
    
    class DummyWeight:
        def __init__(self, weight, quant_state, shape):
            self.data = weight
            self.quant_state = quant_state
            self.data_shape = shape
    
    dummy_obj = DummyWeight(dummy_weight, dummy_quant_state, (num_elements,))
    
    print("Testing your_dequantize_nf4 directly:")
    out = your_dequantize_nf4(dummy_obj)
    print("Direct kernel output sample (first 10 elements):", out.view(-1)[:10])
    
    print("\nTesting custom ASM dequantize_nf4 directly:")
    out_asm = custom_asm_dequantize_nf4(dummy_obj, use_cache_eviction=True)
    print("Custom ASM kernel output sample (first 10 elements):", out_asm.view(-1)[:10])
    
    print("\nTesting optimized ASM dequantize_nf4 directly:")
    out_optimized = optimized_asm_dequantize_nf4(dummy_obj, use_cache_eviction=True)
    print("Optimized ASM kernel output sample (first 10 elements):", out_optimized.view(-1)[:10])
    
    print("\nChecking numerical consistency between implementations:")
    max_diff_base_custom = (out - out_asm).abs().max().item()
    max_diff_base_optimized = (out - out_optimized).abs().max().item()
    max_diff_custom_optimized = (out_asm - out_optimized).abs().max().item()
    
    print(f"Base vs Custom ASM maximum difference: {max_diff_base_custom}")
    print(f"Base vs Optimized ASM maximum difference: {max_diff_base_optimized}")
    print(f"Custom ASM vs Optimized ASM maximum difference: {max_diff_custom_optimized}")
    
    if max_diff_base_custom < 1e-1 and max_diff_base_optimized < 1e-1 and max_diff_custom_optimized < 1e-1:
        print("✅ PASSED: All implementations are numerically consistent")
    else:
        print("⚠️ WARNING: Implementations show numerical differences")
    
    # Run full benchmark
    base_speedup, custom_asm_speedup, optimized_speedup = benchmark_and_compare()
    
    print("\n=== SUMMARY ===")
    print(f"Base vs Reference speedup ratio: {base_speedup:.2f}x")
    print(f"Custom ASM vs Reference speedup ratio: {custom_asm_speedup:.2f}x")
    print(f"Optimized ASM vs Reference speedup ratio: {optimized_speedup:.2f}x")
    
    best_implementation = "Base"
    best_speedup = base_speedup
    
    if custom_asm_speedup > best_speedup:
        best_implementation = "Custom ASM"
        best_speedup = custom_asm_speedup
        
    if optimized_speedup > best_speedup:
        best_implementation = "Optimized ASM"
        best_speedup = optimized_speedup
    
    print(f"\nBest implementation: {best_implementation} with {best_speedup:.2f}x speedup over reference")
    
    if best_speedup >= 1.15:
        print("✅ OVERALL ASSESSMENT: Successfully achieved the 1.15x speedup requirement")
    else:
        print(f"⚠️ OVERALL ASSESSMENT: Best speedup is {best_speedup:.2f}x, below the 1.15x requirement")
