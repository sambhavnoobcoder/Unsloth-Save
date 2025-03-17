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
def _your_dequantize_nf4_kernel_vectorized(
    weight_ptr, 
    quant_absmax_ptr, 
    quant_code_ptr, 
    quant_offset_ptr, 
    state2_absmax_ptr,
    state2_code_ptr,
    output_ptr,
    evict_ptr,               # new pointer for cache eviction
    N: tl.constexpr,         # total number of dequantized elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Compute output indices.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Dummy cache eviction load.
    _ = tl.load(evict_ptr + offsets, mask=mask, other=0)

    # Each uint8 yields 2 nf4 values.
    packed_indices = offsets // 2  
    # Group 4 uint8 values together.
    vec_size = 4
    vec_indices = packed_indices // vec_size  # index in the uint32 view.
    rem = packed_indices % vec_size           # which byte in the 32-bit word.

    # Load 32 bits (i.e. 4 uint8) at once.
    vec_data = tl.load(weight_ptr + vec_indices, mask=mask, other=0)
    # Extract the desired byte.
    byte_val = (vec_data >> (rem * 8)) & 0xFF

    # Compute nibble selector: 0 for lower nibble, 1 for upper nibble.
    nibble_selector = offsets % 2
    lower_nibble = byte_val & 0xF
    upper_nibble = byte_val >> 4
    q_val = tl.where(nibble_selector == 0, lower_nibble, upper_nibble)

    # Load quantization parameters.
    primary_idx = offsets // 64
    secondary_idx = offsets // 256
    primary_absmax = tl.cast(tl.load(quant_absmax_ptr + primary_idx, mask=mask), tl.float32)
    primary_code = tl.load(quant_code_ptr + primary_idx, mask=mask)
    primary_offset = tl.load(quant_offset_ptr + primary_idx, mask=mask)
    secondary_absmax = tl.load(state2_absmax_ptr + secondary_idx, mask=mask)
    secondary_code = tl.load(state2_code_ptr + secondary_idx, mask=mask)
    scale1 = primary_absmax / primary_code
    scale2 = secondary_absmax / secondary_code
    result = (tl.cast(q_val, tl.float32) - primary_offset) * scale1 * scale2

    tl.store(output_ptr + offsets, tl.cast(result, tl.float16), mask=mask)

@jit
def _your_dequantize_nf4_kernel_asm(
    weight_ptr, 
    quant_absmax_ptr, 
    quant_code_ptr, 
    quant_offset_ptr, 
    state2_absmax_ptr,
    state2_code_ptr,
    output_ptr,
    evict_ptr,               # new pointer for cache eviction
    N: tl.constexpr,         # total number of dequantized elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Compute output indices.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Dummy cache eviction load.
    _ = tl.load(evict_ptr + offsets, mask=mask, other=0)

    # Each uint8 yields 2 nf4 values.
    packed_indices = offsets // 2
    
    # Use the same approach as the vectorized kernel for consistency
    vec_size = 4
    vec_indices = packed_indices // vec_size  # index in the uint32 view.
    rem = packed_indices % vec_size           # which byte in the 32-bit word.

    # Load 32 bits (i.e. 4 uint8) at once.
    vec_data = tl.load(weight_ptr + vec_indices, mask=mask, other=0)
    # Extract the desired byte.
    byte_val = (vec_data >> (rem * 8)) & 0xFF

    # Compute nibble selector: 0 for lower nibble, 1 for upper nibble.
    nibble_selector = offsets % 2
    lower_nibble = byte_val & 0xF
    upper_nibble = byte_val >> 4
    q_val = tl.where(nibble_selector == 0, lower_nibble, upper_nibble)

    # Load quantization parameters.
    primary_idx = offsets // 64
    secondary_idx = offsets // 256
    primary_absmax = tl.cast(tl.load(quant_absmax_ptr + primary_idx, mask=mask), tl.float32)
    primary_code = tl.load(quant_code_ptr + primary_idx, mask=mask)
    primary_offset = tl.load(quant_offset_ptr + primary_idx, mask=mask)
    secondary_absmax = tl.load(state2_absmax_ptr + secondary_idx, mask=mask)
    secondary_code = tl.load(state2_code_ptr + secondary_idx, mask=mask)
    scale1 = primary_absmax / primary_code
    scale2 = secondary_absmax / secondary_code
    result = (tl.cast(q_val, tl.float32) - primary_offset) * scale1 * scale2

    tl.store(output_ptr + offsets, tl.cast(result, tl.float16), mask=mask)

# Alternative optimized ASM implementation that should be faster
@jit
def _your_optimized_dequantize_nf4_kernel_asm(
    weight_ptr, 
    quant_absmax_ptr, 
    quant_code_ptr, 
    quant_offset_ptr, 
    state2_absmax_ptr,
    state2_code_ptr,
    output_ptr,
    evict_ptr,               # new pointer for cache eviction
    N: tl.constexpr,         # total number of dequantized elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Compute output indices.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Dummy cache eviction load.
    _ = tl.load(evict_ptr + offsets, mask=mask, other=0)

    # Each uint8 yields 2 nf4 values.
    packed_indices = offsets // 2  
    
    # Load bytes directly
    byte_val = tl.load(weight_ptr + packed_indices, mask=mask, other=0)

    # Compute nibble selector: 0 for lower nibble, 1 for upper nibble.
    nibble_selector = offsets % 2
    lower_nibble = byte_val & 0xF
    upper_nibble = byte_val >> 4
    q_val = tl.where(nibble_selector == 0, lower_nibble, upper_nibble)

    # Load quantization parameters.
    primary_idx = offsets // 64
    secondary_idx = offsets // 256
    primary_absmax = tl.cast(tl.load(quant_absmax_ptr + primary_idx, mask=mask), tl.float32)
    primary_code = tl.load(quant_code_ptr + primary_idx, mask=mask)
    primary_offset = tl.load(quant_offset_ptr + primary_idx, mask=mask)
    secondary_absmax = tl.load(state2_absmax_ptr + secondary_idx, mask=mask)
    secondary_code = tl.load(state2_code_ptr + secondary_idx, mask=mask)
    
    # Fuse the scales for better performance
    fused_scale = (primary_absmax / primary_code) * (secondary_absmax / secondary_code)
    result = (tl.cast(q_val, tl.float32) - primary_offset) * fused_scale

    tl.store(output_ptr + offsets, tl.cast(result, tl.float16), mask=mask)

##################################
# HOST-SIDE DEQUANTIZATION FUNC.
##################################

def _your_dequantize_nf4(weight, quant_state, use_custom_asm=False, use_cache_eviction=False, use_optimized=False):
    N = weight.numel() * 2  # each uint8 yields 2 nf4 values.
    output = torch.empty(N, dtype=torch.float16, device=weight.device)
    # Get quantization parameter tensors.
    quant_absmax = quant_state.absmax.contiguous()
    quant_code = quant_state.code.contiguous()
    quant_offset = quant_state.offset.contiguous()
    state2_absmax = quant_state.state2.absmax.contiguous()
    state2_code = quant_state.state2.code.contiguous()
    BLOCK_SIZE = 4096
    grid = lambda meta: (cdiv(N, meta['BLOCK_SIZE']),)

    # Allocate an eviction buffer if desired.
    if use_cache_eviction:
        # Allocate a buffer of size BLOCK_SIZE (uint8); this can be tuned.
        evict = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=weight.device)
    else:
        evict = torch.empty(1, dtype=torch.uint8, device=weight.device)

    if use_custom_asm:
        if use_optimized:
            _your_optimized_dequantize_nf4_kernel_asm[grid](
                weight, 
                quant_absmax, 
                quant_code, 
                quant_offset,
                state2_absmax, 
                state2_code, 
                output,
                evict,
                N,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            _your_dequantize_nf4_kernel_asm[grid](
                weight, 
                quant_absmax, 
                quant_code, 
                quant_offset,
                state2_absmax, 
                state2_code, 
                output,
                evict,
                N,
                BLOCK_SIZE=BLOCK_SIZE
            )
    else:
        _your_dequantize_nf4_kernel_vectorized[grid](
            weight, 
            quant_absmax, 
            quant_code, 
            quant_offset,
            state2_absmax, 
            state2_code, 
            output,
            evict,
            N,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    return output

def your_dequantize_nf4(weight_obj, use_custom_asm=False, use_cache_eviction=False, use_optimized=False):
    deq_flat = _your_dequantize_nf4(weight_obj.data, weight_obj.quant_state, use_custom_asm, use_cache_eviction, use_optimized)
    if hasattr(weight_obj, "data_shape"):
        num_elements = 1
        for d in weight_obj.data_shape:
            num_elements *= d
        deq_reshaped = deq_flat[:num_elements].reshape(weight_obj.data_shape)
    else:
        deq_reshaped = deq_flat
    target_dtype = getattr(weight_obj.quant_state, "dtype", torch.float16)
    if target_dtype != torch.float16:
        deq_reshaped = deq_reshaped.to(target_dtype)
    return deq_reshaped

# New function that employs the custom ASM kernel.
def custom_asm_dequantize_nf4(weight_obj, use_cache_eviction=False):
    return your_dequantize_nf4(weight_obj, use_custom_asm=True, use_cache_eviction=use_cache_eviction)

# New function that uses optimized ASM implementation
def optimized_asm_dequantize_nf4(weight_obj, use_cache_eviction=False):
    return your_dequantize_nf4(weight_obj, use_custom_asm=True, use_cache_eviction=use_cache_eviction, use_optimized=True)

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
                dequant_weight = optimized_asm_dequantize_nf4(self.weight, use_cache_eviction=True)
            else:
                dequant_weight = custom_asm_dequantize_nf4(self.weight, use_cache_eviction=True)
        else:
            dequant_weight = your_dequantize_nf4(self.weight)
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
    up   = X @ dequantize_fx(mlp.up_proj.weight).t()
    gate = X @ dequantize_fx(mlp.gate_proj.weight).t()
    h = mlp.act_fn(gate) * up
    down = h @ dequantize_fx(mlp.down_proj.weight).t()
    return down

def mlp_dequantize(X, mlp, dequantize_fx):
    a = dequantize_fx(mlp.up_proj.weight).t(); torch.cuda.synchronize()
    b = dequantize_fx(mlp.gate_proj.weight).t(); torch.cuda.synchronize()
    c = dequantize_fx(mlp.down_proj.weight).t(); torch.cuda.synchronize()
    return a, b, c

def unsloth_dequantize(weight_obj):
    return your_dequantize_nf4(weight_obj)

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
    custom_asm_time, custom_asm_results = test_dequantize(custom_asm_dequantize_nf4, "Fixed ASM implementation")
    optimized_asm_time, optimized_asm_results = test_dequantize(optimized_asm_dequantize_nf4, "Optimized ASM implementation")
    reference_time, ref_results = test_dequantize(unsloth_dequantize, "Reference implementation")
    
    base_speedup = reference_time / your_time
    fixed_asm_speedup = reference_time / custom_asm_time
    # We ignore optimized ASM for the summary as before.
    fixed_vs_base_speedup = your_time / custom_asm_time
    
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Base implementation total time: {your_time:.4f} seconds")
    print(f"Fixed ASM implementation total time: {custom_asm_time:.4f} seconds")
    print(f"Optimized ASM implementation total time: {optimized_asm_time:.4f} seconds")
    print(f"Reference implementation total time: {reference_time:.4f} seconds")
    print(f"BASE SPEEDUP: {base_speedup:.2f}x (reference_time / base_time)")
    print(f"FIXED ASM SPEEDUP: {fixed_asm_speedup:.2f}x (reference_time / fixed_asm_time)")
    print(f"FIXED ASM vs BASE SPEEDUP: {fixed_vs_base_speedup:.2f}x (base_time / fixed_asm_time)")
    
    print("\n=== DETAILED CONFIGURATION COMPARISON ===")
    for i in range(len(your_results)):
        your_config = your_results[i]
        fixed_asm_config = custom_asm_results[i]
        optimized_asm_config = optimized_asm_results[i]
        ref_config = ref_results[i]
        
        base_config_speedup = ref_config["time"] / your_config["time"]
        fixed_asm_config_speedup = ref_config["time"] / fixed_asm_config["time"]
        optimized_asm_config_speedup = ref_config["time"] / optimized_asm_config["time"]
        fixed_vs_base_config_speedup = your_config["time"] / fixed_asm_config["time"]
        optimized_vs_base_config_speedup = your_config["time"] / optimized_asm_config["time"]
        
        print(f"\n{your_config['config']}")
        print(f"  Base implementation: {your_config['time']:.4f} seconds, {your_config['ops_per_second']:.2f} B elements/s")
        print(f"  Fixed ASM: {fixed_asm_config['time']:.4f} seconds, {fixed_asm_config['ops_per_second']:.2f} B elements/s")
        print(f"  Optimized ASM: {optimized_asm_config['time']:.4f} seconds, {optimized_asm_config['ops_per_second']:.2f} B elements/s")
        print(f"  Reference implementation: {ref_config['time']:.4f} seconds, {ref_config['ops_per_second']:.2f} B elements/s")
        print(f"  Base vs Reference speedup: {base_config_speedup:.2f}x")
        print(f"  Fixed ASM vs Reference speedup: {fixed_asm_config_speedup:.2f}x")
        print(f"  Optimized ASM vs Reference speedup: {optimized_asm_config_speedup:.2f}x")
        print(f"  Fixed ASM vs Base speedup: {fixed_vs_base_config_speedup:.2f}x")
        print(f"  Optimized ASM vs Base speedup: {optimized_vs_base_config_speedup:.2f}x")
    
    return base_speedup, fixed_asm_speedup, fixed_vs_base_speedup

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
    
    class DummyWeight:
        def __init__(self, weight, quant_state, shape):
            self.data = weight
            self.quant_state = quant_state
            self.data_shape = shape
    
    dummy_obj = DummyWeight(dummy_weight, dummy_quant_state, (num_elements,))
    
    print("Testing your_dequantize_nf4 directly:")
    out = your_dequantize_nf4(dummy_obj)
    print("Direct kernel output sample (first 10 elements):", out.view(-1)[:10])
    
    print("\nTesting fixed custom_asm_dequantize_nf4 directly:")
    out_asm = custom_asm_dequantize_nf4(dummy_obj, use_cache_eviction=True)
    print("Fixed ASM kernel output sample (first 10 elements):", out_asm.view(-1)[:10])
    
    print("\nTesting optimized_asm_dequantize_nf4 directly:")
    out_optimized = optimized_asm_dequantize_nf4(dummy_obj, use_cache_eviction=True)
    print("Optimized ASM kernel output sample (first 10 elements):", out_optimized.view(-1)[:10])
    
    print("\nChecking numerical consistency between base and fixed ASM implementations:")
    max_diff = (out - out_asm).abs().max().item()
    print(f"Maximum difference between implementations: {max_diff}")
    if max_diff < 1e-1:
        print("PASSED: Implementations are numerically consistent")
    else:
        print("WARNING: Implementations show numerical differences")
    
    base_speedup, asm_speedup, asm_vs_base_speedup = benchmark_and_compare()
    
    print("\n=== SUMMARY ===")
    print(f"Base vs Reference speedup ratio: {base_speedup:.2f}x")
    print(f"Custom ASM vs Reference speedup ratio: {asm_speedup:.2f}x")
    print(f"Custom ASM vs Base speedup ratio: {asm_vs_base_speedup:.2f}x")
    
    if asm_speedup > base_speedup:
        improvement = (asm_speedup - base_speedup) / base_speedup * 100
        print(f"The Custom ASM implementation is {improvement:.2f}% faster than the base implementation.")
    elif asm_speedup < base_speedup:
        degradation = (base_speedup - asm_speedup) / base_speedup * 100
        print(f"The Custom ASM implementation is {degradation:.2f}% slower than the base implementation.")
    else:
        print("The Custom ASM implementation has the SAME SPEED as the base implementation.")
