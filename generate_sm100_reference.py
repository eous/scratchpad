#!/usr/bin/env python3
"""
SM100 Reference Output Generator for Functional Correctness Validation

This script runs on an SM100 machine to generate reference outputs that can be
compared against SM120 implementation for functional correctness.

Usage:
    python generate_sm100_reference.py --output sm100_reference.pkl

Output file contains:
    - System information (GPU model, CUDA version, library versions)
    - FlashMLA Decode outputs (various configurations)
    - FlashMLA Prefill outputs (various configurations)
    - DeepGEMM outputs (various configurations)
    - Intermediate values (LSE, attention weights, etc.)
    - Timing information (bonus)

Expected runtime: ~3-5 minutes on SM100
"""

import torch
import numpy as np
import pickle
import json
import time
import argparse
import sys
import hashlib
import traceback
from pathlib import Path

# Fixed random seed for reproducibility across SM100/SM120
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def validate_environment():
    """Validate that we're on a CUDA-capable machine"""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("ERROR: No CUDA devices found.")
        sys.exit(1)

    # Verify we can allocate on GPU
    try:
        test = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()
        del test
    except RuntimeError as e:
        print(f"ERROR: Cannot allocate CUDA memory: {e}")
        sys.exit(1)

    return True

def deterministic_hash(s: str) -> int:
    """Compute a deterministic hash consistent across Python runs"""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

def get_system_info():
    """Collect system information for reference"""
    info = {
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_compute_capability': torch.cuda.get_device_capability(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Try to get library versions
    try:
        import flash_mla
        info['flashmla_version'] = getattr(flash_mla, '__version__', 'unknown')
    except ImportError:
        info['flashmla_version'] = 'not_installed'

    return info

def quantize_fp8(tensor, scale=None):
    """Quantize tensor to FP8 E4M3 format"""
    if scale is None:
        # Compute scale from tensor statistics
        amax = tensor.abs().max().item()  # Extract scalar value
        scale = amax / 448.0  # FP8 E4M3 max representable value
        scale = max(scale, 1e-12)  # Avoid division by zero

    # Quantize
    scaled = tensor / scale
    # Clamp to FP8 range and convert
    fp8_tensor = torch.clamp(scaled, -448.0, 448.0).to(torch.float8_e4m3fn)

    return fp8_tensor, scale

def generate_test_data(config):
    """Generate test data with fixed seed for reproducibility"""
    b = config['batch_size']
    s_q = config.get('s_q', 1)
    s_kv = config.get('s_kv', 128)
    h_q = config['num_heads_q']
    h_kv = config.get('num_heads_kv', 1)
    d_qk = config['head_dim_qk']
    d_v = config['head_dim_v']
    topk = config['topk']

    # Initialize block_table (only used for decode mode)
    block_table = None

    # Generate Q, K, V with specific seed for this config (deterministic hash)
    seed_offset = deterministic_hash(str(config)) % 1000
    torch.manual_seed(RANDOM_SEED + seed_offset)

    if 'prefill' in config.get('mode', 'decode'):
        # Prefill mode: Q is (s_q, h_q, d_qk)
        Q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device='cuda') * 0.1
        KV = torch.randn(b, s_kv, h_kv, d_qk, dtype=torch.bfloat16, device='cuda') * 0.1

        # Generate sparse indices (sorted, no -1)
        indices = torch.zeros(b, s_q, h_kv, topk, dtype=torch.int32, device='cuda')
        for i in range(b):
            for j in range(s_q):
                for k in range(h_kv):
                    # Random sample of positions, sorted
                    valid_positions = torch.randperm(s_kv, device='cuda')[:topk].sort()[0]
                    indices[i, j, k] = valid_positions
    else:
        # Decode mode: Q is (b, s_q, h_q, d_qk)
        Q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device='cuda') * 0.1

        # KV cache: (num_blocks, block_size, h_kv, d_qk)
        num_blocks = config.get('num_blocks', 32)
        block_size = config.get('block_size', 64)
        KV = torch.randn(num_blocks, block_size, h_kv, d_qk, dtype=torch.bfloat16, device='cuda') * 0.1

        # Generate sparse indices based on pattern
        indices = torch.zeros(b, s_q, topk, dtype=torch.int32, device='cuda')
        pattern = config.get('indices_pattern', 'normal')

        if pattern == 'all_negative':
            # All -1 (no valid tokens)
            indices[:] = -1
        elif pattern == 'single_valid':
            # One valid token, rest -1
            indices[:] = -1
            indices[:, :, 0] = torch.randint(0, min(s_kv, num_blocks * block_size), (b, s_q), device='cuda')
        elif pattern == 'sparse_80':
            # 80% valid, 20% -1
            for i in range(b):
                for j in range(s_q):
                    max_pos = min(s_kv, num_blocks * block_size)
                    num_valid = int(topk * 0.8)
                    valid_positions = torch.randperm(max_pos, device='cuda')[:num_valid].sort()[0]
                    indices[i, j, :num_valid] = valid_positions
                    indices[i, j, num_valid:] = -1
        elif pattern == 'all_same':
            # All indices point to same token
            same_idx = torch.randint(0, min(s_kv, num_blocks * block_size), (1,), device='cuda').item()
            indices[:] = same_idx
        else:
            # Normal: random valid positions
            for i in range(b):
                for j in range(s_q):
                    max_pos = min(s_kv, num_blocks * block_size)
                    valid_positions = torch.randperm(max_pos, device='cuda')[:topk].sort()[0]
                    indices[i, j] = valid_positions

        # Block table
        block_table = torch.arange(num_blocks, dtype=torch.int32, device='cuda').unsqueeze(0).expand(b, -1)

    return {
        'Q': Q,
        'KV': KV,
        'indices': indices,
        'block_table': block_table,  # None for prefill, tensor for decode
    }

# API functions removed - using official repo APIs directly in test functions

def test_flashmla_decode(config):
    """Test FlashMLA decode kernel and capture outputs"""
    print(f"  Testing FlashMLA Decode: {config['name']}")

    try:
        fwd_kvcache_mla = torch.ops._flashmla_C.fwd_kvcache_mla
        get_mla_metadata = torch.ops._flashmla_C.get_mla_decoding_metadata
    except Exception as e:
        return {'error': f'Failed to load FlashMLA: {e}'}

    # Generate test data
    data = generate_test_data(config)

    # Handle precision configuration
    precision = config.get('precision', 'fp8')
    if precision == 'bf16':
        # BF16 path: keep KV as-is, use dummy scale
        KV_fp8 = data['KV'].to(torch.float8_e4m3fn)  # Cast for API compatibility
        scale = 1.0
    elif precision == 'fp8':
        # FP8 path: quantize with specified or computed scale
        if 'fp8_scale' in config:
            scale = config['fp8_scale']
            scaled = data['KV'] / scale
            KV_fp8 = torch.clamp(scaled, -448.0, 448.0).to(torch.float8_e4m3fn)
        else:
            KV_fp8, scale = quantize_fp8(data['KV'])
    else:
        # Default to FP8
        KV_fp8, scale = quantize_fp8(data['KV'])

    # Get metadata
    b = config['batch_size']
    s_q = config.get('s_q', 1)
    topk = config['topk']

    with torch.no_grad():
        try:
            cache_lens = torch.full((b,), 4096, dtype=torch.int32, device='cuda')
            metadata = get_mla_metadata(
                b, s_q, topk,
                cache_lens,
                64,  # block_size
                0,   # fixed_overhead
                1    # num_sm_parts
            )

            # Run kernel
            output = torch.zeros(
                b, s_q, config['num_heads_q'], config['head_dim_v'],
                dtype=torch.bfloat16, device='cuda'
            )
            lse = torch.zeros(
                b, s_q, config['num_heads_q'],
                dtype=torch.float32, device='cuda'
            )

            torch.cuda.synchronize()  # Ensure previous ops complete
            start = time.time()
            fwd_kvcache_mla(
                data['Q'], KV_fp8, data['indices'], data['block_table'],
                output, lse,
                metadata['tile_scheduler_metadata'],
                metadata['num_splits'],
                metadata['total_num_splits'],
                None, None,  # split accumulators
                scale, scale,
                0.08838834764831845,  # sm_scale for d_qk=128
                0.08838834764831845 / np.log(2),  # sm_scale_log2
            )
            torch.cuda.synchronize()
            elapsed = time.time() - start

            return {
                'output': output.cpu().numpy(),
                'lse': lse.cpu().numpy(),
                'output_shape': list(output.shape),
                'output_stats': {
                    'min': float(output.min().item()),
                    'max': float(output.max().item()),
                    'mean': float(output.mean().item()),
                    'std': float(output.std().item()),
                    'num_nan': int(torch.isnan(output).sum().item()),
                    'num_inf': int(torch.isinf(output).sum().item()),
                },
                'lse_stats': {
                    'min': float(lse.min().item()),
                    'max': float(lse.max().item()),
                    'mean': float(lse.mean().item()),
                    'num_nan': int(torch.isnan(lse).sum().item()),
                    'num_inf': int(torch.isinf(lse).sum().item()),
                },
                'time_ms': elapsed * 1000,
                'success': True,
            }
        except Exception as e:
            return {'error': str(e), 'success': False}

def test_flashmla_prefill(config):
    """Test FlashMLA prefill kernel and capture outputs"""
    print(f"  Testing FlashMLA Prefill: {config['name']}")

    try:
        sparse_prefill_fwd = torch.ops._flashmla_C.sparse_prefill_fwd
    except Exception as e:
        return {'error': f'Failed to load FlashMLA: {e}'}

    # Generate test data
    data = generate_test_data(config)

    b = config['batch_size']
    s_q = config.get('s_q', 4)
    s_kv = config.get('s_kv', 128)
    h_q = config['num_heads_q']
    h_kv = config.get('num_heads_kv', 1)
    d_v = config['head_dim_v']
    topk = config['topk']

    with torch.no_grad():
        try:
            output = torch.zeros(b, s_q, h_q, d_v, dtype=torch.bfloat16, device='cuda')
            lse = torch.zeros(b, s_q, h_q, dtype=torch.float32, device='cuda')
            max_logits = torch.zeros(b, s_q, h_q, dtype=torch.float32, device='cuda')

            torch.cuda.synchronize()  # Ensure previous ops complete
            start = time.time()
            sparse_prefill_fwd(
                data['Q'], data['KV'], data['indices'],
                output, max_logits, lse,
                0.08838834764831845,  # sm_scale
                0.08838834764831845 / np.log(2),  # sm_scale_div_log2
            )
            torch.cuda.synchronize()
            elapsed = time.time() - start

            return {
                'output': output.cpu().numpy(),
                'lse': lse.cpu().numpy(),
                'max_logits': max_logits.cpu().numpy(),
                'output_shape': list(output.shape),
                'output_stats': {
                    'min': float(output.min().item()),
                    'max': float(output.max().item()),
                    'mean': float(output.mean().item()),
                    'std': float(output.std().item()),
                    'num_nan': int(torch.isnan(output).sum().item()),
                    'num_inf': int(torch.isinf(output).sum().item()),
                },
                'lse_stats': {
                    'min': float(lse.min().item()),
                    'max': float(lse.max().item()),
                    'mean': float(lse.mean().item()),
                    'num_nan': int(torch.isnan(lse).sum().item()),
                    'num_inf': int(torch.isinf(lse).sum().item()),
                },
                'time_ms': elapsed * 1000,
                'success': True,
            }
        except Exception as e:
            return {'error': str(e), 'success': False}

def test_deepgemm(config):
    """Test DeepGEMM MQA logits kernel and capture outputs"""
    print(f"  Testing DeepGEMM: {config['name']}")

    try:
        from deep_gemm.impls.sm100_fp8_paged_mqa_logits import sm100_fp8_paged_mqa_logits
    except Exception as e:
        return {'error': f'Failed to load DeepGEMM: {e}'}

    # Generate test data
    b = config['batch_size']
    h_q = config['num_heads_q']
    d = config['head_dim_qk']
    max_seqlen = config.get('max_seqlen', 2048)
    topk = config['topk']

    torch.manual_seed(RANDOM_SEED + deterministic_hash(config['name']) % 1000)

    with torch.no_grad():
        try:
            # Q: (b, h_q, d)
            Q = torch.randn(b, h_q, d, dtype=torch.bfloat16, device='cuda') * 0.1

            # KV cache (FP8): (num_blocks, block_size, d)
            num_blocks = 32
            block_size = 64
            KV = torch.randn(num_blocks, block_size, d, dtype=torch.bfloat16, device='cuda') * 0.1
            KV_fp8 = KV.to(torch.float8_e4m3fn)  # Convert BF16 to FP8
            scale = torch.tensor([0.01], dtype=torch.float32, device='cuda')

            # Indices: (b, topk)
            indices = torch.randint(0, num_blocks * block_size, (b, topk), dtype=torch.int32, device='cuda')
            indices = indices.sort(dim=1)[0]

            # Block table
            block_table = torch.arange(num_blocks, dtype=torch.int32, device='cuda').unsqueeze(0).expand(b, -1)

            # Output
            logits = torch.zeros(b, topk, dtype=torch.bfloat16, device='cuda')

            torch.cuda.synchronize()  # Ensure previous ops complete
            start = time.time()
            sm100_fp8_paged_mqa_logits(
                Q, KV_fp8, indices, block_table,
                logits,
                scale, scale,
                0.08838834764831845,  # sm_scale
            )
            torch.cuda.synchronize()
            elapsed = time.time() - start

            return {
                'logits': logits.cpu().numpy(),
                'logits_shape': list(logits.shape),
                'logits_stats': {
                    'min': float(logits.min().item()),
                    'max': float(logits.max().item()),
                    'mean': float(logits.mean().item()),
                    'std': float(logits.std().item()),
                    'num_nan': int(torch.isnan(logits).sum().item()),
                    'num_inf': int(torch.isinf(logits).sum().item()),
                },
                'time_ms': elapsed * 1000,
                'success': True,
            }
        except Exception as e:
            return {'error': str(e), 'success': False}

def main():
    parser = argparse.ArgumentParser(description='Generate SM100 reference outputs')
    parser.add_argument('--output', type=str, default='sm100_reference.pkl',
                        help='Output file path (default: sm100_reference.pkl)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick subset of tests (30 seconds)')
    args = parser.parse_args()

    print("=" * 80)
    print("SM100 Reference Output Generator")
    print("=" * 80)

    # Validate CUDA environment
    print("\n[Validating CUDA environment...]")
    validate_environment()
    print("  ✓ CUDA available and working")

    # Collect system info
    print("\n[1/6] Collecting system information...")
    system_info = get_system_info()
    print(f"  GPU: {system_info['gpu_name']}")
    print(f"  Compute Capability: {system_info['gpu_compute_capability']}")
    print(f"  CUDA: {system_info['cuda_version']}")

    results = {
        'system_info': system_info,
        'random_seed': RANDOM_SEED,
        'flashmla_decode': {},
        'flashmla_prefill': {},
        'deepgemm': {},
        'edge_cases': {},
        'mixed_precision': {},
    }

    # Define test configurations
    if args.quick:
        decode_configs = [
            {'name': 'small', 'batch_size': 1, 's_q': 1, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 32, 'mode': 'decode'},
        ]
        prefill_configs = [
            {'name': 'small', 'batch_size': 1, 's_q': 4, 's_kv': 128, 'num_heads_q': 32, 'num_heads_kv': 1, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 32, 'mode': 'prefill'},
        ]
        deepgemm_configs = [
            {'name': 'small', 'batch_size': 1, 'num_heads_q': 32, 'head_dim_qk': 128, 'topk': 32},
        ]
        edge_case_configs = []
        mixed_precision_configs = []
    else:
        decode_configs = [
            # Small: single batch, minimal tokens
            {'name': 'small_single', 'batch_size': 1, 's_q': 1, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 32, 'mode': 'decode'},
            # Medium: multi-batch, standard config
            {'name': 'medium_multi', 'batch_size': 4, 's_q': 2, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 64, 'mode': 'decode'},
            # Large: many query tokens (critical for s_q > 1 bug)
            {'name': 'large_s_q', 'batch_size': 2, 's_q': 8, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 64, 'mode': 'decode'},
            # DeepSeek V3 config
            {'name': 'deepseek_v3', 'batch_size': 1, 's_q': 1, 'num_heads_q': 128, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 128, 'mode': 'decode'},
        ]

        prefill_configs = [
            # Small: single batch, few queries
            {'name': 'small_single', 'batch_size': 1, 's_q': 4, 's_kv': 128, 'num_heads_q': 32, 'num_heads_kv': 1, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 32, 'mode': 'prefill'},
            # Medium: multi-batch
            {'name': 'medium_multi', 'batch_size': 4, 's_q': 8, 's_kv': 256, 'num_heads_q': 32, 'num_heads_kv': 1, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 64, 'mode': 'prefill'},
            # Long sequence
            {'name': 'long_seq', 'batch_size': 1, 's_q': 4, 's_kv': 2048, 'num_heads_q': 32, 'num_heads_kv': 1, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 128, 'mode': 'prefill'},
        ]

        deepgemm_configs = [
            # Small
            {'name': 'small', 'batch_size': 1, 'num_heads_q': 32, 'head_dim_qk': 128, 'topk': 32},
            # Medium
            {'name': 'medium', 'batch_size': 4, 'num_heads_q': 32, 'head_dim_qk': 128, 'topk': 64},
            # DeepSeek V3
            {'name': 'deepseek_v3', 'batch_size': 1, 'num_heads_q': 128, 'head_dim_qk': 128, 'topk': 128},
        ]

        # Edge case configurations (special indices patterns)
        edge_case_configs = [
            # All -1 indices (no valid tokens)
            {'name': 'edge_all_negative', 'batch_size': 2, 's_q': 2, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 32, 'mode': 'decode', 'indices_pattern': 'all_negative'},
            # Single valid token (rest -1)
            {'name': 'edge_single_valid', 'batch_size': 2, 's_q': 2, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 32, 'mode': 'decode', 'indices_pattern': 'single_valid'},
            # Sparse realistic (80% valid, 20% -1)
            {'name': 'edge_sparse_80', 'batch_size': 2, 's_q': 2, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 64, 'mode': 'decode', 'indices_pattern': 'sparse_80'},
            # All same index (degenerate attention to single token)
            {'name': 'edge_all_same', 'batch_size': 2, 's_q': 2, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 32, 'mode': 'decode', 'indices_pattern': 'all_same'},
        ]

        # Mixed precision configurations (BF16 vs FP8 paths)
        mixed_precision_configs = [
            # BF16 path (non-quantized KV)
            {'name': 'precision_bf16', 'batch_size': 2, 's_q': 2, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 32, 'mode': 'decode', 'precision': 'bf16'},
            # FP8 path with different scales
            {'name': 'precision_fp8_small_scale', 'batch_size': 2, 's_q': 2, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 32, 'mode': 'decode', 'precision': 'fp8', 'fp8_scale': 0.001},
            {'name': 'precision_fp8_large_scale', 'batch_size': 2, 's_q': 2, 'num_heads_q': 32, 'head_dim_qk': 128, 'head_dim_v': 512, 'topk': 32, 'mode': 'decode', 'precision': 'fp8', 'fp8_scale': 0.1},
        ]

    # Test FlashMLA Decode
    print("\n[2/6] Testing FlashMLA Decode kernels...")
    for config in decode_configs:
        result = test_flashmla_decode(config)
        results['flashmla_decode'][config['name']] = result
        if result.get('success'):
            print(f"    ✓ {config['name']}: {result['output_stats']['mean']:.6f} mean, {result['time_ms']:.2f}ms")
        else:
            print(f"    ✗ {config['name']}: {result.get('error', 'unknown error')}")

    # Test FlashMLA Prefill
    print("\n[3/6] Testing FlashMLA Prefill kernels...")
    for config in prefill_configs:
        result = test_flashmla_prefill(config)
        results['flashmla_prefill'][config['name']] = result
        if result.get('success'):
            print(f"    ✓ {config['name']}: {result['output_stats']['mean']:.6f} mean, {result['time_ms']:.2f}ms")
        else:
            print(f"    ✗ {config['name']}: {result.get('error', 'unknown error')}")

    # Test DeepGEMM
    print("\n[4/6] Testing DeepGEMM kernels...")
    for config in deepgemm_configs:
        result = test_deepgemm(config)
        results['deepgemm'][config['name']] = result
        if result.get('success'):
            print(f"    ✓ {config['name']}: {result['logits_stats']['mean']:.6f} mean, {result['time_ms']:.2f}ms")
        else:
            print(f"    ✗ {config['name']}: {result.get('error', 'unknown error')}")

    # Test Edge Cases
    if edge_case_configs:
        print("\n[5/6] Testing Edge Case Indices...")
        for config in edge_case_configs:
            result = test_flashmla_decode(config)
            results['edge_cases'][config['name']] = result
            if result.get('success'):
                print(f"    ✓ {config['name']}: {result['output_stats']['mean']:.6f} mean, "
                      f"NaN={result['output_stats']['num_nan']}, {result['time_ms']:.2f}ms")
            else:
                print(f"    ✗ {config['name']}: {result.get('error', 'unknown error')}")

    # Test Mixed Precision
    if mixed_precision_configs:
        print("\n[6/6] Testing Mixed Precision Paths...")
        for config in mixed_precision_configs:
            result = test_flashmla_decode(config)
            results['mixed_precision'][config['name']] = result
            if result.get('success'):
                print(f"    ✓ {config['name']}: {result['output_stats']['mean']:.6f} mean, {result['time_ms']:.2f}ms")
            else:
                print(f"    ✗ {config['name']}: {result.get('error', 'unknown error')}")

    # Save results
    print(f"\n[DONE] Saving results to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)  # Protocol 4 for compatibility

    # Also save human-readable summary
    summary_path = output_path.with_suffix('.json')
    summary = {
        'system_info': system_info,
        'summary': {
            'flashmla_decode': {k: v.get('output_stats', {}) for k, v in results['flashmla_decode'].items()},
            'flashmla_prefill': {k: v.get('output_stats', {}) for k, v in results['flashmla_prefill'].items()},
            'deepgemm': {k: v.get('logits_stats', {}) for k, v in results['deepgemm'].items()},
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Verify saved files
    print(f"\n[Verifying saved files...]")
    try:
        with open(output_path, 'rb') as f:
            verify_results = pickle.load(f)
        assert verify_results['random_seed'] == RANDOM_SEED
        print(f"  ✓ Verified {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  ✓ Verified {summary_path.name} ({summary_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"  ✗ ERROR: Failed to verify output: {e}")
        sys.exit(1)

    print(f"\nResults saved to:")
    print(f"  - {args.output} (full data with arrays)")
    print(f"  - {summary_path} (human-readable summary)")

    print("\n" + "=" * 80)
    print("Reference generation complete!")
    print("Copy these files to your local machine and use compare_sm100_sm120.py to compare.")
    print("=" * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print("\n" + "=" * 80)
        print("FATAL ERROR - Script failed!")
        print("=" * 80)
        print(f"\nError: {e}\n")
        traceback.print_exc()

        # Try to save error log
        try:
            print("\nAttempting to save error log...")
            with open('error_log.txt', 'w') as f:
                f.write(f"Fatal error during SM100 reference generation\n")
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
            print("  ✓ Error log saved to error_log.txt")
        except:
            pass

        sys.exit(1)
