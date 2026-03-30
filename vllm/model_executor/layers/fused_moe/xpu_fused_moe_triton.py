# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops

XPU_TRITON_FUSED_MOE_AUTOTUNE_PARAM_SPACE = {
    "BLOCK_SIZE_N": [32, 64, 128],
    "BLOCK_SIZE_K": [32, 64, 128],
    "GROUP_SIZE_M": [16, 32],
    "NFUSED_N": [1, 2, 4],
}

XPU_TRITON_FUSED_MOE_AUTOTUNE_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "GROUP_SIZE_M": group_size_m,
            "NFUSED_N": nfused_n,
        },
        num_stages=1 if block_size_n == 32 else 2,
        num_warps=4 if block_size_n == 32 else 8,
    )
    for block_size_n in XPU_TRITON_FUSED_MOE_AUTOTUNE_PARAM_SPACE[
        "BLOCK_SIZE_N"
    ]
    for block_size_k in XPU_TRITON_FUSED_MOE_AUTOTUNE_PARAM_SPACE[
        "BLOCK_SIZE_K"
    ]
    for group_size_m in XPU_TRITON_FUSED_MOE_AUTOTUNE_PARAM_SPACE[
        "GROUP_SIZE_M"
    ]
    for nfused_n in XPU_TRITON_FUSED_MOE_AUTOTUNE_PARAM_SPACE["NFUSED_N"]
    if (block_size_n, group_size_m) in ((32, 32), (64, 16))
    if not (block_size_n == 64 and nfused_n == 4)
]

XPU_TRITON_FUSED_MOE_TUNING_MODE_ENV = (
    "VLLM_XPU_FUSED_MOE_TRITON_TUNING_MODE"
)
XPU_TRITON_FUSED_MOE_FIXED_META_ENVS = {
    "BLOCK_SIZE_N": "VLLM_XPU_FUSED_MOE_TRITON_BLOCK_SIZE_N",
    "BLOCK_SIZE_K": "VLLM_XPU_FUSED_MOE_TRITON_BLOCK_SIZE_K",
    "GROUP_SIZE_M": "VLLM_XPU_FUSED_MOE_TRITON_GROUP_SIZE_M",
    "NFUSED_N": "VLLM_XPU_FUSED_MOE_TRITON_NFUSED_N",
    "num_warps": "VLLM_XPU_FUSED_MOE_TRITON_NUM_WARPS",
    "num_stages": "VLLM_XPU_FUSED_MOE_TRITON_NUM_STAGES",
}
XPU_TRITON_FUSED_MOE_DEFAULT_FIXED_META = {
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 32,
    "NFUSED_N": 1,
    "num_warps": 4,
    "num_stages": 1,
}


def _get_xpu_triton_fused_moe_tuning_mode() -> str:
    return os.getenv(XPU_TRITON_FUSED_MOE_TUNING_MODE_ENV,
                     "autotune").strip().lower()


def _get_xpu_triton_fused_moe_fixed_meta() -> dict[str, int]:
    meta = XPU_TRITON_FUSED_MOE_DEFAULT_FIXED_META.copy()
    for key, env_name in XPU_TRITON_FUSED_MOE_FIXED_META_ENVS.items():
        value = os.getenv(env_name)
        if value is not None:
            meta[key] = int(value)

    if (meta["BLOCK_SIZE_N"], meta["GROUP_SIZE_M"]) not in ((32, 32),
                                                               (64, 16)):
        raise ValueError(
            "Unsupported fixed Triton MoE meta combination: "
            f"BLOCK_SIZE_N={meta['BLOCK_SIZE_N']}, "
            f"GROUP_SIZE_M={meta['GROUP_SIZE_M']}."
        )
    if meta["BLOCK_SIZE_N"] == 64 and meta["NFUSED_N"] == 4:
        raise ValueError(
            "Unsupported fixed Triton MoE meta combination: "
            "BLOCK_SIZE_N=64 and NFUSED_N=4."
        )
    return meta


@triton.jit
def _fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    NFUSED_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N * NFUSED_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    if NFUSED_N >= 2:
        n_tile_base = pid_n * NFUSED_N * BLOCK_SIZE_N
        offs_bn_range = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        offs_bn0 = (n_tile_base + offs_bn_range) % N
        offs_bn1 = (n_tile_base + BLOCK_SIZE_N + offs_bn_range) % N
        if NFUSED_N == 4:
            offs_bn2 = (n_tile_base + 2 * BLOCK_SIZE_N + offs_bn_range) % N
            offs_bn3 = (n_tile_base + 3 * BLOCK_SIZE_N + offs_bn_range) % N
        offs_bn = offs_bn0
    else:
        offs_bn = (pid_n * BLOCK_SIZE_N
                   + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am
                      + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    b_ptrs = (b_ptr + off_experts * stride_be
              + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))
    if NFUSED_N >= 2:
        b_ptrs1 = (b_ptr + off_experts * stride_be
                   + (offs_k[:, None] * stride_bk
                      + offs_bn1[None, :] * stride_bn))
        if NFUSED_N == 4:
            b_ptrs2 = (b_ptr + off_experts * stride_be
                       + (offs_k[:, None] * stride_bk
                          + offs_bn2[None, :] * stride_bn))
            b_ptrs3 = (b_ptr + off_experts * stride_be
                       + (offs_k[:, None] * stride_bk
                          + offs_bn3[None, :] * stride_bn))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if NFUSED_N >= 2:
        accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                dtype=tl.float32)
        if NFUSED_N == 4:
            accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                    dtype=tl.float32)
            accumulator3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                    dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None]
                    & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        if NFUSED_N >= 2:
            b1 = tl.load(b_ptrs1,
                         mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                         other=0.0)
            if NFUSED_N == 4:
                b2 = tl.load(b_ptrs2,
                             mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                             other=0.0)
                b3 = tl.load(b_ptrs3,
                             mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                             other=0.0)
        accumulator += tl.dot(a, b)
        if NFUSED_N >= 2:
            accumulator1 += tl.dot(a, b1)
            if NFUSED_N == 4:
                accumulator2 += tl.dot(a, b2)
                accumulator3 += tl.dot(a, b3)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        if NFUSED_N >= 2:
            b_ptrs1 += BLOCK_SIZE_K * stride_bk
            if NFUSED_N == 4:
                b_ptrs2 += BLOCK_SIZE_K * stride_bk
                b_ptrs3 += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
        if NFUSED_N >= 2:
            accumulator1 = accumulator1 * moe_weight[:, None]
            if NFUSED_N == 4:
                accumulator2 = accumulator2 * moe_weight[:, None]
                accumulator3 = accumulator3 * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    if NFUSED_N >= 2:
        accumulator1 = accumulator1.to(compute_type)
        if NFUSED_N == 4:
            accumulator2 = accumulator2.to(compute_type)
            accumulator3 = accumulator3.to(compute_type)

    if NFUSED_N == 1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
            None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        offs_cn_range = tl.arange(0, BLOCK_SIZE_N)
        offs_cn0 = n_tile_base + offs_cn_range
        c_ptrs0 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn0[
            None, :]
        c_mask0 = token_mask[:, None] & (offs_cn0[None, :] < N)
        tl.store(c_ptrs0, accumulator, mask=c_mask0)

        offs_cn1 = n_tile_base + BLOCK_SIZE_N + offs_cn_range
        c_ptrs1 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn1[
            None, :]
        c_mask1 = token_mask[:, None] & (offs_cn1[None, :] < N)
        tl.store(c_ptrs1, accumulator1, mask=c_mask1)

        if NFUSED_N == 4:
            offs_cn2 = n_tile_base + 2 * BLOCK_SIZE_N + offs_cn_range
            c_ptrs2 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn2[
                None, :]
            c_mask2 = token_mask[:, None] & (offs_cn2[None, :] < N)
            tl.store(c_ptrs2, accumulator2, mask=c_mask2)

            offs_cn3 = n_tile_base + 3 * BLOCK_SIZE_N + offs_cn_range
            c_ptrs3 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn3[
                None, :]
            c_mask3 = token_mask[:, None] & (offs_cn3[None, :] < N)
            tl.store(c_ptrs3, accumulator3, mask=c_mask3)


fused_moe_kernel = triton.autotune(configs=XPU_TRITON_FUSED_MOE_AUTOTUNE_CONFIGS,
                                   key=["N", "K", "EM"])(
                                       _fused_moe_kernel)


def _moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids = torch.empty((topk_ids.numel() + num_experts * (block_size - 1),),
                             dtype=torch.int32,
                             device=topk_ids.device)
    expert_ids = torch.empty((topk_ids.numel() + num_experts,),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1,),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


def _invoke_triton_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    block_size_m: int,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"] * META["NFUSED_N"]),)

    kernel_args = (
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
    )
    kernel_kwargs = {
        "MUL_ROUTED_WEIGHT": mul_routed_weight,
        "top_k": top_k,
        "compute_type": tl.bfloat16
        if A.dtype == torch.bfloat16 else tl.float16,
        "BLOCK_SIZE_M": block_size_m,
    }

    if _get_xpu_triton_fused_moe_tuning_mode() == "fixed":
        fixed_meta = _get_xpu_triton_fused_moe_fixed_meta()
        _fused_moe_kernel[grid](
            *kernel_args,
            **kernel_kwargs,
            BLOCK_SIZE_N=fixed_meta["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=fixed_meta["BLOCK_SIZE_K"],
            GROUP_SIZE_M=fixed_meta["GROUP_SIZE_M"],
            NFUSED_N=fixed_meta["NFUSED_N"],
            num_warps=fixed_meta["num_warps"],
            num_stages=fixed_meta["num_stages"],
        )
        return

    fused_moe_kernel[grid](*kernel_args, **kernel_kwargs)


def xpu_fused_moe_triton(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scales: torch.Tensor | None,
    w13_bias: torch.Tensor | None,
    w2: torch.Tensor,
    w2_scales: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    n_experts_per_token: int,
    activation: str,
    num_experts: int,
    ep_rank: int = 0,
    ep_size: int = 1,
    expert_map: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    is_fp8: bool = False,
    is_int4: bool = False,
    is_mxfp4: bool = False,
) -> torch.Tensor:
    #print("Running XPU fused MoE with Triton kernel")
    if output is None:
        output = torch.empty_like(hidden_states)
    else:
        assert output.shape == hidden_states.shape, (
            "output shape must be the same as hidden_states shape")

    inter_size = list(w13.shape)[-2] // 2

    if w13_scales is not None or w2_scales is not None:
        raise NotImplementedError(
            "xpu_fused_moe_triton does not support quantized scales yet.")
    if w13_bias is not None or w2_bias is not None:
        raise NotImplementedError(
            "xpu_fused_moe_triton does not support expert bias yet.")
    if is_fp8 or is_int4 or is_mxfp4:
        raise NotImplementedError(
            "xpu_fused_moe_triton only supports unquantized weights.")
    if ep_rank != 0 or ep_size != 1 or expert_map is not None:
        raise NotImplementedError(
            "xpu_fused_moe_triton does not support expert parallel remapping.")

    assert hidden_states.is_contiguous()
    assert w13.is_contiguous() and w2.is_contiguous()
    assert topk_ids.size(-1) == n_experts_per_token
    assert hidden_states.shape[1] == w13.shape[2]
    assert hidden_states.shape[1] == w2.shape[1]
    assert w13.shape[0] == num_experts
    assert w2.shape[0] == num_experts

    num_rows, hidden_size = list(hidden_states.shape)
    num_moe_inputs = n_experts_per_token * num_rows

    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    if not topk_ids.is_contiguous():
        topk_ids = topk_ids.contiguous()
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    block_size_m = 16 if num_rows <= num_experts else 64

    gemm1_output = torch.empty((num_rows, n_experts_per_token, 2 * inter_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        _moe_align_block_size_triton(topk_ids, block_size_m, num_experts)

    _invoke_triton_fused_moe_kernel(hidden_states, w13, gemm1_output,
                                    topk_weights, topk_ids, sorted_token_ids,
                                    expert_ids, num_tokens_post_padded, False,
                                    n_experts_per_token, block_size_m)

    act_output = torch.empty((num_moe_inputs, inter_size),
                             dtype=gemm1_output.dtype,
                             device=gemm1_output.device)
    gemm1_output_2d = gemm1_output.view(num_moe_inputs, 2 * inter_size)
    if activation == "silu":
        torch.ops._C.silu_and_mul(act_output, gemm1_output_2d)
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(act_output, gemm1_output_2d)
    elif activation == "swigluoai":
        torch.ops._C.swigluoai_and_mul(act_output, gemm1_output_2d, 1.702, 7.0)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

    gemm2_output = torch.empty((num_rows, n_experts_per_token, hidden_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)
    _invoke_triton_fused_moe_kernel(act_output, w2, gemm2_output,
                                    topk_weights, topk_ids, sorted_token_ids,
                                    expert_ids, num_tokens_post_padded, True, 1,
                                    block_size_m)

    torch.sum(gemm2_output, dim=1, out=output)
    return output
