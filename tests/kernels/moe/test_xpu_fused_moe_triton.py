# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if current_platform.is_xpu():
    from vllm.model_executor.layers.fused_moe.xpu_fused_moe_triton import (
        xpu_fused_moe_triton,
    )
    from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe


DEVICE = "xpu"


def _reference_fused_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_rows, hidden_size = hidden_states.shape
    intermediate_size = w2.shape[-1]
    output = torch.zeros((num_rows, hidden_size),
                         device=hidden_states.device,
                         dtype=torch.float32)

    for row_idx in range(num_rows):
        row = hidden_states[row_idx].to(torch.float32)
        for topk_idx in range(topk_ids.shape[1]):
            expert_idx = int(topk_ids[row_idx, topk_idx].item())
            gate_up = row @ w13[expert_idx].to(torch.float32).transpose(0, 1)
            gate, up = gate_up.split(intermediate_size, dim=-1)
            activated = F.silu(gate) * up
            expert_out = activated @ w2[expert_idx].to(torch.float32).transpose(0,
                                                                                 1)
            output[row_idx] += (topk_weights[row_idx, topk_idx].to(torch.float32)
                                * expert_out)

    return output.to(hidden_states.dtype)


@pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU fused MoE comparison test requires XPU.",
)
@torch.inference_mode()
def test_xpu_fused_moe_triton_matches_cutlass_and_reference():
    torch.manual_seed(0)

    num_rows = 32
    hidden_size = 128
    intermediate_size = 96
    num_experts = 8
    topk = 2
    dtype = torch.bfloat16

    hidden_states = torch.randn((num_rows, hidden_size),
                                device=DEVICE,
                                dtype=dtype) / 10
    w13 = torch.randn((num_experts, 2 * intermediate_size, hidden_size),
                      device=DEVICE,
                      dtype=dtype) / 10
    w2 = torch.randn((num_experts, hidden_size, intermediate_size),
                     device=DEVICE,
                     dtype=dtype) / 10

    scores = torch.randn((num_rows, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    topk_scores, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    topk_weights = torch.softmax(topk_scores, dim=-1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    reference = _reference_fused_moe(hidden_states, w13, w2, topk_weights,
                                     topk_ids)

    cutlass_output = xpu_fused_moe(
        hidden_states=hidden_states.clone(),
        w13=w13.clone(),
        w13_scales=None,
        w13_bias=None,
        w2=w2.clone(),
        w2_scales=None,
        w2_bias=None,
        topk_weights=topk_weights.clone(),
        topk_ids=topk_ids.clone(),
        n_experts_per_token=topk,
        activation="silu",
        num_experts=num_experts,
    )

    triton_output = xpu_fused_moe_triton(
        hidden_states=hidden_states.clone(),
        w13=w13.clone(),
        w13_scales=None,
        w13_bias=None,
        w2=w2.clone(),
        w2_scales=None,
        w2_bias=None,
        topk_weights=topk_weights.clone(),
        topk_ids=topk_ids.clone(),
        n_experts_per_token=topk,
        activation="silu",
        num_experts=num_experts,
    )

    torch.testing.assert_close(cutlass_output,
                               reference,
                               atol=1e-2,
                               rtol=1e-2)
    torch.testing.assert_close(triton_output,
                               reference,
                               atol=1e-2,
                               rtol=1e-2)
    torch.testing.assert_close(triton_output,
                               cutlass_output,
                               atol=1e-3,
                               rtol=1e-3)