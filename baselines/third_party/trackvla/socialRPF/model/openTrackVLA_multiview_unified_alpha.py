from __future__ import annotations

from typing import List, Optional

import torch

from socialRPF.model.openTrackVLA_multiview_mixed import (
    ModelConfig,
    OpenTrackVLAMixed,
)


class OpenTrackVLAUnifiedAlpha(OpenTrackVLAMixed):
    """
    unified 训练专用模型包装器。

    与 OpenTrackVLAMixed 的主要区别是：
    - 训练/评估时优先接收 batch 内按样本提供的 alpha；
    - model 内部的 self.alpha_task 仅作为 fallback。
    """

    def forward_navigation(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        instructions: Optional[List[str]] = None,
        instruction_input_ids: Optional[torch.Tensor] = None,
        instruction_attention_mask: Optional[torch.Tensor] = None,
        yaw_hist: Optional[torch.Tensor] = None,
        yaw_curr: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if alpha is not None:
            valid_shape = (
                (alpha.dim() == 2 and alpha.size(-1) == self.action_dims)
                or (alpha.dim() == 3 and alpha.size(1) == 1 and alpha.size(-1) == self.action_dims)
            )
            if not valid_shape:
                raise ValueError(
                    f"alpha must have shape (B, {self.action_dims}) or (B, 1, {self.action_dims}), got {tuple(alpha.shape)}"
                )

        return super().forward_navigation(
            coarse_tokens=coarse_tokens,
            coarse_tidx=coarse_tidx,
            fine_tokens=fine_tokens,
            fine_tidx=fine_tidx,
            instructions=instructions,
            instruction_input_ids=instruction_input_ids,
            instruction_attention_mask=instruction_attention_mask,
            yaw_hist=yaw_hist,
            yaw_curr=yaw_curr,
            alpha=alpha,
        )
