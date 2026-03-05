# SPDX-License-Identifier: Apache-2.0
# Standard
import os
from dataclasses import dataclass, field

# Third Party
import torch
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class LayerwiseAttentionSync:
    """Synchronize per-layer attention outputs across distributed ranks.

    The synchronization computes a rank-global feature score vector using
    ``torch.distributed.all_reduce(SUM)``. Every rank receives the same global
    vector, so each rank can derive an identical sorting result.
    """

    enabled: bool = False
    sort_layer: int = 5
    descending: bool = True
    latest_sorted_indices: dict[int, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LayerwiseAttentionSync":
        enabled = os.getenv("LMC_ASCEND_ATTN_SYNC_ENABLED", "0") == "1"
        sort_layer = int(os.getenv("LMC_ASCEND_ATTN_SYNC_SORT_LAYER", "5"))
        descending = os.getenv("LMC_ASCEND_ATTN_SYNC_DESC", "1") == "1"
        return cls(enabled=enabled, sort_layer=sort_layer, descending=descending)

    def _dist_ready(self) -> bool:
        return (
            self.enabled
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

    def sync(self, attn_output: torch.Tensor, layer_id: int) -> torch.Tensor | None:
        """Synchronize and optionally sort global scores for a layer.

        Args:
            attn_output: Attention output tensor with shape
                ``[num_tokens, num_heads, head_size]``.
            layer_id: Zero-based layer index.

        Returns:
            Sorted indices tensor for ``sort_layer`` only. For all other layers,
            returns ``None``.
        """

        if not self._dist_ready():
            return None

        score_vec = attn_output.float().reshape(attn_output.shape[0], -1).sum(dim=0)
        torch.distributed.all_reduce(score_vec, op=torch.distributed.ReduceOp.SUM)

        if layer_id != self.sort_layer:
            return None

        sorted_indices = torch.argsort(score_vec, descending=self.descending)
        self.latest_sorted_indices[layer_id] = sorted_indices
        logger.info(
            "Layer %s global attention score top-8 indices: %s",
            layer_id,
            sorted_indices[:8].tolist(),
        )
        return sorted_indices
