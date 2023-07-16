from transformers import LogitsProcessor
import torch
from typing import List

class ForcedPrefixLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, prefix: List[int]):
        self.prefix = prefix

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len <= len(self.prefix):
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.prefix[cur_len-1]]] = -float("inf")
            scores[:, self.prefix[cur_len-1]] = 0
        return scores