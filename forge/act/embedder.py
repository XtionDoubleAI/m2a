"""Dense embedder: BGE-M3 via plain transformers (CLS pooling + L2 norm).

Matches the paper's dense retrieval setting (BGE-M3 dense vectors). We avoid the
FlagEmbedding package because recent versions refuse torch<2.6 with .bin weights;
the dense head of BGE-M3 is exactly last_hidden_state CLS, so this is equivalent.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class BGEM3Dense:
    def __init__(self, model_dir: str, device: str = "cuda:1", max_length: int = 1024,
                 batch_size: int = 16, use_fp16: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(
            model_dir, torch_dtype=torch.float16 if use_fp16 else torch.float32
        ).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            out = self.model(**inputs).last_hidden_state[:, 0]      # CLS
            out = torch.nn.functional.normalize(out, dim=-1)
            vecs.append(out.float().cpu().numpy())
        return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, 1024))
