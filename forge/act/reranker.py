"""Cross-encoder reranker (bge-reranker-v2-m3).

Unlike the bi-encoder embedder (query and passage encoded separately, cosine
compared), a cross-encoder takes the (query, passage) pair jointly and outputs
a relevance score -- slower but far more accurate, so it is used to rerank the
top-N candidates of the cheap first stage (standard RAG pipeline practice).

Scores are raw logits; apply sigmoid for a [0, 1] relevance probability
(used by the verbatim channel threshold, R3).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker:
    def __init__(self, model_dir: str, device: str = "cuda:1",
                 batch_size: int = 16, max_length: int = 1024, use_fp16: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, torch_dtype=torch.float16 if use_fp16 else torch.float32,
        ).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    @torch.no_grad()
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """pairs: [(query, passage), ...] -> sigmoid relevance in [0, 1]."""
        out: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            inputs = self.tokenizer(
                [q for q, _ in batch], [p for _, p in batch],
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits.view(-1).float()
            out.extend(torch.sigmoid(logits).cpu().tolist())
        return out
