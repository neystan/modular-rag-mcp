"""Chunk 稀疏向量编码。"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from typing import Any

import jieba

from core.trace import TraceContext
from core.types import Chunk, ChunkRecord

# 关闭 jieba 的 debug 日志
jieba.setLogLevel(jieba.logging.INFO)

# 英文/数字 token 规则（中文交给 jieba）
ENGLISH_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class SparseEncoder:
    """将 chunk 文本转换为 BM25 可消费的 term weight 结构。"""

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        trace_context = trace if isinstance(trace, TraceContext) else None

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            term_counts = Counter(tokens)
            sparse_vector = self._build_sparse_vector(term_counts)
            metadata = copy.deepcopy(chunk.metadata)
            metadata["sparse_doc_length"] = len(tokens)
            metadata["sparse_unique_terms"] = len(term_counts)

            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                    sparse_vector=sparse_vector,
                )
            )

        if trace_context is not None:
            trace_context.record_stage(
                "sparse_encoder.success",
                {"count": len(records), "non_empty": sum(1 for record in records if record.sparse_vector)},
            )
        return records

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("sparse encoder input error: text must be string")
        tokens: list[str] = []
        # jieba 对中文分词
        for word in jieba.cut(text):
            word = word.strip().lower()
            if not word:
                continue
            # 跳过纯标点和空白
            if not any(c.isalnum() or c == "_" or c == "-" for c in word):
                continue
            # 英文 token 进一步规范化
            for match in ENGLISH_TOKEN_PATTERN.finditer(word):
                tokens.append(match.group().lower())
            # 如果是纯中文词（jieba 分出的），直接加入
            if ENGLISH_TOKEN_PATTERN.fullmatch(word):
                continue  # 英文已处理
            if word and not ENGLISH_TOKEN_PATTERN.fullmatch(word):
                tokens.append(word)
        return tokens

    @staticmethod
    def _build_sparse_vector(term_counts: Counter[str]) -> dict[str, float]:
        if not term_counts:
            return {}

        total_terms = sum(term_counts.values())
        weights: dict[str, float] = {}
        for term, count in term_counts.items():
            # 使用子线性 TF，便于后续 BM25Indexer 直接消费或再映射。
            weights[term] = round((1.0 + math.log(count)) / total_terms, 8)
        return weights
