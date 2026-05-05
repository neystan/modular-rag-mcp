"""向量化模块。"""

from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder

__all__ = [
    "DenseEncoder",
    "SparseEncoder",
]
