# -*- coding: utf-8 -*-
"""FAISS vector index + metadata persistence."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np


class FaissVectorIndex:
    def __init__(
        self,
        dim: int,
        index_dir: str | Path | None = None,
    ) -> None:
        self.dim = dim
        self.index_dir = Path(
            index_dir or Path(__file__).resolve().parent.parent / "data" / "faiss"
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.index_dir / "index.faiss"
        self._meta_path = self.index_dir / "meta.pkl"
        self.index = faiss.IndexFlatIP(dim)
        self.meta: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.meta = []

    def load(self) -> bool:
        if not self._index_path.is_file() or not self._meta_path.is_file():
            return False
        self.index = faiss.read_index(str(self._index_path))
        self.dim = int(self.index.d)
        with self._meta_path.open("rb") as f:
            self.meta = pickle.load(f)
        return True

    @classmethod
    def try_load(cls, index_dir: str | Path) -> FaissVectorIndex | None:
        """Load index from disk; infer vector dim from FAISS file."""
        index_dir = Path(index_dir)
        ip = index_dir / "index.faiss"
        mp = index_dir / "meta.pkl"
        if not ip.is_file() or not mp.is_file():
            return None
        index = faiss.read_index(str(ip))
        dim = int(index.d)
        obj = cls(dim, index_dir)
        obj.index = index
        with mp.open("rb") as f:
            obj.meta = pickle.load(f)
        return obj

    def save(self) -> None:
        faiss.write_index(self.index, str(self._index_path))
        with self._meta_path.open("wb") as f:
            pickle.dump(self.meta, f)

    def add(self, vectors: np.ndarray, metas: list[dict[str, Any]]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.meta.extend(metas)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[float, dict[str, Any]]]:
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        faiss.normalize_L2(query_vec)
        if self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(query_vec, k)
        out: list[tuple[float, dict[str, Any]]] = []
        for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0 or i >= len(self.meta):
                continue
            out.append((float(s), self.meta[i]))
        return out
