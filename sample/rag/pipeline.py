# -*- coding: utf-8 -*-
"""
RAG: FAISS retrieve + rerank + context string for LLM.

Embedding:
  - local: sentence-transformers (no API key)
  - dashscope: RAG_EMBEDDING_BACKEND=dashscope + DASHSCOPE_API_KEY

Rerank:
  - local: CrossEncoder
  - dashscope: RAG_RERANK_BACKEND=dashscope (e.g. gte-rerank-v2)

Full Alibaba stack example:
  RAG_EMBEDDING_BACKEND=dashscope
  RAG_RERANK_BACKEND=dashscope
  RAG_DASHSCOPE_EMBED_MODEL=text-embedding-v1
  RAG_EMBEDDING_DIM=1536
  RAG_DASHSCOPE_RERANK_MODEL=gte-rerank-v2
  RAG_QWEN_MODEL=qwen-plus-2025-07-28

pip install sentence-transformers dashscope
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import numpy as np

# Load environment variables from .env file
load_dotenv()

# Manually set DASHSCOPE_API_KEY from system environment if not set
if not os.environ.get('DASHSCOPE_API_KEY'):
    import ctypes
    # Get environment variable from user registry
    def get_env_var(name):
        buffer = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_int(1024)
        # Read from HKEY_CURRENT_USER\Environment
        result = ctypes.windll.advapi32.RegGetValueW(
            0x80000001,  # HKEY_CURRENT_USER
            "Environment",
            name,
            0,
            None,
            buffer,
            ctypes.byref(size)
        )
        if result == 0:
            return buffer.value
        return None
    
    api_key = get_env_var('DASHSCOPE_API_KEY')
    if api_key:
        os.environ['DASHSCOPE_API_KEY'] = api_key
        print(f"Set DASHSCOPE_API_KEY from system environment: {api_key[:10]}...")

from .chunking import chunk_text
from .transcript_store import Source, TranscriptStore
from .vector_index import FaissVectorIndex


class RAGPipeline:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        embedding_backend: str | None = None,
        embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dashscope_embed_model: str | None = None,
        rerank_backend: str | None = None,
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        dashscope_rerank_model: str | None = None,
        chunk_max_chars: int = 400,
        chunk_overlap: int = 80,
    ) -> None:
        self.store = TranscriptStore(data_dir)
        # Try to use dashscope first, but fallback to local if API key is missing
        try:
            if embedding_backend is None and os.environ.get("RAG_EMBEDDING_BACKEND") is None:
                # Check if API key is available
                import dashscope
                api_key = os.environ.get("DASHSCOPE_API_KEY")
                if api_key:
                    dashscope.api_key = api_key
                    self.embedding_backend = "dashscope"
                else:
                    raise RuntimeError("No DASHSCOPE_API_KEY found")
            else:
                self.embedding_backend = (
                    (embedding_backend or os.environ.get("RAG_EMBEDDING_BACKEND", "local"))
                    .strip()
                    .lower()
                )
        except Exception as e:
            print(f"Error initializing dashscope: {e}")
            print("Falling back to local embedding backend")
            self.embedding_backend = "local"
        
        try:
            if rerank_backend is None and os.environ.get("RAG_RERANK_BACKEND") is None:
                # Check if API key is available
                import dashscope
                api_key = os.environ.get("DASHSCOPE_API_KEY")
                if api_key:
                    dashscope.api_key = api_key
                    self.rerank_backend = "dashscope"
                else:
                    raise RuntimeError("No DASHSCOPE_API_KEY found")
            else:
                self.rerank_backend = (
                    (rerank_backend or os.environ.get("RAG_RERANK_BACKEND", "local"))
                    .strip()
                    .lower()
                )
        except Exception as e:
            print(f"Error initializing dashscope rerank: {e}")
            print("Falling back to local rerank backend")
            self.rerank_backend = "local"
        self._embed_model_name = embed_model_name
        self._dashscope_embed_model = (
            dashscope_embed_model
            or os.environ.get("RAG_DASHSCOPE_EMBED_MODEL", "text-embedding-v1")
        ).strip()
        self._rerank_model_name = rerank_model_name
        self._dashscope_rerank_model = (
            dashscope_rerank_model
            or os.environ.get("RAG_DASHSCOPE_RERANK_MODEL", "gte-rerank-v2")
        ).strip()
        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap = chunk_overlap
        self._sentence_transformer = None
        self._reranker = None
        self._dim: int | None = None
        self._faiss: FaissVectorIndex | None = None

    def _get_sentence_transformer(self):
        if self._sentence_transformer is None:
            from sentence_transformers import SentenceTransformer

            self._sentence_transformer = SentenceTransformer(self._embed_model_name)
            self._dim = int(self._sentence_transformer.get_sentence_embedding_dimension())
        return self._sentence_transformer

    def _ensure_dim_for_new_index(self) -> None:
        if self._dim is not None:
            return
        if self.embedding_backend == "dashscope":
            from .aliyun_dashscope import default_embedding_dim

            self._dim = default_embedding_dim()
            return
        self._get_sentence_transformer()

    def _encode_texts(
        self, texts: list[str], show_progress: bool = False
    ) -> np.ndarray:
        if self.embedding_backend == "dashscope":
            from .aliyun_dashscope import embed_texts_dashscope

            return embed_texts_dashscope(texts, model=self._dashscope_embed_model)
        st = self._get_sentence_transformer()
        emb = st.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=show_progress,
        )
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
        return np.asarray(emb, dtype=np.float32)

    def _get_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self._rerank_model_name)
            except Exception as e:
                print(f"Error loading cross encoder: {e}")
                print("Using dummy reranker for demonstration purposes")
                # Use a dummy reranker for demonstration
                class DummyReranker:
                    def predict(self, pairs):
                        return [0.5 for _ in pairs]
                self._reranker = DummyReranker()
        return self._reranker

    def _get_faiss(self) -> FaissVectorIndex:
        if self._faiss is None:
            faiss_dir = self.store.data_dir / "faiss"
            loaded = FaissVectorIndex.try_load(faiss_dir)
            if loaded is not None:
                self._faiss = loaded
                self._dim = loaded.dim
            else:
                self._ensure_dim_for_new_index()
                assert self._dim is not None
                idx = FaissVectorIndex(self._dim, faiss_dir)
                idx.load()
                self._faiss = idx
        return self._faiss

    def invalidate_faiss_cache(self) -> None:
        self._faiss = None

    def save_utterance(self, text: str, source: Source = "teacher") -> str:
        return self.store.append(text, source)

    def rebuild_index_from_store(self, source: Source | None = "teacher") -> int:
        faiss_dir = self.store.data_dir / "faiss"
        loaded = FaissVectorIndex.try_load(faiss_dir)
        if loaded is not None:
            self._dim = loaded.dim
        else:
            self._ensure_dim_for_new_index()
        assert self._dim is not None
        idx = FaissVectorIndex(self._dim, faiss_dir)
        idx.reset()

        rows = self.store.load_texts_by_source(source)
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        for utt_id, src, full in rows:
            for i, ch in enumerate(
                chunk_text(full, self.chunk_max_chars, self.chunk_overlap)
            ):
                texts.append(ch)
                metas.append(
                    {
                        "utterance_id": utt_id,
                        "source": src,
                        "chunk_index": i,
                        "text": ch,
                    }
                )
        if not texts:
            idx.save()
            self._faiss = idx
            self.invalidate_faiss_cache()
            return 0

        emb = self._encode_texts(texts, show_progress=True)
        idx.add(emb, metas)
        idx.save()
        self._faiss = idx
        return len(texts)

    def add_utterance_and_index(self, text: str, source: Source = "teacher") -> None:
        utt_id = self.save_utterance(text, source)
        if not utt_id:
            return
        faiss_dir = self.store.data_dir / "faiss"
        loaded = FaissVectorIndex.try_load(faiss_dir)
        if loaded is not None:
            self._dim = loaded.dim
            idx = loaded
        else:
            self._ensure_dim_for_new_index()
            assert self._dim is not None
            idx = FaissVectorIndex(self._dim, faiss_dir)

        chunks = chunk_text(text, self.chunk_max_chars, self.chunk_overlap)
        if not chunks:
            return
        metas = [
            {
                "utterance_id": utt_id,
                "source": source,
                "chunk_index": i,
                "text": ch,
            }
            for i, ch in enumerate(chunks)
        ]
        emb = self._encode_texts(chunks, show_progress=False)
        idx.add(emb, metas)
        idx.save()
        self.invalidate_faiss_cache()

    def retrieve_and_rerank(
        self,
        question: str,
        retrieve_top_k: int = 20,
        rerank_top_k: int = 5,
    ) -> list[dict[str, Any]]:
        question = (question or "").strip()
        if not question:
            return []

        idx = self._get_faiss()
        if idx.index.ntotal == 0:
            return []

        qv = self._encode_texts([question], show_progress=False)
        qv = np.asarray(qv, dtype=np.float32)
        hits = idx.search(qv, retrieve_top_k)
        if not hits:
            return []

        candidates = [m for _, m in hits]
        texts = [m["text"] for m in candidates]

        if self.rerank_backend == "dashscope":
            from .aliyun_dashscope import rerank_with_dashscope

            order = rerank_with_dashscope(
                question,
                texts,
                top_n=min(rerank_top_k, len(candidates)),
                model=self._dashscope_rerank_model,
            )
            out: list[dict[str, Any]] = []
            for item in order:
                i = int(item["index"])
                if 0 <= i < len(candidates):
                    m = candidates[i]
                    out.append(
                        {
                            "score": float(item["relevance_score"]),
                            "text": m["text"],
                            "meta": m,
                        }
                    )
            return out

        reranker = self._get_reranker()
        pairs = [[question, t] for t in texts]
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: float(x[0]),
            reverse=True,
        )[:rerank_top_k]

        return [
            {"score": float(s), "text": m["text"], "meta": m}
            for s, m in ranked
        ]

    def build_llm_context(
        self,
        question: str,
        retrieve_top_k: int = 20,
        rerank_top_k: int = 5,
    ) -> str:
        items = self.retrieve_and_rerank(
            question,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
        )
        if not items:
            return ""

        lines = [
            "Course content context for LLM:",
            "",
        ]
        for i, it in enumerate(items, 1):
            lines.append(f"[Content {i}] {it['text']}")
            lines.append("")
        return "\n".join(lines).strip()

    def answer_with_qwen(
        self,
        question: str,
        retrieve_top_k: int = 20,
        rerank_top_k: int = 5,
    ) -> str:
        """RAG context + DashScope Qwen (uses DASHSCOPE_API_KEY and RAG_QWEN_MODEL)."""
        from .aliyun_dashscope import answer_with_rag_context

        ctx = self.build_llm_context(
            question,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
        )
        return answer_with_rag_context(question, ctx)
    
    def clear_history(self) -> None:
        """Clear all course history information"""
        # Clear transcript store
        self.store.clear()
        # Clear vector index
        faiss_dir = self.store.data_dir / "faiss"
        if faiss_dir.exists():
            import shutil
            shutil.rmtree(faiss_dir)
        # Recreate faiss directory
        faiss_dir.mkdir(parents=True, exist_ok=True)
        # Invalidate cache
        self.invalidate_faiss_cache()
        print("Course history cleared successfully")
