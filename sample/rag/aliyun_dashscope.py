# -*- coding: utf-8 -*-
"""
Alibaba DashScope (Model Studio): embedding + rerank + Qwen chat.
One API key for all: set DASHSCOPE_API_KEY (never commit the real value).

Recommended env (full cloud stack, no local ST / CrossEncoder for RAG):
    RAG_EMBEDDING_BACKEND=dashscope
    RAG_RERANK_BACKEND=dashscope
    RAG_DASHSCOPE_EMBED_MODEL=text-embedding-v1
    RAG_EMBEDDING_DIM=1536
    RAG_DASHSCOPE_RERANK_MODEL=gte-rerank-v2
    RAG_QWEN_MODEL=qwen-plus-2025-07-28

Install: pip install dashscope
Docs: https://help.aliyun.com/zh/model-studio/
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Defaults aligned with your stack (overridable via env)
_DEFAULT_EMBED_MODEL = "text-embedding-v1"
_DEFAULT_RERANK_MODEL = "gte-rerank-v2"
_DEFAULT_QWEN_MODEL = "qwen-plus-2025-07-28"
_DEFAULT_EMBED_DIM = 1536  # text-embedding-v1


def get_dashscope_api_key() -> str:
    key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        # Try to get from system registry
        import ctypes
        def get_env_var(name):
            buffer = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_int(1024)
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
        key = get_env_var("DASHSCOPE_API_KEY") or ""
    if not key:
        raise RuntimeError(
            "Missing DASHSCOPE_API_KEY. Set it in environment or a local .env file."
        )
    return key


def embed_texts_dashscope(
    texts: list[str],
    model: str | None = None,
) -> np.ndarray:
    """DashScope Text Embedding. Returns (n, dim) float32."""
    import dashscope
    from dashscope import TextEmbedding

    model = (model or os.environ.get("RAG_DASHSCOPE_EMBED_MODEL") or _DEFAULT_EMBED_MODEL).strip()
    dashscope.api_key = get_dashscope_api_key()

    all_rows: list[list[float]] = []
    batch_size = 25
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = TextEmbedding.call(model=model, input=batch)
        code = getattr(resp, "status_code", None)
        if code != 200:
            msg = getattr(resp, "message", str(resp))
            raise RuntimeError(f"DashScope embedding failed: {msg}")
        out = resp.output
        if isinstance(out, dict):
            emb_list = out.get("embeddings") or []
        else:
            emb_list = getattr(out, "embeddings", None) or []
        if not emb_list:
            raise RuntimeError(f"DashScope embedding empty output: {resp}")
        for item in emb_list:
            if isinstance(item, dict):
                all_rows.append(item["embedding"])
            else:
                all_rows.append(getattr(item, "embedding"))
    return np.asarray(all_rows, dtype=np.float32)


def rerank_with_dashscope(
    query: str,
    documents: list[str],
    top_n: int,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    DashScope TextReRank. Returns list of {index, relevance_score} in API order (best first).
    """
    from dashscope import TextReRank

    model = (model or os.environ.get("RAG_DASHSCOPE_RERANK_MODEL") or _DEFAULT_RERANK_MODEL).strip()
    if not documents:
        return []
    top_n = min(max(top_n, 1), len(documents))
    resp = TextReRank.call(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False,
        api_key=get_dashscope_api_key(),
    )
    code = getattr(resp, "status_code", None)
    if code != 200:
        msg = getattr(resp, "message", str(resp))
        raise RuntimeError(f"DashScope rerank failed: {msg}")

    out = resp.output
    if isinstance(out, dict):
        raw_results = out.get("results") or []
    else:
        raw_results = getattr(out, "results", None) or []

    parsed: list[dict[str, Any]] = []
    for r in raw_results:
        if isinstance(r, dict):
            idx = int(r["index"])
            score = float(r.get("relevance_score", r.get("score", 0.0)))
        else:
            idx = int(getattr(r, "index"))
            score = float(getattr(r, "relevance_score", getattr(r, "score", 0.0)))
        parsed.append({"index": idx, "relevance_score": score})
    return parsed


def chat_qwen(
    user_content: str,
    system_content: str | None = None,
    model: str | None = None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return chat_qwen_messages(messages, model=model)


def chat_qwen_messages(
    messages: list[dict[str, Any]],
    model: str | None = None,
) -> str:
    import dashscope
    from dashscope import Generation

    model = (model or os.environ.get("RAG_QWEN_MODEL") or _DEFAULT_QWEN_MODEL).strip()
    dashscope.api_key = get_dashscope_api_key()
    resp = Generation.call(
        model=model,
        messages=messages,
        result_format="message",
        api_key=get_dashscope_api_key(),
    )
    code = getattr(resp, "status_code", None)
    if code != 200:
        msg = getattr(resp, "message", str(resp))
        raise RuntimeError(f"DashScope chat failed: {msg}")
    choice = resp.output["choices"][0]
    msg_obj = choice["message"]
    if isinstance(msg_obj, dict):
        content = msg_obj.get("content", "")
    else:
        content = getattr(msg_obj, "content", "")
    return content.strip() if isinstance(content, str) else str(content)


def answer_with_rag_context(
    question: str,
    context: str,
    model: str | None = None,
) -> str:
    system = (
        "You are an AI tutor for live courses. Use the provided course content to answer student questions accurately. "
        "Only use information from the provided context, do not make up any information. "
        "Keep your answers concise and focused on the student's question."
    )
    user = f"Course content:\n{context}\n\nStudent question:\n{question}"
    return chat_qwen(user, system_content=system, model=model)


def default_embedding_dim() -> int:
    """Avoid probe API call when RAG_EMBEDDING_DIM is unset."""
    explicit = os.environ.get("RAG_EMBEDDING_DIM")
    if explicit:
        return int(explicit.strip())
    return _DEFAULT_EMBED_DIM
