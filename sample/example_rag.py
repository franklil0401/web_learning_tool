# -*- coding: utf-8 -*-
# 在项目根目录运行: python example_rag.py

from rag.pipeline import RAGPipeline


def main() -> None:
    pipe = RAGPipeline()

    pipe.save_utterance("今天我们讲牛顿第二定律，F 等于 m 乘以 a�?", "teacher")
    pipe.save_utterance("加速度与合外力成正比，与质量成反比�?", "teacher")

    n = pipe.rebuild_index_from_store(source="teacher")
    print("indexed chunks:", n)

    q = "力和加速度有什么关系？"
    ctx = pipe.build_llm_context(q, retrieve_top_k=10, rerank_top_k=3)
    print("--- LLM context ---")
    print(ctx or "(empty)")


if __name__ == "__main__":
    main()
