import gc
from dataclasses import dataclass
from pathlib import Path
import threading

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.questions import (
    build_prompt,
    clean_generated_answer,
    load_chunk_lookup,
    retrieve,
)


@dataclass(frozen=True)
class QueryOptions:
    model_path: str
    embedding_model: str
    chunks_path: Path
    collection_name: str
    qdrant_url: str
    top_k: int
    candidate_multiplier: int
    max_new_tokens: int
    show_context: bool
    do_sample: bool
    enable_thinking: bool
    temperature: float
    top_p: float
    repetition_penalty: float
    no_repeat_ngram_size: int


class RagWebState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embedder_name: str | None = None
        self._embedder: SentenceTransformer | None = None
        self._chunks_path: Path | None = None
        self._chunk_lookup: dict[str, dict] | None = None
        self._model_path: str | None = None
        self._tokenizer = None
        self._model = None

    def answer(self, question: str, options: QueryOptions) -> dict:
        with self._lock:
            embedder = self._get_embedder(options.embedding_model)
            chunk_lookup = self._get_chunk_lookup(options.chunks_path)
            contexts = retrieve(
                question,
                embedder=embedder,
                qdrant_url=options.qdrant_url,
                collection_name=options.collection_name,
                chunk_lookup=chunk_lookup,
                chunks_path=options.chunks_path,
                embedding_model_name=options.embedding_model,
                top_k=options.top_k,
                candidate_multiplier=options.candidate_multiplier,
            )
            self._ensure_model(options.model_path)
            prompt = build_prompt(
                self._tokenizer,
                question,
                contexts,
                enable_thinking=options.enable_thinking,
            )
            answer = self._generate(options.model_path, prompt, options)

        return {
            "answer": answer,
            "contexts": [
                self._format_context(ctx, options.show_context)
                for ctx in contexts
            ],
        }

    def _format_context(self, ctx: dict, show_context: bool) -> dict:
        payload = {
            "title": ctx.get("title"),
            "url": ctx.get("url"),
            "section": ctx.get("section"),
            "subsection": ctx.get("subsection"),
            "score": ctx.get("score"),
        }
        if show_context:
            payload["text"] = ctx.get("text")

        return payload

    def _get_embedder(self, embedding_model: str) -> SentenceTransformer:
        if self._embedder is None or self._embedder_name != embedding_model:
            self._embedder = SentenceTransformer(embedding_model)
            self._embedder_name = embedding_model

        return self._embedder

    def _get_chunk_lookup(self, chunks_path: Path) -> dict[str, dict]:
        if self._chunk_lookup is None or self._chunks_path != chunks_path:
            self._chunk_lookup = load_chunk_lookup(chunks_path)
            self._chunks_path = chunks_path

        return self._chunk_lookup

    def _generate(self, model_path: str, prompt: str, options: QueryOptions) -> str:
        self._ensure_model(model_path)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        generation_args = {
            "max_new_tokens": options.max_new_tokens,
            "do_sample": options.do_sample,
            "repetition_penalty": options.repetition_penalty,
            "no_repeat_ngram_size": options.no_repeat_ngram_size,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if options.do_sample:
            generation_args["temperature"] = options.temperature
            generation_args["top_p"] = options.top_p

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **generation_args,
            )

        answer = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return clean_generated_answer(answer)

    def _ensure_model(self, model_path: str) -> None:
        if self._model is not None and self._model_path == model_path:
            return

        self._unload_model()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self._model_path = model_path

    def _unload_model(self) -> None:
        self._tokenizer = None
        self._model = None
        self._model_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

