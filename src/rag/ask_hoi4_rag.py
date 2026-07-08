import argparse
import json
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


COLLECTION_NAME = "hoi4_wiki"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_QDRANT_URL = "http://localhost:6333"
CHUNKS_PATH = Path("data/processed/chunks/chunks.jsonl")


def load_chunk_lookup(path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id:
                lookup[chunk_id] = chunk

    return lookup


def retrieve(
    query: str,
    embedder: SentenceTransformer,
    qdrant_url: str,
    collection_name: str,
    chunk_lookup: dict[str, dict],
    chunks_path: Path,
    top_k: int = 5,
) -> list[dict]:
    client = QdrantClient(url=qdrant_url)

    vector = embedder.encode(query, normalize_embeddings=True).tolist()

    results = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k,
    )

    contexts = []

    for hit in results:
        payload = dict(hit.payload or {})
        chunk_id = payload.get("chunk_id")
        chunk = chunk_lookup.get(chunk_id) if chunk_id else None

        if chunk is None:
            raise KeyError(
                f"Chunk '{chunk_id}' nao encontrado em {chunks_path}."
            )

        payload["text"] = chunk["text"]
        contexts.append(payload)

    return contexts


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        [
            f"[Fonte {i + 1}] {ctx['title']}\nURL: {ctx['url']}\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ]
    )

    return f"""
Você é um assistente especialista em Hearts of Iron IV.

Responda em português brasileiro usando apenas o contexto fornecido.
Se a resposta não estiver no contexto, diga que não encontrou informação suficiente.
Cite as fontes pelo título da página.

Contexto:
{context_text}

Pergunta:
{question}

Resposta:
""".strip()


def generate_answer(model_path: str, prompt: str, max_new_tokens: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--collection-name", default=COLLECTION_NAME)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH)
    args = parser.parse_args()

    embedder = SentenceTransformer(args.embedding_model)
    chunk_lookup = load_chunk_lookup(args.chunks_path)
    contexts = retrieve(
        args.question,
        embedder=embedder,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        chunk_lookup=chunk_lookup,
        chunks_path=args.chunks_path,
        top_k=args.top_k,
    )
    prompt = build_prompt(args.question, contexts)
    answer = generate_answer(args.model, prompt, args.max_new_tokens)

    print("\n--- RESPOSTA ---\n")
    print(answer)

    print("\n--- FONTES RECUPERADAS ---\n")
    for ctx in contexts:
        print(f"- {ctx['title']}: {ctx['url']}")


if __name__ == "__main__":
    main()
