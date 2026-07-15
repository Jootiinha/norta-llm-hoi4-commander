import argparse
import json
from pathlib import Path
import sys

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


COLLECTION_NAME = "hoi4_wiki"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_QDRANT_URL = "http://localhost:6333"
CHUNKS_PATH = Path("data/processed/chunks/chunks.jsonl")
DEFAULT_CANDIDATE_MULTIPLIER = 4


def exit_with_message(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def needs_e5_prefix(model_name: str) -> bool:
    return "e5" in model_name.lower()


def format_query(query: str, model_name: str) -> str:
    if not needs_e5_prefix(model_name):
        return query
    return f"query: {query}"


def load_chunk_lookup(path: Path) -> dict[str, dict]:
    if not path.exists():
        exit_with_message(
            f"Arquivo de chunks nao encontrado: {path}\n"
            "Gere os chunks antes da consulta com:\n"
            "  make chunk-data"
        )

    lookup: dict[str, dict] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id:
                lookup[chunk_id] = chunk

    if not lookup:
        exit_with_message(
            f"Nenhum chunk encontrado em {path}.\n"
            "Gere novamente os chunks antes da consulta com:\n"
            "  make chunk-data"
        )

    return lookup


def retrieve(
    query: str,
    embedder: SentenceTransformer,
    qdrant_url: str,
    collection_name: str,
    chunk_lookup: dict[str, dict],
    chunks_path: Path,
    embedding_model_name: str,
    top_k: int = 5,
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
) -> list[dict]:
    client = QdrantClient(url=qdrant_url)
    if not client.collection_exists(collection_name):
        exit_with_message(
            f"Colecao Qdrant '{collection_name}' nao encontrada em {qdrant_url}.\n"
            "Indexe os chunks antes da consulta com:\n"
            "  make index-data\n"
            "Se voce usou outro nome de colecao ao indexar, informe o mesmo nome com "
            "--collection-name."
        )

    vector = embedder.encode(
        format_query(query, embedding_model_name),
        normalize_embeddings=True,
    ).tolist()

    candidate_limit = max(top_k, top_k * candidate_multiplier)
    results = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=candidate_limit,
    ).points

    contexts = []
    seen_sources: set[tuple[str, str]] = set()

    for hit in results:
        payload = dict(hit.payload or {})
        chunk_id = payload.get("chunk_id")
        chunk = chunk_lookup.get(chunk_id) if chunk_id else None

        if chunk is None:
            raise KeyError(
                f"Chunk '{chunk_id}' nao encontrado em {chunks_path}."
            )

        source_key = (
            str(payload.get("url") or ""),
            str(payload.get("title") or ""),
        )
        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        payload["score"] = hit.score
        payload["text"] = chunk["text"]
        payload["section"] = chunk.get("section")
        payload["subsection"] = chunk.get("subsection")
        contexts.append(payload)
        if len(contexts) >= top_k:
            break

    return contexts


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        [
            (
                f"[Fonte {i + 1}] {ctx['title']}\n"
                f"URL: {ctx['url']}\n"
                f"Secao: {ctx.get('section') or '-'}\n"
                f"Subsecao: {ctx.get('subsection') or '-'}\n"
                f"Trecho:\n{ctx['text']}"
            )
            for i, ctx in enumerate(contexts)
        ]
    )

    return f"""
Você e um assistente especialista em Hearts of Iron IV.

Regras obrigatorias:
- Responda em portugues brasileiro usando apenas o contexto fornecido.
- Nao invente nomes de focos, bonus, predicados ou efeitos.
- Se a resposta nao estiver clara no contexto, diga explicitamente que nao encontrou informacao suficiente.
- Ao mencionar um foco, use o nome exato que aparece no contexto.
- Nao repita o enunciado, nao escreva "Fonte 1", "Answer:" ou "Resposta final:".
- Responda no formato exato abaixo.

Contexto:
{context_text}

Pergunta:
{question}

Resposta:
<resposta curta em 2 a 6 frases>

Fontes:
- <titulo da pagina 1>
- <titulo da pagina 2>
""".strip()


def generate_answer(model_path: str, prompt: str, max_new_tokens: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
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
    parser.add_argument("--candidate-multiplier", type=int, default=DEFAULT_CANDIDATE_MULTIPLIER)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--collection-name", default=COLLECTION_NAME)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--show-context", action="store_true")
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
        embedding_model_name=args.embedding_model,
        top_k=args.top_k,
        candidate_multiplier=args.candidate_multiplier,
    )
    prompt = build_prompt(args.question, contexts)
    answer = generate_answer(args.model, prompt, args.max_new_tokens)

    if args.show_context:
        print("\n--- CONTEXTO RECUPERADO ---\n")
        for i, ctx in enumerate(contexts, start=1):
            print(f"[Fonte {i}] {ctx['title']} | score={ctx.get('score', 0):.4f}")
            print(f"URL: {ctx['url']}")
            print(f"Secao: {ctx.get('section') or '-'} | Subsecao: {ctx.get('subsection') or '-'}")
            print(ctx["text"])
            print()

    print("\n--- RESPOSTA ---\n")
    print(answer)

    print("\n--- FONTES RECUPERADAS ---\n")
    for ctx in contexts:
        print(f"- {ctx['title']}: {ctx['url']}")


if __name__ == "__main__":
    main()
