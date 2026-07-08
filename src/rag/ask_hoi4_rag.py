import argparse

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


COLLECTION_NAME = "hoi4_wiki"


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    embedder = SentenceTransformer("BAAI/bge-m3")
    client = QdrantClient(url="http://localhost:6333")

    vector = embedder.encode(query, normalize_embeddings=True).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
    )

    return [hit.payload for hit in results]


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
    args = parser.parse_args()

    contexts = retrieve(args.question, top_k=args.top_k)
    prompt = build_prompt(args.question, contexts)
    answer = generate_answer(args.model, prompt, args.max_new_tokens)

    print("\n--- RESPOSTA ---\n")
    print(answer)

    print("\n--- FONTES RECUPERADAS ---\n")
    for ctx in contexts:
        print(f"- {ctx['title']}: {ctx['url']}")


if __name__ == "__main__":
    main()
