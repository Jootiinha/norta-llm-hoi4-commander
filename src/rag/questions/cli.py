import argparse
from pathlib import Path

from src.config import PIPELINE_CONFIG_PATH, get_config_section, require_config_value

from .config import RagConfig
from .pipeline import Hoi4RagPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PIPELINE_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ask_config = get_config_section("ask", args.config)

    config = RagConfig(
        model_path=str(require_config_value(ask_config, "model", "ask")),
        embedding_model=str(require_config_value(ask_config, "embedding_model", "ask")),
        qdrant_url=str(require_config_value(ask_config, "qdrant_url", "ask")),
        collection_name=str(require_config_value(ask_config, "collection_name", "ask")),
        chunks_path=Path(require_config_value(ask_config, "chunks_path", "ask")),
        top_k=int(require_config_value(ask_config, "top_k", "ask")),
        candidate_multiplier=int(require_config_value(ask_config, "candidate_multiplier", "ask")),
        max_new_tokens=int(require_config_value(ask_config, "max_new_tokens", "ask")),
        enable_thinking=bool(require_config_value(ask_config, "enable_thinking", "ask")),
    )
    question = str(require_config_value(ask_config, "question", "ask")).strip()
    if not question:
        raise SystemExit("Configure uma pergunta em ask.question no YAML.")

    result = Hoi4RagPipeline(config).ask(question)
    contexts = result["contexts"]
    answer = result["answer"]

    show_context = bool(require_config_value(ask_config, "show_context", "ask"))
    if show_context:
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
