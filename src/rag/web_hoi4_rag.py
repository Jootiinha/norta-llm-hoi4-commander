import argparse
import gc
from dataclasses import dataclass
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from urllib.parse import parse_qs, urlparse

from qdrant_client import QdrantClient
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import PIPELINE_CONFIG_PATH, get_config_section, require_config_value
from rag.questions import (
    build_prompt,
    clean_generated_answer,
    load_chunk_lookup,
    retrieve,
)


HTML_PATH = Path(__file__).with_name("web_hoi4_rag.html")


def ask_defaults() -> dict:
    return get_config_section("ask")


def web_defaults() -> dict:
    return get_config_section("web")


def web_generation_defaults() -> dict:
    config = require_config_value(web_defaults(), "generation", "web")
    if not isinstance(config, dict):
        raise SystemExit("Configure 'web.generation' como mapa em configs/pipeline.yaml.")
    return config


def default_qdrant_url() -> str:
    return str(require_config_value(ask_defaults(), "qdrant_url", "ask"))


def load_index_html() -> bytes:
    return HTML_PATH.read_bytes()


def first_query_value(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    if not values:
        return default

    return values[0] or default


def is_local_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ]
    return any((path / filename).exists() for filename in tokenizer_files)


def list_local_models(models_dir: Path = Path("models")) -> list[str]:
    if not models_dir.exists():
        return []

    return [
        str(path)
        for path in sorted(models_dir.iterdir(), key=lambda item: item.name.lower())
        if is_local_model_dir(path)
    ]


def list_chunk_files(chunks_dir: Path) -> list[str]:
    if not chunks_dir.exists():
        return []

    return [
        str(path)
        for path in sorted(chunks_dir.rglob("*.jsonl"), key=lambda item: str(item).lower())
        if path.is_file() and path.name == "chunks.jsonl"
    ]


def list_qdrant_collections(qdrant_url: str) -> tuple[list[str], str | None]:
    try:
        client = QdrantClient(url=qdrant_url, timeout=2.0)
        response = client.get_collections()
    except Exception as exc:
        return [], str(exc)

    return sorted(collection.name for collection in response.collections), None


def get_available_options(qdrant_url: str) -> dict:
    defaults = ask_defaults()
    web_config = web_defaults()
    generation = web_generation_defaults()
    collections, collections_error = list_qdrant_collections(qdrant_url)
    models = list_local_models()
    chunk_files = list_chunk_files(Path(require_config_value(web_config, "chunks_dir", "web")))

    return {
        "defaults": {
            "model_path": str(require_config_value(defaults, "model", "ask")),
            "embedding_model": str(require_config_value(defaults, "embedding_model", "ask")),
            "chunks_path": str(require_config_value(defaults, "chunks_path", "ask")),
            "collection_name": str(require_config_value(defaults, "collection_name", "ask")),
            "qdrant_url": str(require_config_value(defaults, "qdrant_url", "ask")),
            "top_k": int(require_config_value(defaults, "top_k", "ask")),
            "candidate_multiplier": int(require_config_value(defaults, "candidate_multiplier", "ask")),
            "max_new_tokens": int(require_config_value(defaults, "max_new_tokens", "ask")),
            "do_sample": bool(require_config_value(generation, "do_sample", "web.generation")),
            "enable_thinking": bool(require_config_value(defaults, "enable_thinking", "ask")),
            "temperature": float(require_config_value(generation, "temperature", "web.generation")),
            "top_p": float(require_config_value(generation, "top_p", "web.generation")),
            "repetition_penalty": float(require_config_value(generation, "repetition_penalty", "web.generation")),
            "no_repeat_ngram_size": int(require_config_value(generation, "no_repeat_ngram_size", "web.generation")),
            "show_context": False,
        },
        "models": models,
        "embedding_models": list(require_config_value(web_config, "embedding_models", "web")),
        "chunk_files": chunk_files,
        "collections": collections,
        "collections_error": collections_error,
    }


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


class RagRequestHandler(BaseHTTPRequestHandler):
    state: RagWebState

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self._send_headers("application/json; charset=utf-8", 0, HTTPStatus.NOT_FOUND)
            return

        self._send_headers("text/html; charset=utf-8", len(load_index_html()))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(load_index_html(), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/options":
            query = parse_qs(parsed.query)
            qdrant_url = first_query_value(query, "qdrant_url", default_qdrant_url())
            self._send_json(get_available_options(qdrant_url))
            return

        self._send_json({"error": "Rota nao encontrada."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/ask":
            self._send_json({"error": "Rota nao encontrada."}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json()
            question = str(payload.get("question") or "").strip()
            if not question:
                raise ValueError("Informe uma pergunta.")

            defaults = ask_defaults()
            generation = web_generation_defaults()
            options = QueryOptions(
                model_path=str(payload.get("model_path") or require_config_value(defaults, "model", "ask")),
                embedding_model=str(payload.get("embedding_model") or require_config_value(defaults, "embedding_model", "ask")),
                chunks_path=Path(str(payload.get("chunks_path") or require_config_value(defaults, "chunks_path", "ask"))),
                collection_name=str(payload.get("collection_name") or require_config_value(defaults, "collection_name", "ask")),
                qdrant_url=str(payload.get("qdrant_url") or require_config_value(defaults, "qdrant_url", "ask")),
                top_k=max(1, min(20, int(payload.get("top_k") or require_config_value(defaults, "top_k", "ask")))),
                candidate_multiplier=max(
                    1,
                    min(20, int(payload.get("candidate_multiplier") or require_config_value(defaults, "candidate_multiplier", "ask"))),
                ),
                max_new_tokens=max(32, min(2048, int(payload.get("max_new_tokens") or require_config_value(defaults, "max_new_tokens", "ask")))),
                show_context=bool(payload.get("show_context")),
                do_sample=bool(payload.get("do_sample") if "do_sample" in payload else require_config_value(generation, "do_sample", "web.generation")),
                enable_thinking=bool(payload.get("enable_thinking") or require_config_value(defaults, "enable_thinking", "ask")),
                temperature=max(0.1, min(2.0, float(payload.get("temperature") or require_config_value(generation, "temperature", "web.generation")))),
                top_p=max(0.05, min(1.0, float(payload.get("top_p") or require_config_value(generation, "top_p", "web.generation")))),
                repetition_penalty=max(
                    1.0,
                    min(2.0, float(payload.get("repetition_penalty") or require_config_value(generation, "repetition_penalty", "web.generation"))),
                ),
                no_repeat_ngram_size=max(
                    0,
                    min(20, int(payload.get("no_repeat_ngram_size") or require_config_value(generation, "no_repeat_ngram_size", "web.generation"))),
                ),
            )
            result = self.state.answer(question, options)
            self._send_json(result)
        except SystemExit as exc:
            self._send_json({"error": str(exc) or "Consulta interrompida."}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self._send_headers(content_type, len(body), status)
        self.wfile.write(body)

    def _send_headers(
        self,
        content_type: str,
        content_length: int,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface web local para perguntar ao RAG de HOI4.")
    parser.add_argument("--config", type=Path, default=PIPELINE_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config_section("web", args.config)
    host = str(require_config_value(config, "host", "web"))
    port = int(require_config_value(config, "port", "web"))
    RagRequestHandler.state = RagWebState()
    try:
        server = ThreadingHTTPServer((host, port), RagRequestHandler)
    except OSError as exc:
        if exc.errno == 98:
            raise SystemExit(
                f"A porta {port} ja esta em uso em {host}.\n"
                "Encerre o servidor anterior ou use outra porta com:\n"
                "  edite web.port em configs/pipeline.yaml"
            ) from exc
        raise

    print(f"Interface web disponivel em http://{host}:{port}")
    print("Pressione Ctrl+C para encerrar.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor web.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
