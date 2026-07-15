import argparse
from dataclasses import dataclass
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import urlparse

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from ask_hoi4_rag import (
    CHUNKS_PATH,
    COLLECTION_NAME,
    DEFAULT_CANDIDATE_MULTIPLIER,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QDRANT_URL,
    build_prompt,
    load_chunk_lookup,
    retrieve,
)


DEFAULT_MODEL_PATH = "models/qwen3-0.6b"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


HTML_PATH = Path(__file__).with_name("web_hoi4_rag.html")


def load_index_html() -> bytes:
    return HTML_PATH.read_bytes()


@dataclass(frozen=True)
class QueryOptions:
    model_path: str
    embedding_model: str
    chunks_path: Path
    collection_name: str
    qdrant_url: str
    top_k: int
    max_new_tokens: int


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
                candidate_multiplier=DEFAULT_CANDIDATE_MULTIPLIER,
            )
            prompt = build_prompt(question, contexts)
            answer = self._generate(options.model_path, prompt, options.max_new_tokens)

        return {
            "answer": answer,
            "contexts": [
                {
                    "title": ctx.get("title"),
                    "url": ctx.get("url"),
                    "section": ctx.get("section"),
                    "subsection": ctx.get("subsection"),
                    "score": ctx.get("score"),
                    "text": ctx.get("text"),
                }
                for ctx in contexts
            ],
        }

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

    def _generate(self, model_path: str, prompt: str, max_new_tokens: int) -> str:
        if self._model is None or self._model_path != model_path:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self._model_path = model_path

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        return self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()


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
        if parsed.path != "/":
            self._send_json({"error": "Rota nao encontrada."}, HTTPStatus.NOT_FOUND)
            return

        self._send_bytes(load_index_html(), "text/html; charset=utf-8")

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

            options = QueryOptions(
                model_path=str(payload.get("model_path") or DEFAULT_MODEL_PATH),
                embedding_model=str(payload.get("embedding_model") or DEFAULT_EMBEDDING_MODEL),
                chunks_path=Path(str(payload.get("chunks_path") or CHUNKS_PATH)),
                collection_name=str(payload.get("collection_name") or COLLECTION_NAME),
                qdrant_url=str(payload.get("qdrant_url") or DEFAULT_QDRANT_URL),
                top_k=max(1, min(20, int(payload.get("top_k") or 5))),
                max_new_tokens=max(32, min(2048, int(payload.get("max_new_tokens") or 350))),
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
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RagRequestHandler.state = RagWebState()
    try:
        server = ThreadingHTTPServer((args.host, args.port), RagRequestHandler)
    except OSError as exc:
        if exc.errno == 98:
            raise SystemExit(
                f"A porta {args.port} ja esta em uso em {args.host}.\n"
                "Encerre o servidor anterior ou use outra porta com:\n"
                "  make web-rag WEB_PORT=7861"
            ) from exc
        raise

    print(f"Interface web disponivel em http://{args.host}:{args.port}")
    print("Pressione Ctrl+C para encerrar.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor web.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
