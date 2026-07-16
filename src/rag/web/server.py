import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.config import PIPELINE_CONFIG_PATH, get_config_section, require_config_value

from .options import (
    ask_defaults,
    default_qdrant_url,
    first_query_value,
    get_available_options,
    web_generation_defaults,
)
from .state import QueryOptions, RagWebState


HTML_PATH = Path(__file__).resolve().parents[1] / "web_hoi4_rag.html"


def load_index_html() -> bytes:
    return HTML_PATH.read_bytes()


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

