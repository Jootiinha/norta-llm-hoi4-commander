import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm


DEFAULT_API_URL = "https://hoi4.paradoxwikis.com/api.php"
OUTPUT_PATH = Path("data/raw/hoi4_wiki/hoi4_pages.jsonl")
DEFAULT_USER_AGENT = "joao-hoi4-study-bot/0.1 (contato: joaocrm@outlook.com)"

HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
}

_THREAD_LOCAL = threading.local()


class WikiApiError(RuntimeError):
    """Raised when the MediaWiki API does not return usable JSON."""


def build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.headers["User-Agent"] = user_agent
    return session


def thread_session(user_agent: str) -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = build_session(user_agent)
        _THREAD_LOCAL.session = session
    return session


def api_get(
    session: requests.Session,
    api_url: str,
    params: dict[str, Any],
    retries: int = 3,
) -> dict[str, Any]:
    base_params = {
        "format": "json",
        "formatversion": "2",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                api_url,
                params={**base_params, **params},
                timeout=30,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type.lower():
                snippet = " ".join(response.text[:300].split())
            return response.json()
        except requests.JSONDecodeError as exc:
            snippet = " ".join(response.text[:300].split())
            raise WikiApiError(
                f"Falha ao decodificar JSON da MediaWiki API. Trecho: {snippet!r}"
            ) from exc
        except WikiApiError:
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep((1.5 * attempt) + random.uniform(0.0, 0.5))

    raise WikiApiError(f"Falha ao consultar MediaWiki API: {last_error}") from last_error


def get_all_pages(
    session: requests.Session,
    api_url: str,
    list_delay: float,
) -> list[dict[str, Any]]:
    pages = []
    apcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": 0,
            "aplimit": "max",
        }

        if apcontinue:
            params["apcontinue"] = apcontinue

        data = api_get(session, api_url, params)
        pages.extend(data["query"]["allpages"])

        if "continue" not in data:
            break

        apcontinue = data["continue"]["apcontinue"]
        if list_delay > 0:
            time.sleep(list_delay)

    return pages


def clean_html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    content = soup.select_one(".mw-parser-output") or soup

    for selector in [
        ".toc",
        ".mw-editsection",
        ".navbox",
        ".metadata",
        ".ambox",
        ".hatnote",
        "script",
        "style",
        "noscript",
    ]:
        for tag in content.select(selector):
            tag.decompose()

    markdown = md(str(content), heading_style="ATX")

    lines = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[edit]"):
            continue
        lines.append(line)

    return "\n".join(lines)


def get_page_content(
    session: requests.Session,
    api_url: str,
    title: str,
) -> dict[str, Any] | None:
    data = api_get(
        session,
        api_url,
        {
            "action": "parse",
            "page": title,
            "prop": "text|revid|displaytitle",
            "redirects": 1,
        }
    )

    if "parse" not in data:
        return None

    parsed = data["parse"]
    html = parsed["text"]
    text = clean_html_to_markdown(html)

    if len(text) < 200:
        return None

    return {
        "page_id": parsed.get("pageid"),
        "revision_id": parsed.get("revid"),
        "title": parsed.get("title", title),
        "display_title": parsed.get("displaytitle", title),
        "url": f"https://hoi4.paradoxwikis.com/{title.replace(' ', '_')}",
        "text": text,
    }


def read_existing_titles(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    titles = set()
    with output_path.open(encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = record.get("title")
            if isinstance(title, str):
                titles.add(title)
    return titles


def collect_page(
    api_url: str,
    user_agent: str,
    title: str,
    delay: float,
) -> tuple[str, dict[str, Any] | None, str | None]:
    if delay > 0:
        time.sleep(delay)

    session = thread_session(user_agent)
    try:
        return title, get_page_content(session, api_url, title), None
    except Exception as exc:
        return title, None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url",
        default=os.environ.get("HOI4_WIKI_API_URL", DEFAULT_API_URL),
        help="Endpoint MediaWiki API da HOI4 Wiki.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Arquivo JSONL de saida.",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("HOI4_WIKI_USER_AGENT", DEFAULT_USER_AGENT),
        help="Header User-Agent enviado para a MediaWiki API.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Numero de paginas coletadas em paralelo. Use 1 para modo conservador.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.7,
        help="Pausa por pagina antes de chamar action=parse.",
    )
    parser.add_argument(
        "--list-delay",
        type=float,
        default=0.2,
        help="Pausa entre chamadas de paginacao da lista allpages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limita a quantidade de paginas processadas. 0 processa todas.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pula paginas ja presentes no JSONL de saida.",
    )
    args = parser.parse_args()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = build_session(args.user_agent)
    pages = get_all_pages(session, args.api_url, args.list_delay)
    print(f"Total de páginas encontradas: {len(pages)}")

    existing_titles = read_existing_titles(output_path) if args.resume else set()
    if existing_titles:
        print(f"Paginas ja coletadas: {len(existing_titles)}")

    titles = [
        page["title"]
        for page in pages
        if isinstance(page.get("title"), str) and page["title"] not in existing_titles
    ]
    if args.limit > 0:
        titles = titles[: args.limit]

    workers = max(1, args.workers)
    mode = "a" if args.resume else "w"
    written = 0
    failed = 0

    with output_path.open(mode, encoding="utf-8") as file:
        if workers == 1:
            iterator = tqdm(titles, desc="Coletando paginas")
            for title in iterator:
                _, item, error = collect_page(args.api_url, args.user_agent, title, args.delay)
                if error:
                    failed += 1
                    print(f"Erro ao processar {title}: {error}")
                    continue
                if item:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    file.flush()
                    written += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        collect_page,
                        args.api_url,
                        args.user_agent,
                        title,
                        args.delay,
                    )
                    for title in titles
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Coletando paginas",
                ):
                    title, item, error = future.result()
                    if error:
                        failed += 1
                        print(f"Erro ao processar {title}: {error}")
                        continue
                    if item:
                        file.write(json.dumps(item, ensure_ascii=False) + "\n")
                        file.flush()
                        written += 1

    print(f"Arquivo salvo em: {output_path}")
    print(f"Paginas novas salvas: {written}")
    print(f"Falhas: {failed}")


if __name__ == "__main__":
    try:
        main()
    except WikiApiError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
