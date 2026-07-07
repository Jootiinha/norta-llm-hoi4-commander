import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm

import random


DEFAULT_API_URL = "https://hoi4.paradoxwikis.com/api.php"
OUTPUT_PATH = Path("data/raw/hoi4_wiki/hoi4_pages.jsonl")
DEFAULT_USER_AGENT = "joao-hoi4-study-bot/0.1 (contato: joaocrm@outlook.com)"

HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
}


class WikiApiError(RuntimeError):
    """Raised when the MediaWiki API does not return usable JSON."""


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
            time.sleep(random.randint(0,1) * attempt)

    raise WikiApiError(f"Falha ao consultar MediaWiki API: {last_error}") from last_error


def get_all_pages(session: requests.Session, api_url: str) -> list[dict[str, Any]]:
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
        time.sleep(random.randint(0, 0.5))

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
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)
    session.headers["User-Agent"] = args.user_agent

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages = get_all_pages(session, args.api_url)
    print(f"Total de páginas encontradas: {len(pages)}")

    with output_path.open("w", encoding="utf-8") as f:
        for page in tqdm(pages):
            title = page["title"]

            try:
                item = get_page_content(session, args.api_url, title)
                if item:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as exc:
                print(f"Erro ao processar {title}: {exc}")

            time.sleep(0.7)

    print(f"Arquivo salvo em: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except WikiApiError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
