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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PIPELINE_CONFIG_PATH, get_config_section, require_config_value

HEADERS = {
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
    retries: int,
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
                if "Client Challenge" in response.text:
                    raise WikiApiError(
                        "A HOI4 Wiki retornou uma pagina de Client Challenge em vez "
                        "de JSON. O endpoint MediaWiki esta bloqueando requisicoes "
                        "automatizadas neste momento. Tente novamente mais tarde ou "
                        "informe outro endpoint com --api-url/HOI4_WIKI_API_URL."
                    )
                raise WikiApiError(
                    "Resposta nao JSON da MediaWiki API "
                    f"(status={response.status_code}, content-type={content_type!r}, "
                    f"trecho={snippet!r})."
                )
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
    retries: int,
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

        data = api_get(session, api_url, params, retries)
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
    retries: int,
) -> dict[str, Any] | None:
    data = api_get(
        session,
        api_url,
        {
            "action": "parse",
            "page": title,
            "prop": "text|revid|displaytitle",
            "redirects": 1,
        },
        retries,
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
    retries: int,
) -> tuple[str, dict[str, Any] | None, str | None]:
    if delay > 0:
        time.sleep(delay)

    session = thread_session(user_agent)
    try:
        return title, get_page_content(session, api_url, title, retries), None
    except Exception as exc:
        return title, None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PIPELINE_CONFIG_PATH)
    args = parser.parse_args()
    config = get_config_section("collect", args.config)

    api_url = os.environ.get("HOI4_WIKI_API_URL", require_config_value(config, "api_url", "collect"))
    user_agent = os.environ.get("HOI4_WIKI_USER_AGENT", require_config_value(config, "user_agent", "collect"))
    output_path = Path(require_config_value(config, "output", "collect"))
    workers = max(1, int(require_config_value(config, "workers", "collect")))
    delay = float(require_config_value(config, "delay", "collect"))
    list_delay = float(require_config_value(config, "list_delay", "collect"))
    limit = int(require_config_value(config, "limit", "collect"))
    resume = bool(require_config_value(config, "resume", "collect"))
    retries = int(require_config_value(config, "retries", "collect"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = build_session(user_agent)
    pages = get_all_pages(session, api_url, list_delay, retries)
    print(f"Total de páginas encontradas: {len(pages)}")

    existing_titles = read_existing_titles(output_path) if resume else set()
    if existing_titles:
        print(f"Paginas ja coletadas: {len(existing_titles)}")

    titles = [
        page["title"]
        for page in pages
        if isinstance(page.get("title"), str) and page["title"] not in existing_titles
    ]
    if limit > 0:
        titles = titles[:limit]

    mode = "a" if resume else "w"
    written = 0
    failed = 0

    with output_path.open(mode, encoding="utf-8") as file:
        if workers == 1:
            iterator = tqdm(titles, desc="Coletando paginas")
            for title in iterator:
                _, item, error = collect_page(api_url, user_agent, title, delay, retries)
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
                        api_url,
                        user_agent,
                        title,
                        delay,
                        retries,
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
