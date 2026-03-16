"""Download or copy attachments referenced by the sendit markdown index."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import urlopen
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_env import get_env, load_repo_env


load_repo_env(__file__)


try:
    from loguru import logger
except ImportError as error:
    raise SystemExit(
        "Brak zaleznosci 'loguru'. Zainstaluj ja poleceniem: pip install loguru"
    ) from error


MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
AUTOLINK_RE = re.compile(r"<(https?://[^>]+)>")
BARE_URL_RE = re.compile(r"(?<!\()(?P<url>https?://[^\s)>]+)")
INCLUDE_FILE_RE = re.compile(r"""\[include\s+file=["']([^"']+)["']\]""")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Odczytuje index.md, wyodrebnia linki do zalacznikow i zapisuje "
            "je w katalogu attachments."
        )
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("index.md"),
        help="Sciezka do pliku markdown z linkami. Domyslnie: index.md",
    )
    parser.add_argument(
        "--attachments-dir",
        type=Path,
        default=Path("attachments"),
        help="Katalog docelowy dla zalacznikow. Domyslnie: attachments",
    )
    parser.add_argument(
        "--source-url",
        default=get_env("SENDIT_SOURCE_INDEX_URL"),
        help=(
            "Zrodlowy URL index.md, uzywany do rozwiazywania wzglednych "
            "sciezek zalacznikow. Domyslnie: SENDIT_SOURCE_INDEX_URL z .env"
        ),
    )
    return parser.parse_args()


def read_markdown(index_path: Path) -> str:
    try:
        return index_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return index_path.read_text(encoding="utf-8-sig")


def strip_optional_title(target: str) -> str:
    cleaned = target.strip()
    if cleaned.startswith("<") and cleaned.endswith(">"):
        return cleaned[1:-1].strip()

    if " " in cleaned:
        possible_url, *_ = cleaned.split(" ", 1)
        if "://" in possible_url or possible_url.endswith((".md", ".pdf", ".doc", ".docx")):
            return possible_url

    return cleaned


def extract_attachment_targets(markdown: str) -> list[str]:
    targets: list[str] = []

    for raw_target in MARKDOWN_LINK_RE.findall(markdown):
        target = strip_optional_title(raw_target)
        if target and not target.startswith("#"):
            targets.append(target)

    targets.extend(AUTOLINK_RE.findall(markdown))
    targets.extend(match.group("url") for match in BARE_URL_RE.finditer(markdown))
    targets.extend(INCLUDE_FILE_RE.findall(markdown))

    unique_targets: list[str] = []
    seen: set[str] = set()
    for target in targets:
        normalized = target.strip()
        if not normalized or normalized.startswith("#"):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_targets.append(normalized)

    return unique_targets


def filename_from_target(target: str) -> str:
    parsed = urlparse(target)
    candidate = unquote(Path(parsed.path).name if parsed.scheme else Path(target).name)
    return candidate or "downloaded_attachment"


def download_remote_file(url: str, destination: Path) -> None:
    with urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def copy_local_file(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)


def materialize_attachment(
    target: str,
    index_dir: Path,
    attachments_dir: Path,
    source_url: str,
) -> str:
    destination = attachments_dir / filename_from_target(target)

    if destination.exists():
        return f"SKIP  {target} -> {destination.name} (plik juz istnieje)"

    parsed = urlparse(target)
    if parsed.scheme in {"http", "https"}:
        try:
            download_remote_file(target, destination)
        except (HTTPError, URLError) as error:
            return f"ERROR {target} ({error})"
        return f"OK    {target} -> {destination.name}"

    source = (index_dir / target).resolve()
    if source.exists():
        copy_local_file(source, destination)
        return f"OK    {target} -> {destination.name}"

    remote_url = urljoin(source_url, target)
    try:
        download_remote_file(remote_url, destination)
    except (HTTPError, URLError) as error:
        return (
            f"MISS  {target} (brak lokalnie; nie udalo sie pobrac z "
            f"{remote_url}: {error})"
        )

    return f"OK    {target} -> {destination.name} (pobrano z {remote_url})"


def main() -> int:
    args = parse_args()
    index_path = args.index.resolve()
    if not args.source_url:
        logger.error("Brak SENDIT_SOURCE_INDEX_URL w .env albo --source-url.")
        return 1

    if not index_path.exists():
        logger.error("Brak pliku: {}", index_path)
        return 1

    attachments_dir = args.attachments_dir
    if not attachments_dir.is_absolute():
        attachments_dir = index_path.parent / attachments_dir
    attachments_dir.mkdir(parents=True, exist_ok=True)

    markdown = read_markdown(index_path)
    targets = extract_attachment_targets(markdown)

    if not targets:
        logger.info("Nie znaleziono zadnych linkow do zalacznikow.")
        return 0

    logger.info("Znaleziono {} unikalnych zalacznikow.", len(targets))
    for target in targets:
        result = materialize_attachment(
            target=target,
            index_dir=index_path.parent,
            attachments_dir=attachments_dir,
            source_url=args.source_url,
        )
        if result.startswith("OK"):
            logger.success(result)
        elif result.startswith("SKIP"):
            logger.info(result)
        elif result.startswith("MISS"):
            logger.warning(result)
        else:
            logger.error(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
