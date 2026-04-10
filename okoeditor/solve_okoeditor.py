"""Solve the AG3NTS okoeditor task with an OpenRouter tool-calling agent."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib import request

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.flags import extract_flag
from devs_utilities.http import HttpRequestError
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    build_task_site_name,
    OpenRouterClient,
    OpenRouterError,
    parse_json_content,
    ToolCall,
)
from devs_utilities.repo_env import get_env, get_int_env, get_llm_model, get_optional_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="okoeditor")

TASK_NAME = "okoeditor"
OKO_BASE_URL = "https://example.invalid"
DEFAULT_MODEL = get_llm_model("OKOEDITOR_MODEL")
DEFAULT_MAX_STEPS = get_int_env("OKOEDITOR_MAX_STEPS", 12)
DEFAULT_API_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30)
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 60)
STEP_RETRY_ATTEMPTS = get_int_env("OKOEDITOR_STEP_RETRY_ATTEMPTS", 3) or 3
STEP_RETRY_BASE_DELAY_SECONDS = float(
    get_int_env("OKOEDITOR_STEP_RETRY_BASE_DELAY_SECONDS", 4) or 4
)
OUTPUT_DIR = Path(__file__).resolve().parent
LAST_TRANSCRIPT_PATH = OUTPUT_DIR / "last_transcript.json"
LAST_DONE_RESPONSE_PATH = OUTPUT_DIR / "last_done_response.json"
LAST_UPDATE_RESPONSE_PATH = OUTPUT_DIR / "last_update_response.json"
SYSTEM_PROMPT_PATH = OUTPUT_DIR / "system_prompt.txt"

INITIAL_USER_PROMPT = """Solve okoeditor efficiently.
Use find_targets or evaluate_state first, perform only the required edits, then submit_done."""


def load_system_prompt() -> str:
    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError(f"System prompt file is empty: {SYSTEM_PROMPT_PATH}")
    return prompt


SYSTEM_PROMPT = load_system_prompt()


class OkoEditorError(RuntimeError):
    """Raised when the OKO session, parser, or agent returns invalid data."""


def is_retryable_openrouter_error(error: Exception) -> bool:
    """Return True for transient upstream/provider failures."""

    lowered = str(error).casefold()
    return any(
        marker in lowered
        for marker in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "timed out",
            "internal server error",
            "temporarily unavailable",
            "operation was aborted",
        )
    )


@dataclass(slots=True)
class AppConfig:
    ag3nts_api_key: str
    verify_url: str
    oko_login: str
    oko_password: str
    openrouter_api_key: str
    openrouter_url: str
    model: str
    site_url: str | None
    site_name: str | None
    max_steps: int
    api_timeout_seconds: int
    show_tool_results: bool


@dataclass(slots=True)
class AgentState:
    final_flag: str | None = None
    final_response: Any = None
    last_update_response: Any = None
    targets: dict[str, Any] | None = None
    decoy_incident_id: str | None = None
    completed: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help=f"OpenRouter model. Default: {DEFAULT_MODEL}.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum tool-calling rounds. Default: {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--show-tool-results",
        action="store_true",
        help="Print every tool result during the run.",
    )
    parser.add_argument(
        "--transcript-path",
        default=str(LAST_TRANSCRIPT_PATH),
        help="Where to save the final transcript.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    ag3nts_api_key = (get_env("AG3NTS_API_KEY") or "").strip()
    openrouter_api_key = (get_env("OPENROUTER_API_KEY") or "").strip()
    openrouter_url = (get_env("OPENROUTER_BASE_URL") or "").strip()
    model = (args.model or DEFAULT_MODEL).strip()
    site_url = get_optional_env("OPENROUTER_SITE_URL") or get_optional_env("OPENROUTER_APP_URL")
    site_name = build_task_site_name(__file__, task_name=TASK_NAME)

    missing: list[str] = []
    if not ag3nts_api_key:
        missing.append("AG3NTS_API_KEY")
    if not openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")
    if not openrouter_url:
        missing.append("OPENROUTER_BASE_URL")
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be a positive integer.")

    return AppConfig(
        ag3nts_api_key=ag3nts_api_key,
        verify_url=AG3NTS_VERIFY_URL,
        oko_login="Zofia",
        oko_password="Zofia2026!",
        openrouter_api_key=openrouter_api_key,
        openrouter_url=openrouter_url,
        model=model,
        site_url=site_url,
        site_name=site_name,
        max_steps=args.max_steps,
        api_timeout_seconds=DEFAULT_API_TIMEOUT_SECONDS,
        show_tool_results=args.show_tool_results,
    )


class OkoWebClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": f"{OKO_BASE_URL}/",
        }
        self.reset_session()

    def reset_session(self) -> None:
        self._cookies = CookieJar()
        self._opener = request.build_opener(request.HTTPCookieProcessor(self._cookies))
        self._logged_in = False

    def _open(self, path: str, *, data: bytes | None = None) -> str:
        url = f"{OKO_BASE_URL}{path}"
        http_request = request.Request(url, data=data, headers=self._headers)
        try:
            with self._opener.open(http_request, timeout=self._config.api_timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            raise OkoEditorError(f"Failed to fetch {url}: {exc}") from exc

    def ensure_login(self) -> None:
        if self._logged_in:
            return

        self._open("/")
        payload = urllib.parse.urlencode(
            {
                "action": "login",
                "login": self._config.oko_login,
                "password": self._config.oko_password,
                "access_key": self._config.ag3nts_api_key,
            }
        ).encode("utf-8")
        response_html = self._open("/", data=payload)
        if "Logowanie operatora" in response_html or "Access key must be a valid API key." in response_html:
            raise OkoEditorError("OKO login failed with the configured AG3NTS_API_KEY.")
        self._logged_in = True

    def list_records(self, page: str) -> dict[str, Any]:
        self.ensure_login()
        normalized = normalize_page(page)
        page_path = page_list_path(normalized)
        html_text = self._open(page_path)
        entries = parse_list_entries(normalized, html_text)
        return {
            "page": normalized,
            "count": len(entries),
            "entries": entries,
        }

    def get_record(self, page: str, record_id: str) -> dict[str, Any]:
        self.ensure_login()
        normalized = normalize_page(page)
        record = normalize_record_id(record_id)
        detail_path = f"{page_detail_base_path(normalized)}/{record}"
        html_text = self._open(detail_path)
        return parse_detail_page(normalized, record, html_text)

    def batch_get_records(self, requests: list[dict[str, str]]) -> list[dict[str, Any]]:
        return [
            self.get_record(request_item["page"], request_item["id"])
            for request_item in requests
        ]


def normalize_page(page: str) -> str:
    candidate = page.strip().lower()
    if candidate not in {"incydenty", "notatki", "zadania"}:
        raise OkoEditorError('Page must be one of: "incydenty", "notatki", "zadania".')
    return candidate


def normalize_record_id(record_id: str) -> str:
    candidate = record_id.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", candidate):
        raise OkoEditorError("Record id must be a 32-character hex string.")
    return candidate


def page_list_path(page: str) -> str:
    return "/" if page == "incydenty" else f"/{page}"


def page_detail_base_path(page: str) -> str:
    return f"/{page}"


def strip_tags(value: str) -> str:
    return html.unescape(re.sub(r"<.*?>", "", value)).strip()


def parse_list_entries(page: str, html_text: str) -> list[dict[str, Any]]:
    if page in {"incydenty", "notatki"}:
        pattern = re.compile(
            r'<a class="entry-link" href="(?P<href>/[^"]+)">\s*'
            r'<article class="list-item(?: list-item--done)?">\s*<div>\s*'
            r"<strong>(?P<title>.*?)</strong>\s*<p>(?P<summary>.*?)</p>",
            re.S,
        )
        return [
            {
                "id": entry_id_from_href(match.group("href")),
                "href": match.group("href"),
                "title": strip_tags(match.group("title")),
                "summary": strip_tags(match.group("summary")),
            }
            for match in pattern.finditer(html_text)
        ]

    pattern = re.compile(
        r'<a class="task-main-link" href="(?P<href>/zadania/[0-9a-f]{32})">\s*'
        r"<strong>(?P<title>.*?)</strong>.*?"
        r'class="metric metric-link metric--(?P<status>pending|done)"',
        re.S,
    )
    return [
        {
            "id": entry_id_from_href(match.group("href")),
            "href": match.group("href"),
            "title": strip_tags(match.group("title")),
            "status": match.group("status"),
        }
        for match in pattern.finditer(html_text)
    ]


def parse_detail_page(page: str, record_id: str, html_text: str) -> dict[str, Any]:
    hero_title_match = re.search(r'<h2 class="hero-title">(.*?)</h2>', html_text, re.S)
    content_match = re.search(r'<p class="detail-content">(.*?)</p>', html_text, re.S)
    if not hero_title_match or not content_match:
        raise OkoEditorError(f"Could not parse detail page for {page}/{record_id}.")

    detail: dict[str, Any] = {
        "page": page,
        "id": record_id,
        "title": strip_tags(hero_title_match.group(1)),
        "content": strip_tags(content_match.group(1)),
    }
    if page == "zadania":
        status_match = re.search(r'class="metric metric-link metric--(pending|done)"', html_text)
        if status_match:
            detail["status"] = status_match.group(1)
    return detail


def entry_id_from_href(href: str) -> str:
    match = re.search(r"/([0-9a-f]{32})$", href)
    if not match:
        raise OkoEditorError(f"Could not extract entry id from href: {href}")
    return match.group(1)


def normalize_incident_title(title: str, content: str | None) -> str:
    candidate = title.strip()
    content_text = (content or "").strip()
    combined = f"{candidate}\n{content_text}".lower()

    prefix = ""
    if any(keyword in combined for keyword in ("zwier", "bobr", "faun", "ssak")):
        prefix = "MOVE04"
    elif any(keyword in combined for keyword in ("ludzie", "ludzi", "człow", "czlow", "sylwet")):
        prefix = "MOVE01"

    title_without_code = re.sub(r"^(MOVE|PROB|RECO)[0-9]{2}\s+", "", candidate, flags=re.I)
    if prefix:
        return f"{prefix} {title_without_code}".strip()
    return candidate


def is_animal_text(*values: str | None) -> bool:
    combined = "\n".join(value or "" for value in values).lower()
    return any(keyword in combined for keyword in ("zwier", "bobr", "faun", "ssak"))


def is_human_text(*values: str | None) -> bool:
    combined = "\n".join(value or "" for value in values).lower()
    return any(
        keyword in combined
        for keyword in ("ludzie", "ludzi", "człow", "czlow", "osob", "piesi", "sylwet", "ruch ludzi")
    )


def mentions_skolwin(*values: str | None) -> bool:
    return "skolwin" in "\n".join(value or "" for value in values).lower()


def mentions_komarowo(*values: str | None) -> bool:
    return "komarowo" in "\n".join(value or "" for value in values).lower()


def title_has_code(title: str | None, expected_code: str) -> bool:
    return bool(title and title.strip().upper().startswith(f"{expected_code.upper()} "))


def build_candidate_decoy_ids(incident_entries: list[dict[str, Any]]) -> list[str]:
    preferred: list[str] = []
    fallback: list[str] = []
    for entry in incident_entries:
        incident_id = entry["id"]
        if mentions_skolwin(entry.get("title"), entry.get("summary")):
            continue
        if mentions_komarowo(entry.get("title"), entry.get("summary")):
            preferred.append(incident_id)
            continue
        if "domatowo" not in f"{entry.get('title', '')}\n{entry.get('summary', '')}".lower():
            preferred.append(incident_id)
        else:
            fallback.append(incident_id)
    return preferred + fallback


def pick_existing_or_candidate_decoy_id(
    incident_entries: list[dict[str, Any]],
    candidate_ids: list[str],
    remembered_id: str | None = None,
) -> str | None:
    if remembered_id:
        return remembered_id

    existing_match = next(
        (
            entry["id"]
            for entry in incident_entries
            if title_has_code(entry.get("title"), "MOVE01")
            and mentions_komarowo(entry.get("title"), entry.get("summary"))
            and is_human_text(entry.get("title"), entry.get("summary"))
        ),
        None,
    )
    if existing_match:
        return existing_match

    return candidate_ids[0] if candidate_ids else None


def compact_text(value: str | None, *, limit: int = 220) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def compact_record(record: dict[str, Any], *, include_content: bool) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "page": record.get("page"),
        "id": record.get("id"),
        "title": compact_text(record.get("title"), limit=120),
    }
    if "status" in record:
        compact["status"] = record.get("status")
    if include_content:
        compact["content"] = compact_text(record.get("content"), limit=260)
    return compact


def compact_tool_result(tool_name: str, result: Any) -> Any:
    if tool_name == "batch_get_records":
        if not isinstance(result, list):
            return result
        return [compact_record(record, include_content=True) for record in result[:6]]

    if not isinstance(result, dict):
        return result

    if tool_name == "find_targets":
        return {
            "skolwin_incident_id": result.get("skolwin_incident_id"),
            "skolwin_task_id": result.get("skolwin_task_id"),
            "coding_note_id": result.get("coding_note_id"),
            "selected_decoy_id": result.get("selected_decoy_id"),
            "candidate_decoy_ids": (result.get("candidate_decoy_ids") or [])[:3],
        }

    if tool_name == "list_records":
        entries = result.get("entries") or []
        compact_entries: list[dict[str, Any]] = []
        for entry in entries[:8]:
            compact_entry = {
                "id": entry.get("id"),
                "title": compact_text(entry.get("title"), limit=100),
            }
            if "status" in entry:
                compact_entry["status"] = entry.get("status")
            if "summary" in entry:
                compact_entry["summary"] = compact_text(entry.get("summary"), limit=120)
            compact_entries.append(compact_entry)
        return {
            "page": result.get("page"),
            "count": result.get("count"),
            "entries": compact_entries,
        }

    if tool_name == "get_record":
        return compact_record(result, include_content=True)

    if tool_name == "update_record":
        compact: dict[str, Any] = {
            "status": result.get("status"),
            "code": result.get("code"),
            "message": result.get("message"),
        }
        operation = result.get("operation")
        if isinstance(operation, dict):
            compact["operation"] = {
                "page": operation.get("page"),
                "id": operation.get("id"),
            }
        updated = result.get("updated")
        if isinstance(updated, dict):
            compact["updated"] = {
                "title": compact_text(updated.get("title"), limit=100),
                "content": compact_text(updated.get("content"), limit=140),
                "done": updated.get("done"),
                "displayStatus": updated.get("displayStatus"),
            }
        sync = result.get("sync")
        if isinstance(sync, dict):
            compact["sync"] = {
                "visible": sync.get("visible"),
                "attempts": sync.get("attempts"),
            }
        return compact

    if tool_name == "batch_update_records":
        results = result.get("results") if isinstance(result, dict) else None
        compact_results: list[Any] = []
        if isinstance(results, list):
            for item in results[:6]:
                compact_results.append(compact_tool_result("update_record", item))
        return {
            "status": result.get("status") if isinstance(result, dict) else None,
            "count": result.get("count") if isinstance(result, dict) else None,
            "results": compact_results,
        }

    if tool_name == "evaluate_state":
        checks = result.get("checks") or {}
        compact_checks: dict[str, Any] = {}
        for key, value in checks.items():
            if not isinstance(value, dict):
                continue
            compact_value: dict[str, Any] = {"ok": value.get("ok")}
            record = value.get("record")
            if isinstance(record, dict):
                compact_value["record"] = compact_record(record, include_content=False)
            candidates = value.get("candidates")
            if isinstance(candidates, list) and candidates:
                first_match = next((item for item in candidates if item.get("ok")), candidates[0])
                compact_value["candidate"] = {
                    "id": first_match.get("id"),
                    "ok": first_match.get("ok"),
                }
                if isinstance(first_match.get("record"), dict):
                    compact_value["record"] = compact_record(first_match["record"], include_content=False)
            compact_checks[key] = compact_value
        return {
            "targets": result.get("targets"),
            "checks": compact_checks,
            "ready_for_done": result.get("ready_for_done"),
            "guidance": result.get("guidance"),
        }

    if tool_name == "submit_done":
        return {
            "code": result.get("code"),
            "message": result.get("message"),
            "flag": result.get("flag"),
            "agent_hint": result.get("agent_hint"),
        }

    return result


def normalize_expected_update(update: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(update)
    normalized["page"] = normalize_page(update["page"])
    normalized["id"] = normalize_record_id(update["id"])

    if normalized["page"] == "incydenty" and normalized.get("title") is not None:
        normalized["title"] = normalize_incident_title(
            str(normalized["title"]),
            str(normalized.get("content") or ""),
        )
        normalized.pop("done", None)
    elif normalized["page"] == "zadania" and normalized.get("done") is not None:
        normalized["done"] = str(normalized["done"]).strip().upper()

    return normalized


def record_matches_expected_update(record: dict[str, Any], expected: dict[str, Any]) -> bool:
    if expected.get("title") is not None and (record.get("title") or "").strip() != expected["title"].strip():
        return False
    if expected.get("content") is not None and (record.get("content") or "").strip() != str(expected["content"]).strip():
        return False
    if expected["page"] == "zadania" and expected.get("done") is not None:
        required_status = "done" if expected["done"] == "YES" else "pending"
        if record.get("status") != required_status:
            return False
    return True


def wait_for_visible_updates(
    web_client: OkoWebClient,
    updates: list[dict[str, Any]],
    *,
    timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    expected_updates = [normalize_expected_update(update) for update in updates]
    deadline = time.time() + timeout_seconds
    attempts = 0
    last_records: list[dict[str, Any]] = []

    while time.time() <= deadline:
        attempts += 1
        web_client.reset_session()
        current_records: list[dict[str, Any]] = []
        all_visible = True

        for expected in expected_updates:
            record = web_client.get_record(expected["page"], expected["id"])
            current_records.append(record)
            if not record_matches_expected_update(record, expected):
                all_visible = False

        if all_visible:
            return {
                "visible": True,
                "attempts": attempts,
                "records": [compact_record(record, include_content=False) for record in current_records],
            }

        last_records = current_records
        time.sleep(poll_interval_seconds)

    return {
        "visible": False,
        "attempts": attempts,
        "records": [compact_record(record, include_content=False) for record in last_records],
    }


class OkoVerifyClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def help(self) -> Any:
        return {
            "action": "help",
            "commands": [
                {"action": "help"},
                {
                    "action": "update",
                    "required_fields": ["page", "id", "action"],
                    "optional_fields": ["content", "title", "done"],
                    "rules": [
                        'At least one of "content" or "title" must be provided.',
                        '"done" is allowed only for page "zadania".',
                        'Page "uzytkownicy" is read-only.',
                    ],
                },
                {"action": "done"},
            ],
        }

    def update(
        self,
        *,
        page: str,
        record_id: str,
        title: str | None = None,
        content: str | None = None,
        done: str | None = None,
    ) -> Any:
        normalized_page = normalize_page(page)
        normalized_title = title
        normalized_content = content
        normalized_done = done

        if normalized_page == "incydenty":
            normalized_done = None
            if normalized_title is not None:
                normalized_title = normalize_incident_title(
                    normalized_title,
                    normalized_content,
                )

        payload: dict[str, Any] = {
            "page": normalized_page,
            "id": normalize_record_id(record_id),
            "action": "update",
        }
        if normalized_title is not None:
            payload["title"] = normalized_title
        if normalized_content is not None:
            payload["content"] = normalized_content
        if normalized_done is not None:
            normalized_done_value = normalized_done.strip().upper()
            if normalized_done_value not in {"YES", "NO"}:
                raise OkoEditorError('Task done must be "YES" or "NO".')
            payload["done"] = normalized_done_value
        return self._submit(payload)

    def done(self) -> Any:
        return self._submit({"action": "done"})

    def batch_update(self, updates: list[dict[str, Any]]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for update in updates:
            result = self.update(
                page=update["page"],
                record_id=update["id"],
                title=update.get("title"),
                content=update.get("content"),
                done=update.get("done"),
            )
            results.append(result if isinstance(result, dict) else {"result": result})
        return {
            "status": "success",
            "count": len(results),
            "results": results,
        }

    def _submit(self, answer: dict[str, Any]) -> Any:
        try:
            return submit_task_answer(
                self._config.verify_url,
                api_key=self._config.ag3nts_api_key,
                task=TASK_NAME,
                answer=answer,
                timeout_seconds=self._config.api_timeout_seconds,
            )
        except HttpRequestError as exc:
            raise OkoEditorError(str(exc)) from exc


def find_targets(web_client: OkoWebClient) -> dict[str, Any]:
    incidents = web_client.list_records("incydenty")
    tasks = web_client.list_records("zadania")
    notes = web_client.list_records("notatki")

    skolwin_incident = next(
        (
            entry
            for entry in incidents["entries"]
            if mentions_skolwin(entry.get("title"), entry.get("summary"))
        ),
        None,
    )
    skolwin_task = next(
        (
            entry
            for entry in tasks["entries"]
            if mentions_skolwin(entry.get("title"))
        ),
        None,
    )
    if not skolwin_task and skolwin_incident:
        try:
            candidate_task = web_client.get_record("zadania", skolwin_incident["id"])
        except OkoEditorError:
            candidate_task = None
        if candidate_task and mentions_skolwin(
            candidate_task.get("title"),
            candidate_task.get("content"),
        ):
            skolwin_task = {
                "id": candidate_task["id"],
                "title": candidate_task.get("title", ""),
                "status": candidate_task.get("status"),
            }
    coding_note = next(
        (
            entry
            for entry in notes["entries"]
            if "kodowania incydentów" in entry.get("title", "").lower()
        ),
        None,
    )

    candidate_decoys = build_candidate_decoy_ids(incidents["entries"])

    return {
        "skolwin_incident_id": skolwin_incident["id"] if skolwin_incident else None,
        "skolwin_task_id": skolwin_task["id"] if skolwin_task else None,
        "coding_note_id": coding_note["id"] if coding_note else None,
        "selected_decoy_id": pick_existing_or_candidate_decoy_id(
            incidents["entries"],
            candidate_decoys,
        ),
        "candidate_decoy_ids": candidate_decoys,
        "incidents": incidents["entries"],
        "tasks": tasks["entries"],
        "notes": notes["entries"],
    }


def evaluate_state(web_client: OkoWebClient, state: AgentState) -> dict[str, Any]:
    targets = state.targets or find_targets(web_client)
    incidents = web_client.list_records("incydenty")
    targets["incidents"] = incidents["entries"]
    targets["candidate_decoy_ids"] = build_candidate_decoy_ids(incidents["entries"])
    targets["selected_decoy_id"] = pick_existing_or_candidate_decoy_id(
        incidents["entries"],
        targets["candidate_decoy_ids"],
        state.decoy_incident_id,
    )
    state.targets = targets

    result: dict[str, Any] = {
        "targets": {
            "skolwin_incident_id": targets.get("skolwin_incident_id"),
            "skolwin_task_id": targets.get("skolwin_task_id"),
            "selected_decoy_id": targets.get("selected_decoy_id"),
            "candidate_decoy_ids": targets.get("candidate_decoy_ids", []),
        },
        "checks": {},
        "ready_for_done": False,
        "guidance": [],
    }

    skolwin_incident_id = targets.get("skolwin_incident_id")
    skolwin_task_id = targets.get("skolwin_task_id")
    selected_decoy_id = targets.get("selected_decoy_id")

    if not skolwin_incident_id:
        result["guidance"].append("Skolwin incident was not found in the incident list.")
    else:
        skolwin_incident = web_client.get_record("incydenty", skolwin_incident_id)
        incident_ok = (
            mentions_skolwin(skolwin_incident.get("title"), skolwin_incident.get("content"))
            and title_has_code(skolwin_incident.get("title"), "MOVE04")
            and is_animal_text(skolwin_incident.get("title"), skolwin_incident.get("content"))
            and not is_human_text(skolwin_incident.get("title"), skolwin_incident.get("content"))
        )
        result["checks"]["skolwin_incident"] = {
            "ok": incident_ok,
            "record": skolwin_incident,
            "expected": "Skolwin incident should describe animals and use MOVE04.",
        }
        if not incident_ok:
            result["guidance"].append(
                "Rewrite the Skolwin incident so it clearly describes animals and starts with MOVE04."
            )

    if not skolwin_task_id:
        result["guidance"].append("Skolwin task was not found in the task list.")
    else:
        skolwin_task = web_client.get_record("zadania", skolwin_task_id)
        task_ok = (
            mentions_skolwin(skolwin_task.get("title"), skolwin_task.get("content"))
            and skolwin_task.get("status") == "done"
            and is_animal_text(skolwin_task.get("title"), skolwin_task.get("content"))
        )
        result["checks"]["skolwin_task"] = {
            "ok": task_ok,
            "record": skolwin_task,
            "expected": 'Skolwin task should be done and clearly mention animals, for example "Widziano tam zwierzęta, na przykład bobry."',
        }
        if not task_ok:
            result["guidance"].append(
                'Update the Skolwin task, mark it done, and use a simple sentence about animals such as "Widziano tam zwierzęta, na przykład bobry."'
            )

    decoy_matches: list[dict[str, Any]] = []
    if selected_decoy_id:
        record = web_client.get_record("incydenty", selected_decoy_id)
        decoy_ok = (
            mentions_komarowo(record.get("title"), record.get("content"))
            and title_has_code(record.get("title"), "MOVE01")
            and is_human_text(record.get("title"), record.get("content"))
        )
        decoy_matches.append({"id": selected_decoy_id, "ok": decoy_ok, "record": record})
    else:
        decoy_ok = False
    result["checks"]["komarowo_decoy"] = {
        "ok": decoy_ok,
        "candidates": decoy_matches,
        "expected": "At least one non-Skolwin incident should describe human movement near Komarowo and start with MOVE01.",
    }
    if not decoy_ok:
        result["guidance"].append(
            "Repurpose one non-Skolwin incident so it reports human movement near Komarowo and starts with MOVE01."
        )

    result["ready_for_done"] = all(
        check.get("ok") is True for check in result["checks"].values()
    )
    if result["ready_for_done"]:
        result["guidance"].append("All visible requirements are satisfied. Submit done now.")

    return result


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_targets",
            "description": "Find the current IDs for the Skolwin incident, Skolwin task, coding note, and candidate decoy incidents.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_records",
            "description": "List visible records on one OKO page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "string",
                        "enum": ["incydenty", "notatki", "zadania"],
                    }
                },
                "required": ["page"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_record",
            "description": "Fetch the full title and body for one OKO record.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "string",
                        "enum": ["incydenty", "notatki", "zadania"],
                    },
                    "id": {"type": "string"},
                },
                "required": ["page", "id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_get_records",
            "description": "Fetch multiple OKO records in one tool call. Prefer this over repeated get_record calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "page": {
                                    "type": "string",
                                    "enum": ["incydenty", "notatki", "zadania"],
                                },
                                "id": {"type": "string"},
                            },
                            "required": ["page", "id"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                        "maxItems": 6,
                    }
                },
                "required": ["requests"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_record",
            "description": "Update one OKO record through the central verify API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "string",
                        "enum": ["incydenty", "notatki", "zadania"],
                    },
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "done": {"type": "string", "enum": ["YES", "NO"]},
                },
                "required": ["page", "id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_update_records",
            "description": "Update multiple OKO records in one tool call. Prefer this when applying the final set of edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "updates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "page": {
                                    "type": "string",
                                    "enum": ["incydenty", "notatki", "zadania"],
                                },
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "content": {"type": "string"},
                                "done": {"type": "string", "enum": ["YES", "NO"]},
                            },
                            "required": ["page", "id"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                        "maxItems": 6,
                    }
                },
                "required": ["updates"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_done",
            "description": "Ask the central verify API whether all required edits are complete.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_state",
            "description": "Inspect the live OKO state and report whether the Skolwin incident, Skolwin task, and Komarowo decoy satisfy the task requirements.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def execute_tool_call(
    tool_call: ToolCall,
    *,
    web_client: OkoWebClient,
    verify_client: OkoVerifyClient,
    state: AgentState,
    show_tool_results: bool,
) -> dict[str, Any]:
    def tool_result_failed(payload: Any) -> bool:
        return isinstance(payload, dict) and payload.get("ok") is False

    def has_syncable_update(update: dict[str, Any]) -> bool:
        record_id = update.get("id")
        return (
            update.get("page") in {"incydenty", "zadania"}
            and isinstance(record_id, str)
            and bool(re.fullmatch(r"[0-9a-f]{32}", record_id.strip().lower()))
        )

    handlers = {
        "find_targets": lambda _: find_targets(web_client),
        "list_records": lambda args: web_client.list_records(args["page"]),
        "get_record": lambda args: web_client.get_record(args["page"], args["id"]),
        "batch_get_records": lambda args: web_client.batch_get_records(args["requests"]),
        "update_record": lambda args: verify_client.update(
            page=args["page"],
            record_id=args["id"],
            title=args.get("title"),
            content=args.get("content"),
            done=args.get("done"),
        ),
        "batch_update_records": lambda args: verify_client.batch_update(args["updates"]),
        "submit_done": lambda _: verify_client.done(),
        "evaluate_state": lambda _: evaluate_state(web_client, state),
    }

    if tool_call.name not in handlers:
        raise OkoEditorError(f"Unknown tool called by the model: {tool_call.name}")

    try:
        result = handlers[tool_call.name](tool_call.arguments)
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "error": str(exc),
            "tool": tool_call.name,
            "arguments": tool_call.arguments,
        }

    if tool_call.name == "find_targets" and isinstance(result, dict):
        state.targets = result

    if tool_call.name == "update_record":
        if not tool_result_failed(result) and has_syncable_update(tool_call.arguments):
            try:
                sync_result = wait_for_visible_updates(web_client, [tool_call.arguments])
            except Exception as exc:  # noqa: BLE001
                sync_result = {"visible": False, "error": str(exc)}
            if isinstance(result, dict):
                result["sync"] = sync_result
        state.last_update_response = result
        write_json(LAST_UPDATE_RESPONSE_PATH, result)
        if (
            tool_call.arguments.get("page") == "incydenty"
            and mentions_komarowo(tool_call.arguments.get("title"), tool_call.arguments.get("content"))
            and is_human_text(tool_call.arguments.get("title"), tool_call.arguments.get("content"))
        ):
            state.decoy_incident_id = tool_call.arguments.get("id")
    if tool_call.name == "batch_update_records":
        sync_updates = [
            update
            for update in tool_call.arguments.get("updates", [])
            if has_syncable_update(update)
        ]
        if not tool_result_failed(result) and sync_updates:
            try:
                sync_result = wait_for_visible_updates(web_client, sync_updates)
            except Exception as exc:  # noqa: BLE001
                sync_result = {"visible": False, "error": str(exc)}
            if isinstance(result, dict):
                result["sync"] = sync_result
        state.last_update_response = result
        write_json(LAST_UPDATE_RESPONSE_PATH, result)
        for update in tool_call.arguments.get("updates", []):
            if (
                update.get("page") == "incydenty"
                and mentions_komarowo(update.get("title"), update.get("content"))
                and is_human_text(update.get("title"), update.get("content"))
            ):
                state.decoy_incident_id = update.get("id")
    if tool_call.name == "submit_done":
        state.final_response = result
        write_json(LAST_DONE_RESPONSE_PATH, result)
        direct_flag = extract_flag(result)
        if direct_flag:
            state.final_flag = direct_flag
            state.completed = True
        if isinstance(result, dict) and result.get("code") == 0:
            state.completed = True
        if isinstance(result, dict):
            error_text = json.dumps(result, ensure_ascii=False).lower()
            if "note's content does not meet the requirements" in error_text or "#osb" in error_text:
                result["agent_hint"] = (
                    'The Skolwin task content is probably too indirect. Rewrite it in simple Polish, for example: '
                    '"Widziano tam zwierzęta, na przykład bobry."'
                )
            for key in ("flag", "answer", "message"):
                value = result.get(key)
                if isinstance(value, str) and "{" in value and "}" in value:
                    state.final_flag = value
                    state.completed = True
                    break

    if show_tool_results:
        logger.info(
            "Tool {}({}) -> {}",
            tool_call.name,
            json.dumps(tool_call.arguments, ensure_ascii=False),
            json.dumps(result, ensure_ascii=False, indent=2),
        )

    compact_result = compact_tool_result(tool_call.name, result)

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(compact_result, ensure_ascii=False),
    }


def parse_tool_message_content(tool_message: dict[str, Any]) -> dict[str, Any]:
    """Decode the JSON payload stored in a synthetic tool message."""

    content = tool_message.get("content")
    if not isinstance(content, str):
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def evaluate_and_maybe_submit_done(
    *,
    step: int,
    suffix: str,
    messages: list[dict[str, Any]],
    web_client: OkoWebClient,
    verify_client: OkoVerifyClient,
    state: AgentState,
    show_tool_results: bool,
) -> bool:
    """Evaluate the live state and auto-submit when the task is already complete."""

    evaluation_message = execute_tool_call(
        ToolCall(id=f"auto_evaluate_state_{step}_{suffix}", name="evaluate_state", arguments={}),
        web_client=web_client,
        verify_client=verify_client,
        state=state,
        show_tool_results=show_tool_results,
    )
    messages.append(evaluation_message)
    evaluation = parse_tool_message_content(evaluation_message)
    if evaluation.get("ready_for_done") is not True:
        return False

    done_message = execute_tool_call(
        ToolCall(id=f"auto_submit_done_{step}_{suffix}", name="submit_done", arguments={}),
        web_client=web_client,
        verify_client=verify_client,
        state=state,
        show_tool_results=show_tool_results,
    )
    messages.append(done_message)
    return bool(state.final_flag or state.completed)


def submit_done_from_ready_state(
    *,
    step: int,
    suffix: str,
    messages: list[dict[str, Any]],
    web_client: OkoWebClient,
    verify_client: OkoVerifyClient,
    state: AgentState,
    show_tool_results: bool,
) -> bool:
    """Submit completion after an already-confirmed ready state."""

    done_message = execute_tool_call(
        ToolCall(id=f"auto_submit_done_{step}_{suffix}", name="submit_done", arguments={}),
        web_client=web_client,
        verify_client=verify_client,
        state=state,
        show_tool_results=show_tool_results,
    )
    messages.append(done_message)
    return bool(state.final_flag or state.completed)


def recover_plaintext_tool_calls(response_text: str, *, step: int) -> list[ToolCall]:
    """Decode malformed plaintext tool-call dumps emitted by weaker models."""

    normalized = response_text.strip()
    if '"name"' not in normalized or '"arguments"' not in normalized:
        return []

    payload_start = normalized.find("[")
    if payload_start < 0:
        payload_start = normalized.find("{")
    if payload_start < 0:
        return []

    try:
        parsed = parse_json_content(normalized[payload_start:])
    except OpenRouterError:
        return []

    raw_calls: list[dict[str, Any]]
    if isinstance(parsed, dict):
        raw_calls = [parsed]
    elif isinstance(parsed, list):
        raw_calls = [item for item in parsed if isinstance(item, dict)]
    else:
        return []

    recovered: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls, start=1):
        name = raw_call.get("name")
        arguments = raw_call.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        recovered.append(
            ToolCall(
                id=f"recovered_tool_call_{step}_{index}",
                name=name,
                arguments=arguments,
            )
        )
    return recovered


def recover_from_non_tool_response(
    *,
    step: int,
    messages: list[dict[str, Any]],
    response_text: str,
    web_client: OkoWebClient,
    verify_client: OkoVerifyClient,
    state: AgentState,
    show_tool_results: bool,
) -> bool:
    """Recover when the model answers in plain text instead of using tools."""

    logger.info("Model returned a non-tool response: {}", response_text)
    stripped_response = response_text.strip()
    recovered_tool_calls = recover_plaintext_tool_calls(stripped_response, step=step)
    if recovered_tool_calls:
        for tool_call in recovered_tool_calls:
            tool_message = execute_tool_call(
                tool_call,
                web_client=web_client,
                verify_client=verify_client,
                state=state,
                show_tool_results=show_tool_results,
            )
            messages.append(tool_message)
            if tool_call.name == "evaluate_state":
                evaluation = parse_tool_message_content(tool_message)
                if evaluation.get("ready_for_done") is True:
                    if submit_done_from_ready_state(
                        step=step,
                        suffix="recovered_evaluate_state",
                        messages=messages,
                        web_client=web_client,
                        verify_client=verify_client,
                        state=state,
                        show_tool_results=show_tool_results,
                    ):
                        return True
            elif tool_call.name in {"update_record", "batch_update_records"}:
                if evaluate_and_maybe_submit_done(
                    step=step,
                    suffix=f"recovered_{tool_call.name}",
                    messages=messages,
                    web_client=web_client,
                    verify_client=verify_client,
                    state=state,
                    show_tool_results=show_tool_results,
                ):
                    return True
            elif tool_call.name == "submit_done" and (state.final_flag or state.completed):
                return True
        return bool(state.final_flag or state.completed)

    if stripped_response.startswith(("TOOLCALL>", "CALL>", "ALL>", "OLCALL>", ">[")):
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous message looked like a malformed tool call. "
                    "Retry immediately using real tool calls with strict JSON arguments."
                ),
            }
        )
        return False
    if evaluate_and_maybe_submit_done(
        step=step,
        suffix="recovery",
        messages=messages,
        web_client=web_client,
        verify_client=verify_client,
        state=state,
        show_tool_results=show_tool_results,
    ):
        return True

    messages.append(
        {
            "role": "user",
            "content": (
                "Use tools only. Do not summarize progress in plain text. "
                "If the task is complete, call evaluate_state and then submit_done. "
                "Otherwise apply the missing edits with update_record or batch_update_records."
            ),
        }
    )
    return False


def run_agent(config: AppConfig) -> tuple[list[dict[str, Any]], AgentState]:
    client = build_task_openrouter_client(
        __file__,
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_url,
        model=config.model,
        task_name=TASK_NAME,
        timeout_seconds=DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
        site_url=config.site_url,
        site_name=config.site_name,
    )
    web_client = OkoWebClient(config)
    verify_client = OkoVerifyClient(config)
    state = AgentState()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INITIAL_USER_PROMPT},
    ]

    for step in range(1, config.max_steps + 1):
        logger.info("Starting tool-calling step {}/{}.", step, config.max_steps)
        delay_seconds = max(0.1, STEP_RETRY_BASE_DELAY_SECONDS)
        completion = None
        for attempt in range(1, STEP_RETRY_ATTEMPTS + 1):
            try:
                completion = client.create_completion(messages, tools=TOOLS)
                break
            except OpenRouterError as exc:
                if attempt >= STEP_RETRY_ATTEMPTS or not is_retryable_openrouter_error(exc):
                    raise
                logger.warning(
                    "Transient OpenRouter failure on step {}/{} attempt {}/{}: {}. Retrying in {:.1f}s.",
                    step,
                    config.max_steps,
                    attempt,
                    STEP_RETRY_ATTEMPTS,
                    exc,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
                delay_seconds *= 2
        if completion is None:
            raise OkoEditorError("OpenRouter did not produce a completion.")
        assistant_message: dict[str, Any] = {"role": "assistant"}
        if completion.content:
            assistant_message["content"] = completion.content
        if completion.tool_calls:
            assistant_message["tool_calls"] = [
                tool_call.to_message_dict() for tool_call in completion.tool_calls
            ]
        messages.append(assistant_message)

        if completion.tool_calls:
            for tool_call in completion.tool_calls:
                tool_message = execute_tool_call(
                    tool_call,
                    web_client=web_client,
                    verify_client=verify_client,
                    state=state,
                    show_tool_results=config.show_tool_results,
                )
                messages.append(tool_message)
                if tool_call.name == "evaluate_state":
                    evaluation = parse_tool_message_content(tool_message)
                    if evaluation.get("ready_for_done") is True:
                        if submit_done_from_ready_state(
                            step=step,
                            suffix="evaluate_state",
                            messages=messages,
                            web_client=web_client,
                            verify_client=verify_client,
                            state=state,
                            show_tool_results=config.show_tool_results,
                        ):
                            return messages, state
                elif tool_call.name in {"update_record", "batch_update_records"}:
                    if evaluate_and_maybe_submit_done(
                        step=step,
                        suffix=tool_call.name,
                        messages=messages,
                        web_client=web_client,
                        verify_client=verify_client,
                        state=state,
                        show_tool_results=config.show_tool_results,
                    ):
                        return messages, state
            if state.final_flag or state.completed:
                return messages, state
            continue

        if completion.content:
            if recover_from_non_tool_response(
                step=step,
                messages=messages,
                response_text=completion.content,
                web_client=web_client,
                verify_client=verify_client,
                state=state,
                show_tool_results=config.show_tool_results,
            ):
                return messages, state

    raise OkoEditorError("OpenRouter tool calling did not finish within the step limit.")


def main() -> int:
    args = parse_args()
    configure_logging()
    config = build_config(args)

    logger.info("Using model {} for task {}.", config.model, TASK_NAME)

    try:
        transcript, state = run_agent(config)
    except (OpenRouterError, OkoEditorError) as exc:
        logger.error("Solver failed: {}", exc)
        return 1

    write_json(Path(args.transcript_path), transcript)

    if state.final_flag:
        logger.success("Flag: {}", state.final_flag)
    else:
        logger.success(
            "Agent finished. Final response: {}",
            json.dumps(state.final_response, ensure_ascii=False, indent=2),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
