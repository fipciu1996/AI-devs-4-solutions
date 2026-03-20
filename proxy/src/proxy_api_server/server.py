"""HTTP server entrypoint for the proxy package redirection assistant."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from devs_utilities.logging import configure_logging, logger as shared_logger
from proxy_api_server.config import Settings, load_settings
from proxy_api_server.models import (
    ConversationRequest,
    ConversationResponse,
    InvalidConversationRequest,
)
from proxy_api_server.openrouter import OpenRouterClient, OpenRouterError, ToolCall
from proxy_api_server.packages_api import PackagesApiClient, PackagesApiError

logger = shared_logger.bind(component="proxy_api_server")
LOG_FILE_LOCK = Lock()
SESSION_LOCKS_GUARD = Lock()
SESSION_LOCKS: dict[str, Lock] = {}
PACKAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_package",
            "description": "Check the current status and location of a package by package ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "packageid": {
                        "type": "string",
                        "description": "Package ID, for example PKG12345678.",
                    }
                },
                "required": ["packageid"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "redirect_package",
            "description": (
                "Redirect a package to a destination using the security code "
                "provided by the operator. If the tool returns confirmation, "
                "that confirmation must be passed back to the operator."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "packageid": {
                        "type": "string",
                        "description": "Package ID, for example PKG12345678.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination code, for example PWR3847PL.",
                    },
                    "code": {
                        "type": "string",
                        "description": "Security code provided by the operator.",
                    },
                },
                "required": ["packageid", "destination", "code"],
                "additionalProperties": False,
            },
        },
    },
]


def _normalize_path(raw_path: str) -> str:
    parsed_path = urlparse(raw_path).path.rstrip("/")
    return parsed_path or "/"


def _get_session_lock(session_id: str) -> Lock:
    with SESSION_LOCKS_GUARD:
        session_lock = SESSION_LOCKS.get(session_id)
        if session_lock is None:
            session_lock = Lock()
            SESSION_LOCKS[session_id] = session_lock
        return session_lock


def load_session_messages(
    log_file: Path,
    session_id: str,
    system_prompt: str,
    max_context_messages: int,
) -> list[dict[str, Any]]:
    """Load prior conversation turns for a single session from the log file."""

    history_messages: list[dict[str, Any]] = []
    if log_file.exists():
        with LOG_FILE_LOCK:
            with log_file.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    raw_line = line.strip()
                    if not raw_line:
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping malformed conversation log entry at line {}",
                            line_number,
                        )
                        continue

                    if record.get("sessionID") != session_id:
                        continue

                    request_data = record.get("request")
                    response_data = record.get("response")

                    if isinstance(request_data, dict) and isinstance(
                        request_data.get("msg"),
                        str,
                    ):
                        history_messages.append(
                            {
                                "role": "user",
                                "content": request_data["msg"],
                            }
                        )
                    if isinstance(response_data, dict) and isinstance(
                        response_data.get("msg"),
                        str,
                    ):
                        history_messages.append(
                            {
                                "role": "assistant",
                                "content": response_data["msg"],
                            }
                        )

    if max_context_messages > 0:
        history_messages = history_messages[-max_context_messages:]
    else:
        history_messages = []

    return [{"role": "system", "content": system_prompt}, *history_messages]


def append_conversation_log(
    log_file: Path,
    request_data: ConversationRequest,
    response_data: ConversationResponse,
    client_ip: str,
    model_name: str,
    package_actions: list[dict[str, Any]],
) -> None:
    """Append a single conversation exchange as a JSON line."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "clientIP": client_ip,
        "sessionID": request_data.session_id,
        "model": model_name,
        "request": request_data.to_api_dict(),
        "response": response_data.to_api_dict(),
    }
    if package_actions:
        record["packageActions"] = package_actions

    with LOG_FILE_LOCK:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_conversation_log(log_file: Path) -> bytes:
    """Return the raw conversation log file contents."""

    if not log_file.exists():
        return b""

    with LOG_FILE_LOCK:
        return log_file.read_bytes()


def read_conversation_records(log_file: Path) -> tuple[list[dict[str, Any]], int]:
    """Read and parse conversation log entries."""

    if not log_file.exists():
        return [], 0

    records: list[dict[str, Any]] = []
    skipped_count = 0
    with LOG_FILE_LOCK:
        with log_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw_line = line.strip()
                if not raw_line:
                    continue
                try:
                    parsed = json.loads(raw_line)
                except json.JSONDecodeError:
                    skipped_count += 1
                    continue
                if isinstance(parsed, dict):
                    records.append(parsed)
                else:
                    skipped_count += 1

    return records, skipped_count


def _format_timestamp(raw_timestamp: Any) -> str:
    if not isinstance(raw_timestamp, str) or not raw_timestamp.strip():
        return "Brak czasu"
    try:
        parsed = datetime.fromisoformat(raw_timestamp)
    except ValueError:
        return raw_timestamp
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _format_package_action(action: dict[str, Any]) -> str:
    tool_name = html.escape(str(action.get("tool", "unknown")))
    status_label = "OK" if action.get("ok") else "ERROR"
    details: list[str] = []

    arguments = action.get("arguments")
    if isinstance(arguments, dict):
        packageid = arguments.get("packageid")
        if isinstance(packageid, str) and packageid.strip():
            details.append(f"packageid={html.escape(packageid.strip())}")
        destination = arguments.get("destination")
        if isinstance(destination, str) and destination.strip():
            details.append(f"destination={html.escape(destination.strip())}")

    tool_result = action.get("toolResult")
    if isinstance(tool_result, dict):
        result = tool_result.get("result")
        if isinstance(result, dict):
            status = result.get("status")
            if isinstance(status, str) and status.strip():
                details.append(f"status={html.escape(status.strip())}")
            location = result.get("location")
            if isinstance(location, str) and location.strip():
                details.append(f"location={html.escape(location.strip())}")
            confirmation = result.get("confirmation")
            if isinstance(confirmation, str) and confirmation.strip():
                details.append(f"confirmation={html.escape(confirmation.strip())}")
        error = tool_result.get("error")
        if isinstance(error, str) and error.strip():
            details.append(f"error={html.escape(error.strip())}")

    details_html = "<br>".join(details) if details else "Brak dodatkowych danych"
    return (
        '<div class="event">'
        f'<div class="event-title">{tool_name} · {status_label}</div>'
        f'<div class="event-body">{details_html}</div>'
        "</div>"
    )


def _render_message_bubble(
    author_label: str,
    message_text: str,
    *,
    bubble_class: str,
) -> str:
    rendered_text = html.escape(message_text).replace("\n", "<br>")
    return (
        f'<div class="bubble-row {bubble_class}">'
        f'<div class="bubble">'
        f'<div class="bubble-author">{html.escape(author_label)}</div>'
        f'<div class="bubble-text">{rendered_text}</div>'
        "</div>"
        "</div>"
    )


def build_logs_html(log_file: Path) -> bytes:
    """Render the conversation log as a simple WhatsApp-like HTML view."""

    records, skipped_count = read_conversation_records(log_file)
    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        session_id = record.get("sessionID")
        if not isinstance(session_id, str) or not session_id.strip():
            session_id = "unknown-session"
        grouped_records.setdefault(session_id, []).append(record)

    sorted_sessions = sorted(
        grouped_records.items(),
        key=lambda item: item[1][-1].get("timestamp", ""),
        reverse=True,
    )

    if not sorted_sessions:
        sessions_html = (
            '<section class="empty-state">'
            "<h2>Brak rozmow</h2>"
            "<p>Plik conversation.log jest pusty albo jeszcze nie istnieje.</p>"
            "</section>"
        )
    else:
        session_blocks: list[str] = []
        for session_id, session_records in sorted_sessions:
            messages_html: list[str] = []
            for record in session_records:
                timestamp = _format_timestamp(record.get("timestamp"))
                messages_html.append(
                    f'<div class="timestamp">{html.escape(timestamp)}</div>'
                )

                request_data = record.get("request")
                if isinstance(request_data, dict) and isinstance(
                    request_data.get("msg"),
                    str,
                ):
                    messages_html.append(
                        _render_message_bubble(
                            "Operator",
                            request_data["msg"],
                            bubble_class="outgoing",
                        )
                    )

                package_actions = record.get("packageActions")
                if isinstance(package_actions, list):
                    for action in package_actions:
                        if isinstance(action, dict):
                            messages_html.append(_format_package_action(action))

                response_data = record.get("response")
                if isinstance(response_data, dict) and isinstance(
                    response_data.get("msg"),
                    str,
                ):
                    messages_html.append(
                        _render_message_bubble(
                            "Asystent",
                            response_data["msg"],
                            bubble_class="incoming",
                        )
                    )

            session_meta = session_records[-1]
            model_name = session_meta.get("model")
            model_label = html.escape(str(model_name)) if model_name else "n/a"
            session_blocks.append(
                '<section class="session-card">'
                '<div class="session-header">'
                f'<div><h2>{html.escape(session_id)}</h2>'
                f'<p>{len(session_records)} wymian · model: {model_label}</p></div>'
                f'<span class="session-time">{html.escape(_format_timestamp(session_meta.get("timestamp")))}</span>'
                "</div>"
                f'<div class="chat-thread">{"".join(messages_html)}</div>'
                "</section>"
            )
        sessions_html = "".join(session_blocks)

    skipped_html = (
        f'<p class="skipped">Pominiete uszkodzone wpisy: {skipped_count}</p>'
        if skipped_count
        else ""
    )
    raw_link = html.escape(f"/logs/raw")
    page = f"""<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Conversation Logs</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #efeae2;
      --panel: #ffffff;
      --border: #d7d7d7;
      --muted: #5f6b76;
      --incoming: #ffffff;
      --outgoing: #d9fdd3;
      --event: #f6f7f8;
      --accent: #0b8f72;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(11, 143, 114, 0.12), transparent 26%),
        linear-gradient(180deg, #f5f1ea 0%, var(--bg) 100%);
      color: #1f2d3a;
    }}
    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
      margin-bottom: 20px;
    }}
    .topbar h1 {{
      margin: 0;
      font-size: 28px;
    }}
    .topbar p {{
      margin: 6px 0 0;
      color: var(--muted);
    }}
    .topbar a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
      white-space: nowrap;
    }}
    .session-card, .empty-state {{
      background: rgba(255, 255, 255, 0.82);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.7);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 12px 35px rgba(26, 47, 61, 0.08);
      margin-bottom: 18px;
    }}
    .session-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      padding-bottom: 14px;
      border-bottom: 1px solid rgba(31, 45, 58, 0.08);
      margin-bottom: 16px;
    }}
    .session-header h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .session-header p, .session-time, .skipped {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .chat-thread {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .timestamp {{
      align-self: center;
      font-size: 12px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid rgba(31, 45, 58, 0.08);
      border-radius: 999px;
      padding: 4px 10px;
      margin: 6px 0 2px;
    }}
    .bubble-row {{
      display: flex;
    }}
    .bubble-row.incoming {{
      justify-content: flex-start;
    }}
    .bubble-row.outgoing {{
      justify-content: flex-end;
    }}
    .bubble {{
      max-width: min(78%, 720px);
      border-radius: 18px;
      padding: 12px 14px;
      box-shadow: 0 3px 10px rgba(31, 45, 58, 0.08);
      border: 1px solid rgba(31, 45, 58, 0.06);
    }}
    .incoming .bubble {{
      background: var(--incoming);
      border-top-left-radius: 6px;
    }}
    .outgoing .bubble {{
      background: var(--outgoing);
      border-top-right-radius: 6px;
    }}
    .bubble-author {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .bubble-text {{
      line-height: 1.5;
      white-space: normal;
      word-break: break-word;
    }}
    .event {{
      align-self: center;
      width: min(84%, 760px);
      background: var(--event);
      border: 1px dashed rgba(31, 45, 58, 0.16);
      border-radius: 16px;
      padding: 10px 12px;
    }}
    .event-title {{
      font-weight: 700;
      margin-bottom: 6px;
      font-size: 13px;
    }}
    .event-body {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      word-break: break-word;
    }}
    .empty-state {{
      text-align: center;
      padding: 40px 24px;
    }}
    @media (max-width: 720px) {{
      .topbar {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .session-header {{
        flex-direction: column;
      }}
      .bubble, .event {{
        max-width: 100%;
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="topbar">
      <div>
        <h1>Logi konwersacji</h1>
        <p>Widok rozmow pogrupowanych po sessionID, w ukladzie zblizonym do komunikatora.</p>
        {skipped_html}
      </div>
      <a href="{raw_link}">Zobacz surowy plik</a>
    </header>
    {sessions_html}
  </main>
</body>
</html>"""
    return page.encode("utf-8")


def _require_tool_argument(
    arguments: dict[str, Any],
    name: str,
    field_name: str,
) -> str:
    value = arguments.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"Tool {name} requires string field '{field_name}'.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"Tool {name} field '{field_name}' must not be empty.")
    return normalized


def execute_package_tool_call(
    packages_client: PackagesApiClient,
    tool_call: ToolCall,
) -> dict[str, Any]:
    """Execute a single package-related tool call."""

    try:
        if tool_call.name == "check_package":
            packageid = _require_tool_argument(
                tool_call.arguments,
                tool_call.name,
                "packageid",
            )
            result = packages_client.check_package(packageid)
            return {
                "ok": True,
                "tool": tool_call.name,
                "packageid": packageid,
                "result": result,
            }

        if tool_call.name == "redirect_package":
            packageid = _require_tool_argument(
                tool_call.arguments,
                tool_call.name,
                "packageid",
            )
            destination = _require_tool_argument(
                tool_call.arguments,
                tool_call.name,
                "destination",
            )
            code = _require_tool_argument(
                tool_call.arguments,
                tool_call.name,
                "code",
            )
            result = packages_client.redirect_package(packageid, destination, code)
            return {
                "ok": True,
                "tool": tool_call.name,
                "packageid": packageid,
                "destination": destination,
                "result": result,
            }

        return {
            "ok": False,
            "tool": tool_call.name,
            "error": f"Unsupported tool '{tool_call.name}'.",
        }
    except (PackagesApiError, ValueError) as exc:
        return {
            "ok": False,
            "tool": tool_call.name,
            "error": str(exc),
        }


def _assistant_tool_message(tool_calls: list[ToolCall], content: str | None) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [tool_call.to_message_dict() for tool_call in tool_calls],
    }


def _tool_result_message(tool_call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result, ensure_ascii=False),
    }


def _ensure_confirmation_in_response(
    assistant_message: str,
    package_actions: list[dict[str, Any]],
) -> str:
    for action in reversed(package_actions):
        if action.get("tool") != "redirect_package" or not action.get("ok"):
            continue
        tool_result = action.get("toolResult")
        if not isinstance(tool_result, dict):
            tool_result = action.get("result")
        if not isinstance(tool_result, dict):
            continue
        result = tool_result.get("result")
        if not isinstance(result, dict):
            continue
        confirmation = result.get("confirmation")
        if not isinstance(confirmation, str) or not confirmation.strip():
            continue
        confirmation = confirmation.strip()
        if confirmation in assistant_message:
            return assistant_message
        return (
            f"{assistant_message.rstrip()}\n\nKod confirmation: {confirmation}"
            if assistant_message.strip()
            else f"Kod confirmation: {confirmation}"
        )
    return assistant_message


def generate_operator_response(
    settings: Settings,
    openrouter_client: OpenRouterClient,
    packages_client: PackagesApiClient,
    messages: list[dict[str, Any]],
) -> tuple[ConversationResponse, list[dict[str, Any]]]:
    """Run the OpenRouter conversation loop with package tool support."""

    conversation_messages = list(messages)
    package_actions: list[dict[str, Any]] = []

    for _ in range(settings.max_tool_round_trips):
        completion = openrouter_client.create_completion(
            conversation_messages,
            tools=PACKAGE_TOOLS,
        )

        if completion.tool_calls:
            conversation_messages.append(
                _assistant_tool_message(completion.tool_calls, completion.content)
            )
            for tool_call in completion.tool_calls:
                tool_result = execute_package_tool_call(packages_client, tool_call)
                package_actions.append(
                    {
                        "id": tool_call.id,
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "ok": bool(tool_result.get("ok")),
                        "toolResult": tool_result,
                    }
                )
                conversation_messages.append(
                    _tool_result_message(tool_call.id, tool_result)
                )
            continue

        assistant_message = completion.content or ""
        assistant_message = _ensure_confirmation_in_response(
            assistant_message,
            package_actions,
        )
        return ConversationResponse(msg=assistant_message), package_actions

    raise OpenRouterError("Model exceeded the maximum number of tool round trips.")


def build_handler(
    settings: Settings,
    openrouter_client: OpenRouterClient,
    packages_client: PackagesApiClient,
) -> type[BaseHTTPRequestHandler]:
    class ProxyApiRequestHandler(BaseHTTPRequestHandler):
        server_version = "ProxyApiServer/1.2"

        def do_GET(self) -> None:
            request_path = _normalize_path(self.path)
            if request_path == "/health":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "path": settings.api_path,
                        "model": settings.openrouter_model,
                        "packagesApiUrl": settings.packages_api_url,
                    },
                )
                return

            if request_path == "/logs":
                try:
                    logs_html = build_logs_html(settings.conversation_log_file)
                except OSError:
                    logger.exception("Failed to read conversation log.")
                    self._write_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": "Failed to read conversation log."},
                    )
                    return

                self._write_bytes(
                    HTTPStatus.OK,
                    logs_html,
                    content_type="text/html; charset=utf-8",
                )
                return

            if request_path == "/logs/raw":
                try:
                    log_contents = read_conversation_log(settings.conversation_log_file)
                except OSError:
                    logger.exception("Failed to read conversation log.")
                    self._write_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": "Failed to read conversation log."},
                    )
                    return

                self._write_bytes(
                    HTTPStatus.OK,
                    log_contents,
                    content_type="text/plain; charset=utf-8",
                    extra_headers={
                        "Content-Disposition": (
                            f'inline; filename="{settings.conversation_log_file.name}"'
                        )
                    },
                )
                return

            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "Endpoint not found."},
            )

        def do_POST(self) -> None:
            request_path = _normalize_path(self.path)
            if request_path != settings.api_path:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "Endpoint not found."},
                )
                return

            try:
                payload = self._read_json_payload()
                request_data = ConversationRequest.from_payload(payload)
            except InvalidConversationRequest as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except json.JSONDecodeError:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Request body must contain valid JSON."},
                )
                return
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            session_lock = _get_session_lock(request_data.session_id)
            with session_lock:
                try:
                    messages = load_session_messages(
                        log_file=settings.conversation_log_file,
                        session_id=request_data.session_id,
                        system_prompt=settings.openrouter_system_prompt,
                        max_context_messages=settings.max_context_messages,
                    )
                    messages.append({"role": "user", "content": request_data.msg})
                    response_data, package_actions = generate_operator_response(
                        settings=settings,
                        openrouter_client=openrouter_client,
                        packages_client=packages_client,
                        messages=messages,
                    )
                    append_conversation_log(
                        log_file=settings.conversation_log_file,
                        request_data=request_data,
                        response_data=response_data,
                        client_ip=self.client_address[0],
                        model_name=settings.openrouter_model,
                        package_actions=package_actions,
                    )
                except OpenRouterError as exc:
                    logger.warning(
                        "OpenRouter request failed for session {}: {}",
                        request_data.session_id,
                        exc,
                    )
                    self._write_json(
                        HTTPStatus.BAD_GATEWAY,
                        {"msg": f"Blad OpenRouter: {exc}"},
                    )
                    return
                except OSError:
                    logger.exception("Failed to access conversation log.")
                    self._write_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"msg": "Nie udalo sie zapisac lub odczytac historii rozmowy."},
                    )
                    return

            logger.info(
                "Handled session {} with model {}",
                request_data.session_id,
                settings.openrouter_model,
            )
            self._write_json(HTTPStatus.OK, response_data.to_api_dict())

        def log_message(self, format: str, *args: object) -> None:
            logger.info("{} - {}", self.address_string(), format % args)

        def _read_json_payload(self) -> object:
            content_length_header = self.headers.get("Content-Length")
            if not content_length_header:
                raise ValueError("Missing Content-Length header.")

            try:
                content_length = int(content_length_header)
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header.") from exc

            raw_payload = self.rfile.read(content_length)
            if not raw_payload:
                raise ValueError("Request body must not be empty.")

            return json.loads(raw_payload.decode("utf-8"))

        def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._write_bytes(
                status,
                encoded,
                content_type="application/json; charset=utf-8",
            )

        def _write_bytes(
            self,
            status: HTTPStatus,
            payload: bytes,
            *,
            content_type: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            if extra_headers:
                for header_name, header_value in extra_headers.items():
                    self.send_header(header_name, header_value)
            self.end_headers()
            self.wfile.write(payload)

    return ProxyApiRequestHandler


def main() -> None:
    configure_logging(name="proxy_api_server")
    settings = load_settings()
    openrouter_client = OpenRouterClient(settings)
    packages_client = PackagesApiClient(settings)
    handler_class = build_handler(settings, openrouter_client, packages_client)
    server = ThreadingHTTPServer((settings.api_host, settings.api_port), handler_class)
    logger.info(
        "Starting proxy API server on {}:{}{} using model {}",
        settings.api_host,
        settings.api_port,
        settings.api_path,
        settings.openrouter_model,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
