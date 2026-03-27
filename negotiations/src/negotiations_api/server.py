"""HTTP server entrypoint for the negotiations task tool."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from negotiations_api.catalog import format_match_output, load_catalog
from negotiations_api.config import Settings, load_settings


logger = logging.getLogger("negotiations_api")


def configure_logging() -> None:
    """Set up minimal structured logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def normalize_path(raw_path: str) -> str:
    """Normalize an HTTP path for routing."""

    return urlparse(raw_path).path.rstrip("/") or "/"


class RequestHandler(BaseHTTPRequestHandler):
    """Serve health and tool requests."""

    settings: Settings

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        path = normalize_path(self.path)
        if path == "/health":
            self.write_json(HTTPStatus.OK, {"status": "ok"})
            return
        self.write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        path = normalize_path(self.path)
        if path != self.settings.api_tool_path:
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            payload = self.read_json()
        except ValueError as error:
            self.write_json(HTTPStatus.BAD_REQUEST, {"output": str(error)})
            return

        raw_params = payload.get("params")
        if not isinstance(raw_params, str) or not raw_params.strip():
            self.write_json(
                HTTPStatus.BAD_REQUEST,
                {"output": "Podaj opis pojedynczego przedmiotu w polu params."},
            )
            return

        try:
            catalog = load_catalog(
                self.settings.data_base_url,
                str(self.settings.data_cache_dir),
                self.settings.request_timeout_seconds,
            )
            match = catalog.find_best_match(raw_params)
        except Exception as error:  # noqa: BLE001
            logger.exception("Catalog lookup failed")
            self.write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"output": f"Blad katalogu: {error}"},
            )
            return

        if match is None:
            self.write_json(
                HTTPStatus.OK,
                {"output": "Brak dopasowania. Podaj typ przedmiotu i parametry, np. 48V 150Ah."},
            )
            return

        self.write_json(HTTPStatus.OK, {"output": format_match_output(match)})

    def read_json(self) -> dict[str, Any]:
        """Read and decode a JSON request body."""

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Niepoprawny JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("JSON musi byc obiektem.")
        return payload

    def write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        """Write a JSON response."""

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_handler(settings: Settings) -> type[RequestHandler]:
    """Bind settings to the handler class."""

    class BoundRequestHandler(RequestHandler):
        pass

    BoundRequestHandler.settings = settings
    return BoundRequestHandler


def main() -> None:
    """Run the HTTP server."""

    configure_logging()
    settings = load_settings()
    server = ThreadingHTTPServer(
        (settings.api_host, settings.api_port),
        create_handler(settings),
    )
    logger.info(
        "Starting negotiations API server on %s:%s%s",
        settings.api_host,
        settings.api_port,
        settings.api_tool_path,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping negotiations API server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
