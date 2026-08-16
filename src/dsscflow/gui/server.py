from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from importlib import resources
from io import BytesIO
import json
import socket
import threading
import uuid
import webbrowser

import pandas as pd

from dsscflow import __version__
from dsscflow.gui.logic import (
    analyse,
    bundle_summary,
    export_zip_bytes,
    load_demo_data,
    parse_csv_text,
    parse_sources_payload,
    publication_checks,
)

_ANALYSES: dict[str, tuple[object, dict[str, object]]] = {}


def _read_static(name: str) -> bytes:
    ref = resources.files("dsscflow.gui").joinpath("static", name)
    with ref.open("rb") as handle:
        return handle.read()


def _json_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _optional_csv(payload: dict, key: str) -> pd.DataFrame | None:
    value = payload.get(key)
    if not value:
        return None
    text = value.get("text") if isinstance(value, dict) else value
    if not text or not str(text).strip():
        return None
    return parse_csv_text(str(text))


def analyse_payload(payload: dict) -> tuple[object, dict[str, object]]:
    mode = payload.get("mode", "demo")
    if mode == "demo":
        demo = load_demo_data()
        transitions = demo["transitions"]
        sources = demo["sources"]
        descriptors = demo["descriptors"]
        ifct = demo["ifct"]
        tdm = demo["tdm"]
    elif mode == "upload":
        transition_item = payload.get("transitions") or {}
        text = transition_item.get("text", "") if isinstance(transition_item, dict) else str(transition_item)
        if not text.strip():
            raise ValueError("A transition CSV is required in upload mode.")
        transitions = parse_csv_text(text)
        sources = parse_sources_payload(payload.get("sources") or [])
        descriptors = _optional_csv(payload, "descriptors")
        ifct = _optional_csv(payload, "ifct")
        tdm = _optional_csv(payload, "tdm")
    else:
        raise ValueError("Unknown dataset mode.")

    widths = tuple(float(x) for x in payload.get("widths", [0.20, 0.30, 0.40]))
    active_sigma = float(payload.get("active_sigma", 0.30))
    active_source = payload.get("active_source") or None
    concentration = float(payload.get("concentration_molar", 1.0e-5))
    path_length = float(payload.get("path_length_cm", 1.0))
    selected_ids = [str(x).strip() for x in payload.get("selected_ids", []) if str(x).strip()]

    bundle = analyse(
        transitions,
        sources,
        widths=widths,
        active_sigma=active_sigma,
        active_source=active_source,
        concentration_molar=concentration,
        path_length_cm=path_length,
        descriptors=descriptors,
        ifct=ifct,
        tdm=tdm,
    )
    checks = publication_checks(bundle) if mode == "demo" else {"available": False}
    summary = bundle_summary(bundle, selected_ids=selected_ids)
    summary["version"] = __version__
    summary["mode"] = mode
    summary["checks"] = checks
    return bundle, summary


class DSSCFlowHandler(BaseHTTPRequestHandler):
    server_version = "DSSCFlowGUI/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Keep terminal output compact.
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            self._send(200, _read_static("index.html"), "text/html; charset=utf-8")
            return
        if path == "/app.js":
            self._send(200, _read_static("app.js"), "text/javascript; charset=utf-8")
            return
        if path == "/style.css":
            self._send(200, _read_static("style.css"), "text/css; charset=utf-8")
            return
        if path == "/api/health":
            self._send(200, _json_bytes({"status": "ok", "version": __version__}), "application/json")
            return
        if path == "/api/export":
            from urllib.parse import urlparse, parse_qs
            token = parse_qs(urlparse(self.path).query).get("token", [""])[0]
            if token not in _ANALYSES:
                self._send(404, _json_bytes({"error": "Unknown or expired analysis token."}), "application/json")
                return
            bundle, checks = _ANALYSES[token]
            body = export_zip_bytes(bundle, checks)
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="DSSCFlow_analysis.zip"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 50_000_000:
                raise ValueError("Request body is empty or too large.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            bundle, summary = analyse_payload(payload)
            token = uuid.uuid4().hex
            _ANALYSES[token] = (bundle, summary.get("checks", {"available": False}))
            # Bound memory in long-running sessions.
            if len(_ANALYSES) > 8:
                for old in list(_ANALYSES)[:-8]:
                    _ANALYSES.pop(old, None)
            summary["token"] = token
            self._send(200, _json_bytes(summary), "application/json")
        except Exception as exc:
            self._send(400, _json_bytes({"error": str(exc), "type": type(exc).__name__}), "application/json")


def choose_port(address: str = "127.0.0.1", preferred: int = 8765) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((address, port))
                return port
            except OSError:
                continue
    raise OSError("No available local port found for DSSCFlow GUI.")


def serve(*, address: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    port = choose_port(address, port)
    httpd = ThreadingHTTPServer((address, port), DSSCFlowHandler)
    url = f"http://{address}:{port}/"
    print(f"DSSCFlow {__version__} GUI: {url}")
    print("Press Ctrl+C to stop the local server.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> None:
    serve()
