#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Serve immutable OpenPI checkpoints to a robot over HTTP/TCP.

Set ``--latest`` to a symlink (or directory) that the training job updates only
after a checkpoint is complete. The robot downloads files between episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse


def build_manifest(
    root: pathlib.Path, latest: pathlib.Path
) -> tuple[str, list[dict[str, str]]]:
    target = latest.resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"latest checkpoint is not a directory: {target}")
    version = target.name
    files: list[dict[str, str]] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(target).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative, "sha256": digest})
    return version, files


def make_handler(root: pathlib.Path, latest: pathlib.Path):
    cached_target: pathlib.Path | None = None
    cached_manifest: tuple[str, list[dict[str, str]]] | None = None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if parsed.path == "/latest.json":
                nonlocal cached_target, cached_manifest
                target = latest.resolve()
                if cached_target != target or cached_manifest is None:
                    cached_target = target
                    cached_manifest = build_manifest(root, latest)
                version, files = cached_manifest
                payload = json.dumps({"version": version, "files": files}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path.startswith("/files/"):
                target_root = latest.resolve()
                relative = pathlib.PurePosixPath(unquote(parsed.path[len("/files/") :]))
                target = (target_root / pathlib.Path(*relative.parts)).resolve()
                if target_root not in target.parents or not target.is_file():
                    self.send_error(404)
                    return
                size = target.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                with target.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        self.wfile.write(block)
                return
            self.send_error(404)

        def log_message(self, format, *args):
            print("policy-sync:", format % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latest", required=True, help="completed checkpoint directory or symlink"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    latest = pathlib.Path(args.latest).expanduser()
    root = latest.parent
    server = ThreadingHTTPServer((args.host, args.port), make_handler(root, latest))
    print(f"Serving policy {latest} on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
