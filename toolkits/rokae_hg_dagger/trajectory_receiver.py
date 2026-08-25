#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.

"""Receive robot-local rollout episodes over HTTP/TCP and store them safely."""

from __future__ import annotations

import argparse
import pathlib
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse


def make_handler(root: pathlib.Path, max_bytes: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if urlparse(self.path).path == "/health":
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_error(404)

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/episodes/") or not parsed.path.endswith(
                ".npz"
            ):
                self.send_error(404)
                return
            name = pathlib.PurePosixPath(unquote(parsed.path[len("/episodes/") :])).name
            if name != pathlib.PurePosixPath(parsed.path[len("/episodes/") :]).name:
                self.send_error(400, "invalid episode name")
                return
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0 or length > max_bytes:
                self.send_error(413, "episode too large")
                return
            target = root / name
            fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=root)
            try:
                with open(fd, "wb", closefd=True) as output:
                    remaining = length
                    while remaining:
                        block = self.rfile.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ConnectionError("connection closed during upload")
                        output.write(block)
                        remaining -= len(block)
                pathlib.Path(temporary).replace(target)
            except Exception:
                pathlib.Path(temporary).unlink(missing_ok=True)
                self.send_error(500, "upload failed")
                return
            self.send_response(201)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format, *args):
            print("trajectory-receiver:", format % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True, help="directory for uploaded episode .npz files"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-mb", type=int, default=2048)
    args = parser.parse_args()
    root = pathlib.Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(root, args.max_mb * 1024 * 1024)
    )
    print(f"Receiving episodes at http://{args.host}:{args.port}/episodes/")
    server.serve_forever()


if __name__ == "__main__":
    main()
