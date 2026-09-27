"""Bounded data validation and a local-only health endpoint."""

from __future__ import annotations

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def csv_inventory(directory: Path) -> dict[str, int]:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise ValueError(f"no CSV fixtures found in {directory}")
    rows = 0
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                raise ValueError(f"{path} has no header")
            rows += sum(1 for row in reader if any(cell.strip() for cell in row))
    return {"csvFiles": len(files), "dataRows": rows}


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/health":
            self.send_error(404)
            return
        payload = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-data", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.check_data:
        print(json.dumps(csv_inventory(args.check_data), sort_keys=True))
        return
    ThreadingHTTPServer((args.host, args.port), HealthHandler).serve_forever()


if __name__ == "__main__":
    main()
