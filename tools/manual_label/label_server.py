#!/usr/bin/env python3
"""
Local labeling server for a manual labeling pack. Standard library only - an annotator
needs plain Python 3.9+, nothing else.

Each annotator only sees and saves the frames assigned to them in frames.json, and saves go
to work/<annotator>/<frame_id>.json (one file per frame). Two people labeling the same pack on
different machines therefore never write the same file, and merging is just copying the other
person's work/<name>/ folder into your pack.

Review mode shows a partner's saved labels read-only and stores your comments under
work/<you>/reviews/, so reviewing never touches the partner's files either.

Usage:
  python tools/manual_label/label_server.py --pack data/manual_v1
  python tools/manual_label/label_server.py --pack data/manual_v1 --user friend --port 8766
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.manual_label.common import (
    CLASS_NAMES, VALID_STATUSES, Pack, now_iso, read_json, validate_boxes, write_json_atomic,
)

UI_PATH = Path(__file__).resolve().parent / "label_ui.html"
_size_cache: Dict[str, Tuple[int, int]] = {}
_lock = threading.Lock()


def jpeg_size(path: Path) -> Tuple[int, int]:
    """(width, height) from the JPEG SOF marker, without an imaging library."""
    key = str(path)
    if key in _size_cache:
        return _size_cache[key]
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"{path} is not a JPEG")
        while True:
            marker = f.read(2)
            if len(marker) < 2 or marker[0] != 0xFF:
                raise ValueError(f"{path}: malformed JPEG")
            if marker[1] in (0xD8, 0x01) or 0xD0 <= marker[1] <= 0xD7:
                continue
            (seg_len,) = struct.unpack(">H", f.read(2))
            if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">xHH", f.read(5))
                _size_cache[key] = (w, h)
                return w, h
            f.seek(seg_len - 2, 1)


class Handler(BaseHTTPRequestHandler):
    pack: Pack  # set on the class before serving

    def log_message(self, fmt, *args):  # keep the console quiet except for errors
        if args and str(args[1])[0] in "45":
            sys.stderr.write("[server] " + fmt % args + "\n")

    # ------------------------------------------------------------------ plumbing
    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=3600")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, code: int = 200) -> None:
        self._send(code, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def _query(self) -> Dict[str, str]:
        return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _user(self, value: Optional[str]) -> str:
        if value not in self.pack.annotators:
            raise PermissionError(f"unknown annotator '{value}'")
        return value

    # ------------------------------------------------------------------ routes
    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send(200, UI_PATH.read_bytes(), "text/html; charset=utf-8", {"Cache-Control": "no-store"})
            elif path.startswith("/img/"):
                self._image(urllib.parse.unquote(path[5:]))
            elif path == "/api/session":
                self._session()
            elif path == "/api/queue":
                self._queue(self._query())
            elif path == "/api/frame":
                self._frame(self._query())
            else:
                self._error(404, "not found")
        except PermissionError as e:
            self._error(403, str(e))
        except (KeyError, ValueError) as e:
            self._error(400, str(e))

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/save":
                self._save(body)
            elif path == "/api/review":
                self._review(body)
            else:
                self._error(404, "not found")
        except PermissionError as e:
            self._error(403, str(e))
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self._error(400, str(e))

    def _image(self, name: str) -> None:
        frame_id = Path(name).stem
        if frame_id not in self.pack.by_id:
            return self._error(404, "unknown frame")
        p = self.pack.image_path(frame_id)
        if not p.exists():
            return self._error(404, f"missing image {p.name} - did you copy the whole pack?")
        self._send(200, p.read_bytes(), "image/jpeg")

    def _session(self) -> None:
        pack = self.pack
        progress = {}
        for a in pack.annotators:
            mine = pack.frames_for(a)
            done = sum(pack.status_of(a, f["id"]) in ("done", "skip") for f in mine)
            progress[a] = {"done": done, "total": len(mine)}
        self._json({"pack_id": pack.meta["pack_id"], "annotators": pack.annotators,
                    "classes": CLASS_NAMES, "progress": progress})

    def _summary(self, owner: str, f: Dict[str, Any], viewer: str) -> Dict[str, Any]:
        rec = self.pack.load_label(owner, f["id"])
        # reviews other people left on this owner's version of the frame
        reviews = [r for r in self.pack.load_reviews_on(f["id"]) if r.get("owner") == owner and r.get("reviewer") != owner]
        mine_review = read_json(self.pack.review_path(viewer, f["id"])) if owner != viewer else None
        return {
            "id": f["id"], "split": f["split"], "camera": f["camera"], "night": f["night"],
            "order": f["order"], "owner": owner, "assignee": f["assignee"],
            "status": rec.get("status", "todo") if rec else "todo",
            "n_boxes": len(rec["boxes"]) if rec else None,
            "updated_at": rec.get("updated_at") if rec else None,
            "has_label_file": rec is not None,
            "n_reviews": len(reviews),
            "review_issue": any(r.get("verdict") == "issue" for r in reviews),
            "my_review": mine_review.get("verdict") if mine_review else None,
        }

    def _queue(self, q: Dict[str, str]) -> None:
        user = self._user(q.get("user"))
        if q.get("mode") == "review":
            items = []
            for f in self.pack.frames:
                for owner in self.pack.owners_of(f["id"]):
                    if owner != user:
                        items.append(self._summary(owner, f, user))
        else:
            items = [self._summary(user, f, user) for f in self.pack.frames_for(user)]
        items.sort(key=lambda i: i["order"])
        self._json({"frames": items})

    def _frame(self, q: Dict[str, str]) -> None:
        user = self._user(q.get("user"))
        frame_id = q["id"]
        f = self.pack.by_id[frame_id]
        owner = q.get("owner") or user
        if not self.pack.is_assigned(frame_id, owner):
            raise PermissionError(f"{frame_id} is not assigned to {owner}")
        img_w, img_h = jpeg_size(self.pack.image_path(frame_id))
        rec = self.pack.load_label(owner, frame_id)
        if q.get("prelabel") == "1" and owner == user:
            boxes, source = self.pack.load_prelabel(frame_id).get("boxes", []), "prelabel"
        elif rec is not None:
            boxes, source = rec["boxes"], "saved"
        elif owner == user:
            pre = self.pack.load_prelabel(frame_id)
            boxes, source = pre.get("boxes", []), "prelabel"
        else:
            boxes, source = [], "none"
        reviews = [r for r in self.pack.load_reviews_on(frame_id) if r.get("owner") == owner and r.get("reviewer") != user]
        my_review = read_json(self.pack.review_path(user, frame_id)) if owner != user else None
        self._json({
            "frame": f, "owner": owner, "img_w": img_w, "img_h": img_h, "source": source,
            "boxes": boxes, "status": rec.get("status", "todo") if rec else "todo",
            "notes": rec.get("notes", "") if rec else "",
            "updated_at": rec.get("updated_at") if rec else None,
            "reviews": reviews, "my_review": my_review,
        })

    def _save(self, body: Dict[str, Any]) -> None:
        user = self._user(body.get("user"))
        frame_id = body["id"]
        if not self.pack.is_assigned(frame_id, user):
            raise PermissionError(f"{frame_id} is assigned to {self.pack.by_id[frame_id]['assignee']}, not {user}")
        status = body.get("status", "in_progress")
        if status not in VALID_STATUSES:
            raise ValueError(f"bad status {status}")
        img_w, img_h = jpeg_size(self.pack.image_path(frame_id))
        boxes = validate_boxes(body.get("boxes", []), img_w, img_h)
        rec = {
            "frame_id": frame_id, "annotator": user, "status": status,
            "notes": str(body.get("notes", ""))[:2000],
            "img_w": img_w, "img_h": img_h, "updated_at": now_iso(),
            "boxes": boxes,
        }
        with _lock:
            write_json_atomic(self.pack.label_path(user, frame_id), rec)
        self._json({"ok": True, "updated_at": rec["updated_at"], "n_boxes": len(boxes)})

    def _review(self, body: Dict[str, Any]) -> None:
        user = self._user(body.get("user"))
        frame_id, owner = body["id"], body["owner"]
        if owner == user:
            raise PermissionError("you cannot review your own frame")
        if not self.pack.is_assigned(frame_id, owner):
            raise PermissionError(f"{frame_id} is not assigned to {owner}")
        verdict = body.get("verdict")
        path = self.pack.review_path(user, frame_id)
        if verdict is None:
            if path.exists():
                path.unlink()
            return self._json({"ok": True, "cleared": True})
        if verdict not in ("ok", "issue"):
            raise ValueError("verdict must be 'ok' or 'issue'")
        rec = {"frame_id": frame_id, "owner": owner, "reviewer": user, "verdict": verdict,
               "comment": str(body.get("comment", ""))[:2000], "updated_at": now_iso()}
        with _lock:
            write_json_atomic(path, rec)
        self._json({"ok": True, **rec})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--user", help="open the browser already signed in as this annotator")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    pack = Pack(args.pack)
    if args.user and args.user not in pack.annotators:
        raise SystemExit(f"--user must be one of {pack.annotators}")
    missing = [f["id"] for f in pack.frames if not pack.image_path(f["id"]).exists()]
    if missing:
        print(f"[warn] {len(missing)} images missing from {pack.root / 'images'} (e.g. {missing[0]})")
    Handler.pack = pack
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/" + (f"?user={args.user}" if args.user else "")
    print(f"Labeling pack {pack.meta['pack_id']} ({len(pack.frames)} frames, annotators: {', '.join(pack.annotators)})")
    print(f"Open {url}   (Ctrl+C to stop - every change is already saved)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
