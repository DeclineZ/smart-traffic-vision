#!/usr/bin/env python3
"""
Zip and unzip pack contents for sharing over Google Drive (standard library only).

  export-pack  zip frames.json + images + prelabels      -> data/<pack>_pack.zip
  send         zip one annotator's work folder           -> data/<pack>_work_<user>.zip
  receive      unpack either kind of zip into data/<pack>/

`receive` for a work zip replaces work/<user>/ entirely, but first checks that it isn't
older than what you already have (which is what happens if someone's own folder gets sent
back to them). If any local label is newer, it stops and lists them; --force overrides.

Usage:
  python tools/manual_label/sync.py export-pack --pack data/manual_v1
  python tools/manual_label/sync.py send        --pack data/manual_v1 --user friend
  python tools/manual_label/sync.py receive     data/manual_v1_work_friend.zip
  python tools/manual_label/sync.py receive     ~/Downloads/manual_v1_pack.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.manual_label.common import REPO_ROOT, Pack, now_iso, read_json

MARKER = "sync.json"


def resolve(p: Path) -> Path:
    p = p.expanduser()
    return (p if p.is_absolute() else Path.cwd() / p).resolve()


def export_pack(pack: Pack, out: Path) -> None:
    files = [pack.root / "frames.json"] + sorted((pack.root / "images").glob("*.jpg")) + sorted((pack.root / "prelabels").glob("*.json"))
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(MARKER, json.dumps({"kind": "pack", "pack_id": pack.meta["pack_id"], "created_at": now_iso()}))
        for i, f in enumerate(files):
            # JPEGs are already compressed; deflating them only costs time
            z.write(f, f.relative_to(pack.root).as_posix(),
                    compress_type=zipfile.ZIP_STORED if f.suffix == ".jpg" else zipfile.ZIP_DEFLATED)
            print(f"  {i + 1}/{len(files)}", end="\r", flush=True)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(files)} files)")


def send(pack: Pack, user: str, out: Path) -> None:
    work = pack.work_dir(user)
    files = sorted(p for p in work.rglob("*.json")) if work.exists() else []
    if not files:
        raise SystemExit(f"nothing to send: {work} has no labels yet")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MARKER, json.dumps({"kind": "work", "pack_id": pack.meta["pack_id"], "user": user, "created_at": now_iso()}))
        for f in files:
            z.write(f, f.relative_to(pack.root).as_posix())
    done = sum(1 for f in files if f.parent == work and (read_json(f) or {}).get("status") in ("done", "skip"))
    print(f"wrote {out}: {len(files)} files ({done} frames done/skipped). Upload it to the shared Drive folder.")


def receive(zip_path: Path, data_dir: Path, force: bool) -> None:
    with zipfile.ZipFile(zip_path) as z:
        try:
            meta = json.loads(z.read(MARKER))
        except KeyError:
            raise SystemExit(f"{zip_path} wasn't made by sync.py (no {MARKER} inside)")
        root = data_dir / meta["pack_id"]
        names = [n for n in z.namelist() if n != MARKER and not n.endswith("/")]
        for n in names:  # never write outside the pack directory
            if Path(n).is_absolute() or ".." in Path(n).parts:
                raise SystemExit(f"refusing suspicious path in zip: {n}")

        if meta["kind"] == "pack":
            existing = read_json(root / "frames.json")
            if existing is not None and not force:
                incoming = json.loads(z.read("frames.json"))
                if [f["id"] for f in incoming["frames"]] != [f["id"] for f in existing["frames"]]:
                    raise SystemExit(f"{root} already holds a different version of this pack; use --force to replace it")
            for n in names:
                if n.startswith("images/") and (root / n).exists():
                    continue
                z.extract(n, root)
            print(f"pack ready in {root}. Start labeling with:\n"
                  f"  python tools/manual_label/label_server.py --pack {root.relative_to(Path.cwd()) if root.is_relative_to(Path.cwd()) else root}")
            return

        user = meta["user"]
        if not (root / "frames.json").exists():
            raise SystemExit(f"{root} doesn't exist yet - receive the pack zip first")
        target = Pack(root).work_dir(user)
        newer_locally = []
        for n in names:
            local = read_json(root / n)
            incoming = json.loads(z.read(n))
            # timestamps have 1 s resolution, so "same second but different content" counts as newer too
            if local and local != incoming and local.get("updated_at", "") >= incoming.get("updated_at", ""):
                newer_locally.append(n)
        incoming_set = set(names)
        dropped = [p.relative_to(root).as_posix() for p in target.rglob("*.json")] if target.exists() else []
        dropped = [d for d in dropped if d not in incoming_set]
        if (newer_locally or dropped) and not force:
            print(f"refusing: your copy of work/{user}/ has work this zip doesn't")
            for n in newer_locally[:10]:
                print(f"  newer here: {n}")
            for n in dropped[:10]:
                print(f"  only here:  {n}")
            raise SystemExit(f"Is this your own folder coming back to you? If you're sure the zip is the latest "
                             f"version of {user}'s work, re-run with --force.")
        if target.exists():
            shutil.rmtree(target)
        for n in names:
            z.extract(n, root)
        print(f"work/{user}/ updated from {zip_path.name} ({len(names)} files, made {meta['created_at']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export-pack")
    p.add_argument("--pack", required=True, type=Path)
    p = sub.add_parser("send")
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--user", required=True)
    p = sub.add_parser("receive")
    p.add_argument("zip", type=Path)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="where packs live (default: data/)")
    p.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.cmd == "receive":
        receive(resolve(args.zip), resolve(args.data_dir), args.force)
        return
    pack = Pack(resolve(args.pack))
    if args.cmd == "export-pack":
        export_pack(pack, pack.root.parent / f"{pack.meta['pack_id']}_pack.zip")
    else:
        if args.user not in pack.annotators:
            raise SystemExit(f"--user must be one of {pack.annotators}")
        send(pack, args.user, pack.root.parent / f"{pack.meta['pack_id']}_work_{args.user}.zip")


if __name__ == "__main__":
    main()
