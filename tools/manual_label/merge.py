#!/usr/bin/env python3
"""
Combine everyone's labels into a YOLO dataset, and check labeling progress / agreement.

Commands:
  status     progress per annotator and split, open review issues, unresolved "?" boxes
  agreement  compare annotators on the calibration frames (labeled by everyone)
  build      write a YOLO dataset (images/{train,val}, labels/{train,val}, data.yaml)

Before `build`, copy your partner's folder into the pack:  <pack>/work/<partner>/
Each frame has exactly one owner, so copying never overwrites anything of yours.

Usage:
  python tools/manual_label/merge.py status    --pack data/manual_v1
  python tools/manual_label/merge.py agreement --pack data/manual_v1
  python tools/manual_label/merge.py build     --pack data/manual_v1 --out data/manual_v1_dataset
  python tools/manual_label/merge.py build     --pack data/manual_v1 data/manual_v2 --out data/manual_v2_dataset
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.manual_label.common import (
    CLASS_NAMES, REPO_ROOT, Pack, iou_xyxy, now_iso, write_json_atomic, xyxy_to_yolo,
)

SMALL_AREA_640 = 32 * 32


def resolve(p: Path) -> Path:
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


# --------------------------------------------------------------------------- status


def cmd_status(packs: List[Pack]) -> None:
    for pack in packs:
        print(f"\n=== {pack.meta['pack_id']} ===")
        print(f"{'annotator':12s} {'split':6s} {'done':>5s} {'skip':>5s} {'in_prog':>7s} {'todo':>5s} {'boxes':>6s}")
        for a in pack.annotators:
            for split in ("calib", "val", "train"):
                frames = [f for f in pack.frames_for(a) if f["split"] == split]
                if not frames:
                    continue
                c, boxes = Counter(), 0
                for f in frames:
                    rec = pack.load_label(a, f["id"])
                    c[rec.get("status", "todo") if rec else "todo"] += 1
                    boxes += len(rec["boxes"]) if rec else 0
                print(f"{a:12s} {split:6s} {c['done']:5d} {c['skip']:5d} {c['in_progress']:7d} {c['todo']:5d} {boxes:6d}")
        missing_work = [a for a in pack.annotators if not pack.work_dir(a).exists()]
        if missing_work:
            print(f"(no work folder yet for: {', '.join(missing_work)} - copy it into {pack.root / 'work'})")
        issues = open_review_issues(pack)
        if issues:
            print(f"\nopen review issues ({len(issues)}): labels not edited since the review said 'needs fixes'")
            for owner, fid, rev in issues:
                print(f"  {fid} (owner {owner}, reviewer {rev['reviewer']}): {rev.get('comment', '')}")
        unresolved = [(a, f["id"], sum(1 for b in rec["boxes"] if b.get("check")))
                      for a in pack.annotators for f in pack.frames_for(a)
                      if (rec := pack.load_label(a, f["id"])) and rec.get("status") == "done"
                      and any(b.get("check") for b in rec["boxes"])]
        for a, fid, n in unresolved:
            print(f"  [warn] {fid} ({a}) is done but still has {n} '?' boxes")


def open_review_issues(pack: Pack) -> List[Tuple[str, str, Dict]]:
    out = []
    for f in pack.frames:
        for rev in pack.load_reviews_on(f["id"]):
            if rev.get("verdict") != "issue":
                continue
            rec = pack.load_label(rev["owner"], f["id"])
            if rec is None or rec.get("updated_at", "") <= rev.get("updated_at", ""):
                out.append((rev["owner"], f["id"], rev))
    return out


# --------------------------------------------------------------------------- agreement


def match_boxes(a: List[Dict], b: List[Dict], thr: float = 0.5) -> List[Tuple[int, int, float]]:
    """Greedy class-agnostic one-to-one matching by IoU."""
    pairs = sorted(((iou_xyxy(x["xyxy"], y["xyxy"]), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, out = set(), set(), []
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, iou))
    return out


def cmd_agreement(packs: List[Pack]) -> None:
    for pack in packs:
        calib = [f for f in pack.frames if f["split"] == "calib"]
        if not calib:
            print(f"{pack.meta['pack_id']}: no calibration frames")
            continue
        print(f"\n=== {pack.meta['pack_id']}: calibration agreement ===")
        anns = pack.annotators
        totals = defaultdict(Counter)
        for f in calib:
            recs = {a: pack.load_label(a, f["id"]) for a in anns}
            ready = [a for a in anns if recs[a] and recs[a].get("status") == "done"]
            if len(ready) < 2:
                print(f"  {f['id']}: waiting for {', '.join(a for a in anns if a not in ready)}")
                continue
            for i, a in enumerate(ready):
                for b in ready[i + 1:]:
                    A, B = recs[a]["boxes"], recs[b]["boxes"]
                    m = match_boxes(A, B)
                    same = sum(1 for i_, j_, _ in m if A[i_]["cls"] == B[j_]["cls"])
                    t = totals[(a, b)]
                    t.update(frames=1, a=len(A), b=len(B), matched=len(m), same=same,
                             only_a=len(A) - len(m), only_b=len(B) - len(m))
                    t["iou_sum"] += sum(iou for *_, iou in m)
                    for i_, j_, _ in m:
                        if A[i_]["cls"] != B[j_]["cls"]:
                            t[f"{CLASS_NAMES[A[i_]['cls']]}->{CLASS_NAMES[B[j_]['cls']]}"] += 1
                    print(f"  {f['id']}: {a}={len(A)} {b}={len(B)} matched={len(m)} same_class={same} "
                          f"only_{a}={len(A) - len(m)} only_{b}={len(B) - len(m)}")
        for (a, b), t in totals.items():
            if not t["frames"]:
                continue
            union = t["matched"] + t["only_a"] + t["only_b"]
            print(f"\n  {a} vs {b} over {t['frames']} frames:")
            print(f"    boxes {a}={t['a']} {b}={t['b']}, found by both={t['matched']} ({100 * t['matched'] / max(union, 1):.0f}% of all vehicles)")
            print(f"    class agreement on shared boxes={100 * t['same'] / max(t['matched'], 1):.0f}%, mean IoU={t['iou_sum'] / max(t['matched'], 1):.2f}")
            conf = {k: v for k, v in t.items() if "->" in k}
            if conf:
                print(f"    class disagreements ({a}->{b}): " + ", ".join(f"{k}={v}" for k, v in sorted(conf.items(), key=lambda kv: -kv[1])))
            print("    -> aim for >90% found-by-both and >95% class agreement; talk through the differences "
                  "in the label UI (Review partner) before labeling more.")


# --------------------------------------------------------------------------- build


def cmd_build(packs: List[Pack], out: Path, allow_incomplete: bool, link: bool) -> None:
    selected, incomplete = [], defaultdict(list)
    seen = set()
    for pack in packs:
        for f in pack.frames:
            if f["split"] not in ("train", "val"):
                continue
            if f["id"] in seen:
                raise SystemExit(f"{f['id']} appears in more than one pack")
            seen.add(f["id"])
            owner = f["assignee"]
            rec = pack.load_label(owner, f["id"])
            status = rec.get("status", "todo") if rec else "todo"
            if status == "skip":
                continue
            if status != "done":
                incomplete[owner].append(f["id"])
                continue
            selected.append((pack, f, rec))

    if incomplete:
        print("frames not marked done yet:")
        for owner, ids in incomplete.items():
            print(f"  {owner}: {len(ids)}  (e.g. {', '.join(ids[:3])})")
        if not allow_incomplete:
            raise SystemExit("refusing to build an incomplete dataset (pass --allow-incomplete to build from done frames only)")

    if out.exists():
        shutil.rmtree(out)
    stats = {s: Counter() for s in ("train", "val")}
    sizes = {s: Counter() for s in ("train", "val")}
    owners = {s: Counter() for s in ("train", "val")}
    manifest = []
    for pack, f, rec in selected:
        split = f["split"]
        img_dst = out / "images" / split / f"{f['id']}.jpg"
        lbl_dst = out / "labels" / split / f"{f['id']}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        src = pack.image_path(f["id"])
        if link:
            try:
                img_dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, img_dst)
        else:
            shutil.copy2(src, img_dst)
        w, h = rec["img_w"], rec["img_h"]
        scale = 640 / max(w, h)
        lines = []
        for b in rec["boxes"]:
            xc, yc, bw, bh = xyxy_to_yolo(b["xyxy"], w, h)
            lines.append(f"{b['cls']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            stats[split][CLASS_NAMES[b["cls"]]] += 1
            area = bw * w * scale * bh * h * scale
            sizes[split]["small" if area < SMALL_AREA_640 else "medium" if area <= 96 * 96 else "large"] += 1
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""))
        owners[split][f["assignee"]] += 1
        manifest.append({"id": f["id"], "split": split, "pack": pack.meta["pack_id"], "owner": f["assignee"],
                         "camera": f["camera"], "night": f["night"], "reason": f["reason"],
                         "n_boxes": len(lines), "label_updated_at": rec.get("updated_at")})

    yaml = (f"# Manually labeled Thai traffic dataset built {now_iso()} from "
            f"{', '.join(p.meta['pack_id'] for p in packs)}\n"
            f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n"
            + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    (out / "data.yaml").write_text(yaml)
    write_json_atomic(out / "manifest.json", {"created_at": now_iso(), "packs": [str(p.root) for p in packs],
                                              "frames": manifest})

    print(f"\nwrote {out}")
    for split in ("train", "val"):
        n = sum(1 for m in manifest if m["split"] == split)
        print(f"  {split}: {n} frames, {sum(stats[split].values())} boxes | "
              + ", ".join(f"{c}={stats[split][c]}" for c in CLASS_NAMES)
              + f" | small={sizes[split]['small']} medium={sizes[split]['medium']} large={sizes[split]['large']}"
              + f" | by {dict(owners[split])}")
    issues = [i for p in packs for i in open_review_issues(p)]
    if issues:
        print(f"  [warn] {len(issues)} frames have an open 'needs fixes' review - run `status` to see them")
    print(f"\nTrain with:\n  python tools/manual_label/train.py --data {out.relative_to(REPO_ROOT).as_posix() if out.is_relative_to(REPO_ROOT) else out}/data.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("status", "agreement", "build"):
        p = sub.add_parser(name)
        p.add_argument("--pack", nargs="+", required=True, type=Path)
        if name == "build":
            p.add_argument("--out", required=True, type=Path)
            p.add_argument("--allow-incomplete", action="store_true", help="build from the frames that are done so far")
            p.add_argument("--copy", action="store_true", help="copy images instead of hard-linking them")
    args = ap.parse_args()
    packs = [Pack(resolve(p)) for p in args.pack]
    if args.cmd == "status":
        cmd_status(packs)
    elif args.cmd == "agreement":
        cmd_agreement(packs)
    else:
        cmd_build(packs, resolve(args.out), args.allow_incomplete, link=not args.copy)


if __name__ == "__main__":
    main()
