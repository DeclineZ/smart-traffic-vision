# Manual Labeling Guide (manual_v1)

This round replaces machine-made labels with labels two people draw and check by hand. Every frame comes from our own CCTV videos in `videos/`. The goal is 300 training frames and 100 validation frames, then a check of whether a model trained on them can match or beat `models/yolo26s_thai_traffic.pt`.

Why manual: the baseline was trained on DINO-mined crops plus teacher-completed frames. Two problems came out of that. Background cars were labeled only partly, so the model learned that small cars are background (34.9% small-vehicle recall on the 42-frame diagnostic set). And the dataset compiler (`tools/compile_multiclass_dataset.py`) put pickups and songthaews under **truck**, while our taxonomy says they are **car**, so the model flips pickups between the two classes. Neither problem goes away by adding more machine labels.

Your partner only needs [MANUAL_LABELING_FRIEND.md](MANUAL_LABELING_FRIEND.md).

---

## 1. What's in the pack

`data/manual_v1/` (not in git, because `data/` is ignored; share it through Google Drive):

| Path | What |
| --- | --- |
| `frames.json` | All 406 frames: split, owner, camera, why it was picked, baseline counts |
| `images/` | Full-resolution 1920x1080 JPEGs |
| `prelabels/` | Machine boxes each frame starts from |
| `work/<name>/` | One JSON per frame you labeled. Only the owner ever writes it. |
| `work/<name>/reviews/` | Comments you left on your partner's frames |
| `_cache/` | Scan cache for re-running the selection. Don't share it. |

| Split | Frames | Per person | Picked how |
| --- | ---: | ---: | --- |
| calib | 6 | 6 (both) | Labeled by **both** people, used only to measure agreement. Not trained on. |
| val | 100 | 50 | ~11 per video, 60% random (stays representative), 40% enriched for bus/truck/three-wheeler |
| train | 300 | 150 | 85% targeted at motorcycles, tuk-tuks, trucks, buses, likely pickups, and frames where the baseline probably called a pickup a truck; 15% random |

Each person's queue is calib, then val, then train, shuffled inside each group, so a half-finished queue still covers every camera.

**Leakage guards.** The baseline was trained only on the first ~206k frames of each video, so the first 260k frames are skipped entirely. Frames are also kept ≥9,000 frames away from every frame in `data/multiclass_dataset` and `data/eval_snapshot_v1` (some of those come from late in the videos). Each video is cut into 10 time blocks: blocks 2 and 7 are val-only, the rest train-only, with a 9,000-frame buffer between them. Train and val never show the same moment.

**Pre-labels** merge three passes: the baseline at 640, the baseline at 1280 (catches some smaller cars), and COCO `yolo26x` at 1280 (good at parked and background cars). A box is drawn **dashed with a ?** when the passes disagree in a way that matters (for example, baseline says truck but COCO says car, which is the classic pickup error) or when confidence is low.

---

## 2. Step by step

### Step 0: put the code on the branch (once)

The tooling lives in `tools/manual_label/` and is not committed yet:

```bash
git add tools/manual_label tests/test_manual_label.py docs/MANUAL_LABELING_GUIDE.md docs/MANUAL_LABELING_FRIEND.md README.md
```

```bash
git commit -m "[feat] manual labeling workflow: frame selection, labeler UI, merge, train, evaluate"
```

```bash
git push
```

### Step 1: share the pack

`data/manual_v1_pack.zip` (~290 MB: frames.json, images, prelabels) is already built. If you ever rebuild the pack, recreate it with:

```bash
python tools/manual_label/sync.py export-pack --pack data/manual_v1
```

Upload it to a shared Google Drive folder and send your partner the link plus [MANUAL_LABELING_FRIEND.md](MANUAL_LABELING_FRIEND.md).

### Step 2: calibration round (do this first, ~1 hour)

Both of you label the same 6 calibration frames, **without looking at each other's work**. Then your partner sends their `work/friend/` folder (see section 3) and you run:

```bash
python tools/manual_label/merge.py agreement --pack data/manual_v1
```

This lists, per frame, how many vehicles both of you found, how many only one of you found, and every class disagreement (for example `car->truck=2`). Open **Review partner** in the labeler and look at the differences together. The usual culprits are tiny background cars, parked cars, pickups vs trucks, and how tight boxes are. Agree on the rule, write it down, then carry on. Aim for >90% "found by both" and >95% class agreement. If two people label inconsistently, the model learns noise, and that is exactly what we're trying to remove.

### Step 3: label

```bash
python tools/manual_label/label_server.py --pack data/manual_v1 --user thiramet
```

The browser opens at `http://127.0.0.1:8765/`. Everything autosaves. The rules and all shortcuts are behind the **? Help** button. The most important ones:

| Do | How |
| --- | --- |
| Draw a box | drag on the image with the current class |
| Class | `1` car, `2` motorcycle, `3` bus, `4` truck, `5` three_wheeler (sets the class of the selected box too) |
| Select / cycle overlapping boxes | click / click again |
| Jump to the next `?` box, confirm it's right | `N`, `C` |
| Delete, undo, redo | `Del`, `Ctrl+Z`, `Ctrl+Y` |
| Zoom, pan, fit | wheel, right-drag or `Space`+drag, `F` |
| Peek under the boxes | hold `H` |
| Brighten a night frame | `B` |
| Frame done, open the next one | `Ctrl+Enter` |

Expect roughly 3–6 minutes a frame once you're used to it (dense frames have 30+ vehicles, many of them small), so about 10–20 hours each for 206 frames. Do val first: the queue already puts it first.

### Step 4: sync while you go

Exchange `work/` folders every session or two (section 3). `status` shows where everyone is, open "needs fixes" reviews, and frames marked done that still have `?` boxes:

```bash
python tools/manual_label/merge.py status --pack data/manual_v1
```

### Step 5: build, train, compare

When `status` shows everything done:

```bash
python tools/manual_label/merge.py build --pack data/manual_v1 --out data/manual_v1_dataset
```

```bash
python tools/manual_label/train.py --data data/manual_v1_dataset/data.yaml
```

`train.py` trains two candidates and then writes `runs/manual_eval/manual_v1_report.md`, comparing them with the baseline on the new 100-frame val set:

- `manual_v1_baseline`: fine-tunes the deployed baseline on the clean labels (80 epochs, SGD lr 0.002). With only 300 frames, this is the one most likely to win.
- `manual_v1_coco`: trains from COCO `yolo26s.pt` on the clean labels only (150 epochs). It answers whether clean data alone gets close.

Each candidate takes roughly 10–20 minutes on the RTX 5060. Use `--init baseline` to train just one. To try a model that's better on small vehicles at higher compute cost, add `--imgsz 960 --batch 8`.

To evaluate any checkpoints yourself, including on the old 42-frame diagnostic set:

```bash
python tools/manual_label/evaluate.py --data data/manual_v1_dataset/data.yaml --models models/yolo26s_thai_traffic.pt runs/train/manual_v1_baseline/weights/best.pt
```

```bash
python tools/manual_label/evaluate.py --data data/eval_snapshot_v1/dataset.yaml --models models/yolo26s_thai_traffic.pt runs/train/manual_v1_baseline/weights/best.pt
```

You can build and train early from whatever is done (`build --allow-incomplete`) to see the trend, but treat those numbers as provisional until val is complete.

---

## 3. Syncing with your partner (Google Drive)

The rule that keeps this conflict-free: **`work/<name>/` is only ever written by `<name>`.**

- **Partner to you.** They run `sync.py send` and upload `data/manual_v1_work_friend.zip`. You download it and run `receive`, which replaces your copy of `work/friend/` and touches nothing of yours.
- **You to partner** (so they can review your frames). Same thing the other way round with `--user thiramet`.
- `receive` refuses a zip that is older than the folder you already have. That blocks the one real accident: someone's own folder being sent back to them and wiping newer work. Never edit files inside someone else's folder.
- Reviews you write go in *your* folder (`work/thiramet/reviews/`), so they reach your partner with your next zip. Frames with a "needs fixes" review show a red **fix** badge in the owner's list (filter **Flagged**).

```bash
python tools/manual_label/sync.py send --pack data/manual_v1 --user thiramet
```

```bash
python tools/manual_label/sync.py receive path/to/manual_v1_work_friend.zip
```

---

## 4. How to read the result

The new val set is fully hand-labeled, so it finally gives a recall number you can trust. Expect the **baseline to score much lower here than on the old 130-frame val set**, because the old val labels were missing the same small cars the model misses. Look at:

1. **Recall (any class) and small-object recall.** Small background vehicles were the main complaint.
2. **GT car predicted truck.** That's the pickup problem. It should drop a lot, since the old dataset literally taught pickup = truck.
3. **Per-camera rows.** `cam45_northeast` was never in the baseline's training data, but the new models do train on it, so look at the other cameras too before believing the overall delta.
4. **best.pt vs last.pt.** `best.pt` was picked using this same val set, which makes it slightly optimistic. If `last.pt` is close, prefer it.

The old 42-frame diagnostic set (`data/eval_snapshot_v1`) is a second, independent-ish check, as long as you remember its lineage caveats.

Promote a model by copying it to `models/` under a new name only once it wins on both.

## 5. Next rounds

If 300 frames gets close, more frames is the cheapest way to improve. Pre-label the next batch with the *new* model (it will make fewer mistakes, so less fixing), and keep the same val set:

```bash
python tools/manual_label/select_frames.py --out data/manual_v2 --exclude-packs data/manual_v1 --val 0 --calibration 0 --train 300 --baseline runs/train/manual_v1_baseline/weights/best.pt
```

```bash
python tools/manual_label/merge.py build --pack data/manual_v1 data/manual_v2 --out data/manual_v2_dataset
```

`select_frames.py --help` lists every selection knob (`--min-gap`, `--train-cap`, `--general-frac`, `--start-skip`, and others). `--dry-run` shows the selection without writing anything; the scan cache makes re-runs instant.

## 6. Files

| File | Purpose |
| --- | --- |
| `tools/manual_label/select_frames.py` | scan videos, select and split frames, assign owners, extract, pre-label |
| `tools/manual_label/prelabel.py` | (re-)pre-label a pack with any model |
| `tools/manual_label/label_server.py` + `label_ui.html` | the labeler (stdlib-only Python server) |
| `tools/manual_label/merge.py` | `status`, `agreement`, `build` |
| `tools/manual_label/sync.py` | `export-pack`, `send`, `receive` zips for Google Drive |
| `tools/manual_label/train.py` | train candidates and auto-compare |
| `tools/manual_label/evaluate.py` | detailed comparison report for any checkpoints / dataset |
| `tests/test_manual_label.py` | `python -m unittest tests.test_manual_label` |
