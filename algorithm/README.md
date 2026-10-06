# Vehicle tracking

The production runner creates an independent tracker and class-voting filter for each camera.

- `byetrack.py` implements the default ByteTrack path, with high- and low-confidence association. Matched detection class/confidence is retained for lane analytics.
- `sort.py` provides the optional `--tracker sort` path and the separate hardware benchmark.
- `utils.py` contains shared association and box geometry helpers.

The runner supplies detections as `[x1, y1, x2, y2, confidence, class]` and consumes tracked boxes with ID, class and confidence. Track IDs are local to a camera; they are not cross-camera vehicle identities.

The detector's default confidence threshold is 0.10 so ByteTrack receives low-confidence candidates. Tune association and class-voting settings against labelled footage before changing them. Camera reconnect resets tracking and motion history so old identities do not cross a stream discontinuity.

Run tracker and pipeline regressions with:

```bash
python -m unittest discover -s tests -v
```
