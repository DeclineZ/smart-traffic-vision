# Thai Traffic Vision Annotation & Review Guide (Batch 2)

This guide defines the human review protocol for the YOLO26s Thai traffic evaluation pack (`data/review_pack_v1`).

---

## 1. Agreed Vehicle Taxonomy

The five target detector classes are defined strictly by visual vehicle morphology (chassis scale and structure):

| Class ID | Class Name | Scope & Included Vehicles | Key Distinctions |
| :---: | :--- | :--- | :--- |
| `0` | `car` | Light vehicles on 4-wheel light chassis: sedans, hatchbacks, station wagons, metered taxis, SUVs, PPVs (Fortuner, Pajero Sport), passenger commuter vans (Toyota Commuter, HiAce), ordinary pickups (Hilux, D-Max, Ranger, Navara), pickup-based songthaews (canopy on 1-ton pickup bed), and high-cage pickups (rohkork). | Light chassis and 4-wheel passenger/utility footprint. Pickups and pickup-based songthaews belong here. |
| `1` | `motorcycle` | Two-wheel motorized transport: commuter scooters and underbones (Wave, Scoopy, Click), sport motorcycles, big bikes, and delivery bikes with courier boxes. | Sidecar tricycles (saleng) belong in `three_wheeler`. |
| `2` | `bus` | Dedicated large passenger transit vehicles: BMTA city transit buses (cream-red, blue NGV, orange), Thai Smile Bus electric buses, intercity tour coaches, double-deckers, and charter buses. | High passenger capacity (> 20 seats). |
| `3` | `truck` | Medium and heavy commercial transport: 6-wheel rigid trucks, medium delivery box trucks, 10-wheel rigid trucks, dump trucks, cement mixers, articulated 18-wheelers (tractor cab + trailer), and truck-based songthaews built on commercial 4-wheel or 6-wheel truck chassis (e.g. Isuzu Elf). | Medium and heavy commercial freight chassis. Standard 1-ton pickups do not belong here. |
| `4` | `three_wheeler` | Three-wheeled motorized transport: Bangkok and provincial tuk-tuks (auto-rickshaws) and motorized salengs (motorcycle with attached cargo bucket). | Motorized 3-wheel configuration. |

---

## 2. Separation of Detector Classes and Controller Weights

The YOLO26s object detector models visual morphology only. Class boundaries are defined by physical shape, scale, and vehicle chassis.

Downstream traffic signal controllers, queue estimators, and intersection managers assign Passenger Car Equivalent (PCE) or delay weights independently in their own software logic. Do not assign traffic weights in the detector or dataset classes. Keep visual classes clean and aligned to physical appearance.

---

## 3. Subtype Metadata Hierarchy

To preserve operational nuance without fragmenting the core 5 classes, every candidate box records an editable subtype in `annotations/annotations.json`:

- Car Subtypes (`class_id: 0`): `sedan`, `hatchback`, `suv_ppv`, `passenger_van`, `taxi`, `pickup`, `pickup_based_songthaew`, `high_cage_pickup`.
- Truck Subtypes (`class_id: 3`): `medium_truck_6w`, `heavy_truck_10w`, `articulated_trailer_18w`, `truck_based_songthaew`, `construction_truck`.
- Three-Wheeler Subtypes (`class_id: 4`): `tuktuk`, `saleng`.
- Ambiguous instances: set `is_ambiguous: true` and record the reason under `ambiguity_reason`.

---

## 4. Annotation Checklist

Apply these rules when reviewing candidate frames in `data/review_pack_v1`:

1. Check every visible vehicle across all lanes, including oncoming traffic, turning bays, and queue heads.
2. Inspect distant vehicles near the horizon (< 32² px in letterbox scale) and partly occluded motorcycles between larger vehicles.
3. Keep the light-versus-heavy boundary consistent:
   - 1-ton pickup (Hilux, D-Max) -> `0: car` (`pickup` or `high_cage_pickup`)
   - 1-ton pickup converted to songthaew -> `0: car` (`pickup_based_songthaew`)
   - Medium 6-wheel truck or truck-chassis songthaew -> `3: truck` (`medium_truck_6w` or `truck_based_songthaew`)
   - Articulated 18-wheeler -> single box enclosing cab and trailer as `3: truck` (`articulated_trailer_18w`)
4. Mark ambiguity explicitly. If night glare, motion blur, or distance prevents determining chassis size, set `is_ambiguous: true`. Never guess.
5. Historical frames and holdout frames without ground truth labels start explicitly unannotated (`unannotated`), not as verified empty backgrounds. Annotate all visible targets from scratch.
6. Once an annotator adds boxes to an unannotated frame or marks it verified, the frame leaves the unannotated state. Distinguish unannotated frames from human-verified empty background frames (`verified_empty_background`).
7. All records start unreviewed. Neither machine proposals nor generated previews count as human approval.

---

## 5. Authoritative Editable Representation & Synchronization Workflow

### Authoritative Structured Store
- **Structured Store**: `annotations/annotations.json` is the authoritative structured store. Each box maintains a permanent `instance_id` (e.g. `cam03_east_f003720_inst_000`), visual class `class_id`, `class_name`, granular `subtype`, `is_ambiguous` flag, and normalized coordinates `bbox_norm` `[xc, yc, w, h]`.
- **YOLO Label Format**: `annotations/labels/<frame_id>.txt` contains standard 5-column YOLO lines (`<class_id> <xc> <yc> <w> <h>`) for standard labeling tools (CVAT, LabelStudio, VSCode).
- **Immutable Proposals**: `proposals/<frame_id>.txt` preserves initial machine proposals. Original proposals and clean raw images are strictly immutable within a pack version.

### Instance Preservation via Spatial IoU Matching
When synchronizing changes from YOLO label files to structured metadata:
- Spatial bipartite matching (IoU $\ge 0.30$) pairs edited YOLO boxes with existing instances.
- Subtypes, ambiguity flags, and notes remain firmly attached to their corresponding physical vehicle even when lines are inserted, deleted, or reordered.
- Newly added boxes receive a fresh unique `instance_id` and provisional subtype.
- Deleted boxes are removed cleanly. Intentionally empty box lists (`boxes: []`) are strictly preserved.

### Conflict Detection Rather than Guessing
- If both YOLO label text and structured JSON are modified independently with conflicting geometries, the synchronizer raises `ConflictingEditError` rather than guessing intent.
- Annotators resolve conflicts explicitly using `--strategy=from_yolo` or `--strategy=from_json`.

### Preview Generation
Whenever annotations are synchronized, box overlay previews in `previews/<frame_id>_preview.jpg` are regenerated and validated on disk.

### Synchronization Commands
```bash
# Auto-detect direction and synchronize YOLO labels, JSON, and previews:
python tools/prepare_review_pack.py --sync

# Explicitly push YOLO label edits into annotations.json and refresh previews:
python tools/prepare_review_pack.py --sync --strategy=from_yolo

# Explicitly push annotations.json changes into YOLO labels and refresh previews:
python tools/prepare_review_pack.py --sync --strategy=from_json
```

---

## 6. Visual Annotation Editor (Batch 2B)

The interactive editor allows annotators to review, draw, resize, move, delete, and classify bounding boxes directly with the mouse on clean images, with zero manual editing of coordinates or raw JSON.

### Launching the Editor
Launch the local editor with a single command:
```bash
python tools/annotation_editor.py --pack data/review_pack_v1
```
The editor starts an HTTP server bound strictly to localhost (`http://127.0.0.1:8080/`) and automatically opens the browser interface.

### Key Capabilities & Workflow
1. **Browse All 42 Frames**: Navigate with `◀ Prev` / `Next ▶` buttons (or `[` / `]`), use the jump selector, or filter by status (`All`, `Unreviewed`, `Draft`, `Verified`, `Unannotated`). Live progress counts are displayed in the header.
2. **Zoom & Pan for Distant Vehicles**: Use the mouse wheel to zoom in smoothly centered on the cursor (up to 25x magnification) for distant or small vehicles (< 32² px in letterbox scale). Pan by holding `Space` and dragging or using middle/right click drag. Click `Fit` (or press `F`) to fit the image to the screen.
3. **Box Drawing & Editing**:
   - **Draw**: Left-click and drag on empty canvas space to draw a new bounding box.
   - **Select**: Left-click on any existing box to select it and view its handles.
   - **Move**: Click and drag inside a selected box to reposition it.
   - **Resize**: Click and drag any of the 8 resize handles on the selected box.
   - **Delete**: Press `Delete` / `Backspace` or click `Delete Box (Del)`.
   - **Undo**: Press `Ctrl+Z` or click `Undo` to revert recent changes.
4. **Taxonomy & Metadata Controls**:
   - Hotkeys `1` to `5` immediately switch the active class or convert the selected box's class (`0: car`, `1: motorcycle`, `2: bus`, `3: truck`, `4: three_wheeler`).
   - Use the Subtype dropdown to choose granular vehicle subtypes (e.g. `pickup_based_songthaew`, `passenger_van`, `truck_based_songthaew`, `saleng`, `tuktuk`).
   - Toggle `Mark Ambiguous` and provide a concrete reason (e.g. night glare, occlusion) when vehicle chassis cannot be definitively determined.
   - Record frame-level notes and frame-wide ambiguity flags in the sidebar.
5. **State Management & Transactional Saving**:
   - Click `💾 Save Draft` (`Ctrl+S`) to save work in progress. Editing any verified annotation automatically returns it to `draft`.
   - Click `✓ Mark Verified` (`Ctrl+Enter`) to explicitly verify a fully reviewed frame. Unannotated empty frames marked verified are recorded as `verified_empty_background`.
   - Unsaved changes are tracked (`UNSAVED CHANGES` badge); navigation away from dirty frames triggers a confirmation dialog to prevent accidental data loss.
   - External modifications are detected via ETag checks to prevent stale overwrites (409 Conflict).
   - Saves reuse `sync_review_pack` with pre-save backup and atomic rollback on failure.

---

## 7. Directory Layout & Artifacts

- Clean raw images (immutable): `data/review_pack_v1/images/<frame_id>.jpg`
- Box previews (color-coded overlays): `data/review_pack_v1/previews/<frame_id>_preview.jpg`
- Original machine proposals (immutable): `data/review_pack_v1/proposals/<frame_id>.txt`
- Editable YOLO labels: `data/review_pack_v1/annotations/labels/<frame_id>.txt`
- Authoritative structured annotations: `data/review_pack_v1/annotations/annotations.json`
- Traceability manifest: `data/review_pack_v1/manifest.json`
- Visual review dashboard: `data/review_pack_v1/review_index.html`
- Visual annotation editor server: `tools/annotation_editor.py`
- Visual annotation editor web UI: `tools/editor_ui.html`

---

## 8. Batch 6: Teacher-Assisted Proposal Review Protocol

To reduce manual annotation burden on incomplete frames, high-capacity teacher models (YOLO26x) generate supplemental proposals for missing background/distant vehicles.

### Workflow & Safeguards
1. **Launch Review on Pilot Pack**:
   ```bash
   python tools/annotation_editor.py --pack data/review_pack_pilot_v1
   ```
2. **Visual Differentiation**:
   - Existing human annotations appear with solid borders in class colors.
   - Candidate additions appear with emerald green dashed borders and `[+NEW]` badges.
   - Class conflicts appear with red dashed borders and `[CONFLICT]` badges.
3. **Review Actions**:
   - Select proposal and press **A** (or click `Accept Proposal`) to adopt it into the draft annotation.
   - Select proposal and press **X** or **Del** (or click `Reject Proposal`) to discard false positives.
   - Adjust proposal box handles or reclassify using keys `1`-`5` if boundary or class needs tuning.
4. **Export Safety**:
   - Only accepted proposals enter approved `annotations/labels/<frame_id>.txt` files.
   - Unaccepted/pending and rejected proposals remain strictly excluded from YOLO label exports.
   - Saving preserves draft review status (`review_status: 'draft'`) and never silently verifies frames.



