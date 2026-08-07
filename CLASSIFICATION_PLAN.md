# Version 6.1.0 — Hierarchical Multimodal Classification

Integrates the pipeline of *When One Modality Is Not Enough: Multimodal Sex and
Life-Stage Classification of Red Deer from Aerial RGB–Thermal Video* (Markoff,
Praschl et al.) into the plugin: cross-modal track matching, DINOv3 embeddings,
three classification heads (occlusion, species, sex) aggregated per track by
quorum vote, and the box-area life-stage cue.

Plan document, deleted when the version ships (like `EXCHANGE_FORMAT_PLAN.md`
before it).

---

## 1. What the paper's pipeline actually is

Read as a sequence of things this plugin has to do, §3.2–§3.3 of the paper says:

1. **Detect and track per modality.** Already done (steps A1/A2).
2. **Register the two modalities at track level.** Fit a 2D affine
   `T: RGB → TH` from corresponding detection centres; compare every RGB track
   to every thermal track by the *median* inter-centre distance over shared
   frames; accept a pair when it shares ≥ 8 frames, `d < 28 px`, and both mean
   detection confidences ≥ 0.20; assignment is one-to-one.
3. **Crop every detection**, from both modalities for a matched pair.
4. **Embed each crop once** with frozen DINOv3 ViT-H+ → a 1280-d CLS vector per
   modality; `matched` = the 2560-d concatenation.
5. **Occlusion head, per frame.** Labels each crop `clear`/`occluded`. Only
   `clear` frames continue. *No vote* — this one stays per frame. In the paper
   this gate is mandatory; here it is **optional** (§5.2a), because it is a
   quality filter rather than a correctness requirement and its own balanced
   accuracy is only 78 %.
6. **Species head, per frame → quorum vote over the clear frames** of the track.
7. **Sex head, per frame → quorum vote over exactly those same clear frames.**
   Binary: `male` vs `female_juvenile`.
8. **Life stage from box area**, per flight, on the surviving individuals
   (Eq. 4) — appearance cannot separate a juvenile from an adult female at this
   resolution, so size does it instead.

The published heads confirm the shape:

| Repo | Files | Input | Output |
|---|---|---|---|
| `cpraschl/bambi-occlusion-classifiers` | `{non_geo,geo_1k,geo_2k}/occlusion_{rgb,thermal,matched}.pt` | `(N, 1280)` or `(N, 2560)` | `(emb (N,256), probs (N,2))`, `m.classes = ["clear","occluded"]` |
| `cpraschl/bambi-red-deer-sex-classifiers` | `{non_geo,geo_1k,geo_2k}/sex_{rgb,thermal,matched}.pt` | same | `m.classes = ["female_juvenile","male"]` |
| species | *not released yet* | same | mapped onto project species |

They are TorchScript `.pt` heads — `torch.jit.load`, call under `torch.no_grad()`,
BatchNorm already folded so no `.eval()` needed. They consume **features, not
images**: the DINOv3 backbone is ours to run.

Backbone: `facebook/dinov3-vith16plus-pretrain-lvd1689m` — **gated** on Hugging
Face, so the user requests access themselves and supplies a token. Needs
`transformers >= 4.56.0`; the CLS vector is `outputs.pooler_output`.

---

## 2. Where it lands in the UI

**Processing tab** (`bambi_dock_widget.py`, `proc_steps_layout`), between the
tracking block and the SAM3 block. SAM3 renumbers A3 → A4.

```
A1. Run Detection
    → Geo-Reference Detections / Add to QGIS / Calculate Perpendicular …
A2. Run Tracking
    → Add Tracks to QGIS / Calculate Track Perpendicular …
A3. Classification                                        ← new
    C1. Match RGB ↔ Thermal Tracks      [T→W]   ⚪ Not started
      → Add Matched Pairs to QGIS
    C2. Compute DINOv3 Embeddings       [T|W]   ⚪ Not started
    C3. Classify (occlusion → species → sex)    ⚪ Not started
    C4. Estimate Life Stage from Size           ⚪ Not started
      → Add Classified Tracks to QGIS
A4. Run SAM3 Segmentation                                 (was A3)
```

C1 is only required when any task is configured `matched`; the UI greys it out
otherwise. C2 runs per modality (its own camera combo); C3 reads whatever
embeddings exist and runs each configured task. C4 needs no models at all and
only depends on tracking + geo-referencing.

**New Configuration sub-tab "Classification"**, inserted after "Tracking":

- **Hugging Face access**
  - Token (password field, show/hide toggle) — **stored in QSettings, not in the
    project file.** A project `.qgz` gets shared; a credential must not travel
    with it. Falls back to `$HF_TOKEN`, then to the `huggingface_hub` cached
    token, so a user who already ran `hf auth login` needs to type nothing.
  - Backbone model id (default `facebook/dinov3-vith16plus-pretrain-lvd1689m`).
  - **Check access** button: resolves the gated repo with the token and reports
    "granted / not granted / no token", with a link to the model page. This is
    the single most likely thing to go wrong, so it gets its own button rather
    than surfacing as a stack trace three steps later.
  - Device (auto / cpu / cuda), batch size, fp16 toggle.

- **Source imagery / projection** — *decided: both supported.*
  - `Perspective (non_geo)` — crops from `frames_{m}/`, boxes straight from
    `detections`.
  - `Orthorectified 1k (geo_1k)` / `Orthorectified 2k (geo_2k)` — crops from the
    per-frame GeoTIFFs written by step P5, boxes obtained by mapping the
    geo-referenced corners through the GeoTIFF's affine transform.
  - The choice picks both the crop source *and* the sub-folder of the head repos,
    so the head always matches the imagery it was trained on. Selecting a geo
    variant without a GeoTIFF export present is caught up front with "run P5
    (Export Frames as GeoTIFF) first", the same prerequisite check every other
    step already does.

- **Classifier mapping** (the table)

  | Task | Modality | Model | Path / repo | Labels |
  |---|---|---|---|---|
  | Occlusion | matched ▾ | Default ▾ | *(bambi-occlusion-classifiers)* | [Map…] |
  | Species | rgb ▾ | Custom ▾ | `…/species_rgb.pt` [Browse] | [Map…] |
  | Sex | matched ▾ | *per species…* [Edit…] | | [Map…] |

  - **Modality** ∈ `thermal` / `rgb` / `matched` — per task.
  - **Model** ∈ `Default` (our released head, auto-downloaded) / `Custom` (local
    `.pt` or `repo::path`) / `Off` (skip the task).
  - **Sex expands per species.** *Edit…* opens a dialog listing the project's
    species (from `project.gpkg`) against a model choice each: `red deer →
    Default`, everything else `Off`, any of them overridable with a custom path.
    Same shape as `bambi_class_mapping_dialog.py`, so it reads as a sibling.
  - **Labels [Map…]** — see §5.3. Every task gets an editable mapping from the
    head's own class labels (read off `m.classes`) onto the project vocabulary,
    so a third-party model with different classes is a configuration change
    rather than a code change.
  - Species ships with **Default disabled and a note** ("not released yet —
    supply a custom model"). The whole code path exists; only the default weights
    are missing, so it starts working the day the repo goes up.

- **Cross-modal matching** (feeds C1)
  - Min shared frames (8), **distance gate in pixels (default 28, editable)**,
    min mean detection confidence (0.20), max frame time offset for the
    RGB↔thermal frame pairing (0.10 s).
  - Thermal-anchored crops (on): for a matched pair, size the RGB crop from the
    thermal box mapped through `T⁻¹`, anchored on the RGB centroid. **Crop-time
    only — the stored detections are never rewritten**, so geo-referencing and
    tracking stay valid. (The paper redraws the boxes; doing that here would
    invalidate two downstream stages for a cosmetic gain.)

- **Crops & voting**
  - Crop padding (%), crop size (224), letterbox vs stretch.
  - **Frame selection for species and sex** (see §5.2a):
    `Visible frames only` (default) / `All frames`.
  - Occlusion probability threshold (only meaningful for the first).
  - Quorum threshold (0.5 = majority), min clear frames to call a track
    (else the track is left `unknown` rather than guessed).
  - Unmatched tracks in `matched` mode: `skip` / `fall back to rgb head` /
    `fall back to thermal head`. The matched heads cannot take zeros for a
    missing modality, so this has to be an explicit choice.
  - "Write results into tracks and detections" (on) — see §5.4.

- **Life stage** (feeds C4)
  - Enable, MAD z-score threshold (−2.0), IQR gap factor (2.0), minimum
    individuals in the flight for the test to fire (default 4 — a robust
    z-score over three animals means nothing).

---

## 3. Data model

### 3.1 New store: cross-modal matches

Modality-independent, so it does **not** live under `bambi_t/` or `bambi_w/`.
New kind `store.MATCHES` at `<target>/matches.gpkg`, reached by a new
`store.matches_path(target_folder)` beside `project_path`.

```sql
CREATE TABLE match_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    affine      TEXT,          -- JSON [[a,b],[c,d]], [tx,ty]  RGB -> TH
    affine_rmse REAL,
    n_pairs     INTEGER,
    config_hash TEXT,
    created_at  TEXT,
    is_active   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE track_matches (
    match_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES match_runs(run_id),
    track_id_t  INTEGER NOT NULL,
    track_id_w  INTEGER NOT NULL,
    shared      INTEGER NOT NULL,   -- frames in common
    median_dist REAL    NOT NULL,   -- Eq. (2)
    conf_t      REAL,
    conf_w      REAL
);

CREATE TABLE detection_matches (      -- the per-frame pairing inside a match
    match_id      INTEGER NOT NULL REFERENCES track_matches(match_id),
    frame_t       INTEGER NOT NULL,
    frame_w       INTEGER NOT NULL,
    detection_id_t INTEGER NOT NULL,
    detection_id_w INTEGER NOT NULL,
    dist          REAL,
    PRIMARY KEY (match_id, detection_id_t)
);
```

`detection_matches` is what C2/C3 join on to build a 2560-d matched feature: it
answers "which RGB detection is this thermal detection" without redoing the
matching.

### 3.2 New stage store: classification

New kind `store.CLASSIFICATION` at `bambi_{m}/classification.gpkg`, holding the
bookkeeping and the predictions. The embeddings themselves are **files on disk**
(§3.4), not blobs in the GeoPackage.

```sql
CREATE TABLE embedding_runs (         -- the only description of a run (§3.4)
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    backbone   TEXT, dim INTEGER, crop_size INTEGER, padding REAL,
    projection TEXT,                  -- non_geo | geo_1k | geo_2k
    thermal_anchored INTEGER,
    folder     TEXT,                  -- "embeddings_t/non_geo"
    plugin_version TEXT,
    created_at TEXT, is_active INTEGER NOT NULL DEFAULT 1
);

-- Which detections this run has embedded. Membership is the only thing here
-- that is not derivable, and it is what makes a re-run incremental (§5.1).
-- The location is pure convention, resolved at read time:
--     folder = embedding_runs.folder
--     file   = poses_{m}.json images[frame].imagefile, extension -> .npz
--     array  = "det_<detection_id>"
CREATE TABLE embeddings (
    detection_id INTEGER NOT NULL,
    run_id       INTEGER NOT NULL REFERENCES embedding_runs(run_id),
    PRIMARY KEY (detection_id, run_id)
);

CREATE TABLE frame_predictions (
    detection_id INTEGER NOT NULL,
    task         TEXT NOT NULL,       -- occlusion | species | sex
    label        TEXT NOT NULL,       -- the head's own label, verbatim
    prob         REAL NOT NULL,
    modality_in  TEXT NOT NULL,       -- rgb | thermal | matched
    model        TEXT,
    PRIMARY KEY (detection_id, task)
);

CREATE TABLE track_predictions (
    track_id    INTEGER NOT NULL,
    task        TEXT NOT NULL,        -- + life_stage, which has no frame rows
    label       TEXT NOT NULL,
    votes       INTEGER NOT NULL,     -- e.g. 106
    n           INTEGER NOT NULL,     -- e.g. 115 clear frames
    fraction    REAL NOT NULL,        -- 0.92
    modality_in TEXT NOT NULL,
    model       TEXT,
    evidence    TEXT,                 -- JSON; life stage keeps area/z/gap here
    PRIMARY KEY (track_id, task)
);
```

`votes/n/fraction` is deliberate: Fig. 5 of the paper reports the male-vote
fraction per individual (`D4 106/115`), and that is exactly the number an
ecologist needs to judge a borderline call. It also lets the user re-vote with a
different quorum without re-running the backbone.

### 3.3 Stage graph (`core/stages.py`)

```python
"track_matching":  ("tracking",),          # both modalities, recorded as "tw"
"embeddings":      ("tracking",),
"classification":  ("embeddings",),
"life_stage":      ("tracking", "georeference"),
```

`STAGE_STORE_KIND["embeddings"] = store.CLASSIFICATION`. Resetting `tracking`
now cascades into all four, which is correct — new track ids invalidate every
prediction keyed on them.

`SCHEMA_VERSION` stays at 3: both new stores are new *files* carrying their own
meta, and `project.gpkg` gains no tables.

### 3.4 Embeddings on disk

Embeddings are an expensive, reusable asset — hours of GPU time that other tools
(the authors' own analysis scripts, a notebook, a future re-ID step) should be
able to read without going through the plugin. The vectors themselves are bulk
binary and belong on disk as **plain files mirroring the frames folder**:

```
frames_t/           frame_000123.jpg     frame_000124.jpg
embeddings_t/
    non_geo/        frame_000123.npz     frame_000124.npz
    geo_2k/         …
```

- **One `.npz` per frame**, named after the frame it came from (basename of the
  poses `imagefile`, extension swapped), holding one `float32(1280,)` array per
  detection in that frame under the key `det_<detection_id>`.
- **No sidecar metadata and no index file.** Everything describing the run —
  backbone, dim, crop size, padding, projection, thermal-anchored flag, plugin
  version, timestamp — lives in `embedding_runs`. The GeoPackage is the single
  source of truth; a `.json` or `.csv` beside it would be a second copy that can
  disagree with the first, which is exactly the class of bug the 6.0 store was
  built to remove.
- **The path is convention, not data.** Because the `.npz` is named after the
  frame's image, a detection's vector is fully addressed by things already known:
  its `frame` gives the file via the poses `imagefile`, and its `detection_id`
  gives the array. The store records only *whether* a detection has been
  embedded — the one fact that is not derivable, and the one the incremental
  re-run needs.
- The store is still perfectly readable from outside the plugin — a GeoPackage
  is SQLite, so `sqlite3` plus `numpy` is the whole dependency list for a script
  that wants a track's vectors. If a flat, joined table is genuinely wanted it
  becomes an **export format** alongside the existing ones in `core/exporters/`,
  written on request into an export folder, never as a shadow copy of the store.

**Why per frame rather than per detection.** One file per detection would mean
tens of thousands of ~5 KB files, which is slow to create and slower to scan on
Windows. One file per frame gives exactly the same file count as `frames_{m}/`,
mirrors it one-to-one so the correspondence is obvious in Explorer, and stays
one `np.load` per frame. If a per-detection layout is wanted instead, it is a
config switch over the same writer — say so and it goes in.

Reading stays cheap: C3 loads a frame's `.npz` once and takes every detection it
needs from it, and the frames are processed in order, so the access pattern is
sequential either way. Re-running one head still costs seconds rather than an
hour of GPU time — **which is the whole reason C2 and C3 are separate steps.**

Sizing, for expectations: 10 000 detections × 1280 float32 ≈ 49 MB per modality
per projection, spread over as many files as the flight has frames.

---

## 4. Cross-modal track matching (`core/track_matching.py`)

Headless, no torch, fully unit-testable.

**Step 1 — frame correspondence.** Thermal and RGB frames are matched by capture
timestamp. `core/labelling.py::_FrameMatcher` already does exactly this and is
already in `core`; promote it to a public `core.frame_matching` module and have
labelling import it from there (no behaviour change, one shared implementation).

**Step 2 — fit the affine `T: RGB → TH`.** The paper least-squares fits from
corresponding centres, but the correspondence is what we are trying to find. So:

1. Seed from *unambiguous* frame pairs — frames where each modality holds
   exactly one detection. Fit `(A, t)` by least squares (Eq. 1).
2. Refine: with the current `T`, pair detections within each frame by Hungarian
   assignment on `‖T(p_rgb) − p_th‖`, refit, repeat (≤ 5 iterations or until the
   RMSE stops improving).
3. If fewer than ~10 seed pairs exist, fall back to the pure scale/offset implied
   by the two frame sizes, and say so in the log — a bad affine silently
   produces zero matches, which is the worst possible failure mode here.

Report the final RMSE in the log and store it in `match_runs.affine_rmse`.

**Step 3 — track cost.** For every (RGB track *i*, thermal track *j*):
`d(i,j) = median over shared frames of ‖T(c_rgb) − c_th‖₂` (Eq. 2), plus the
shared-frame count and both mean confidences.

**Step 4 — assignment.** Build the cost matrix, set gated-out entries to a large
sentinel, run `scipy.optimize.linear_sum_assignment` (already a dependency —
`bambi_processing.py:5306` uses it for the built-in tracker), then **drop any
assignment that violates a gate**. Gates: `shared ≥ min_shared`,
`d < gate_px` (configurable, default 28), `mean conf ≥ min_conf` both sides.

Output: `track_matches` + `detection_matches` rows, and a log summary in the
shape of the paper's Table 2 — `RGB / TH / Confirmed / Unmatched`, which is
directly the diagnostic a user needs ("30 of 94 raw tracks confirmed").

**QGIS layer.** "Add Matched Pairs to QGIS" draws a line layer joining the
geo-referenced centroid of each matched thermal track to its RGB partner,
attributed with `shared`, `median_dist`, `conf_t`, `conf_w`. Cheap to build, and
it makes a bad gate obvious at a glance.

---

## 5. Classification (`core/classification.py` + `core/classification_store.py`)

### 5.1 C2 — crops and embeddings

For each detection in the active tracking run:

1. Obtain the crop, per the configured projection:
   - **non_geo** — load the frame from `frames_{m}/`, crop the stored box.
   - **geo_1k / geo_2k** — load the frame's GeoTIFF from `geotiffs_{m}/`, map the
     geo-referenced corners (`detections_geo.gx1…gy2`) through the raster's
     affine transform to pixel space, crop that. Detections with no geo row are
     skipped and counted, the same way geo-referencing already reports failures.
2. Pad, letterbox to 224×224.
   Matched pair + thermal-anchored on → size the RGB crop from the thermal box
   through `T⁻¹`, anchored on the RGB centroid.
3. Batch through `AutoImageProcessor` + `AutoModel` (DINOv3), take
   `outputs.pooler_output` → 1280-d, `float32`.
   *(Verify at implementation time that `pooler_output == last_hidden_state[:, 0]`
   for this checkpoint; if the transformers pooler adds a head, use the CLS token
   from `last_hidden_state` — the published heads were trained on the raw CLS
   token.)*
4. Write the frame's vectors to `embeddings_{m}/{projection}/<frame>.npz`, one
   array per detection under `det_<detection_id>`, and record
   `(detection_id, run_id)` (§3.4).

Incremental by construction: detections already carrying a row for the active
embedding run are skipped, so adding a few labelled boxes does not re-embed the
flight. Changing the projection starts a **new** embedding run rather than
overwriting the old one, so switching between perspective and geo does not throw
away an hour of GPU time. Cancellable, progress per batch.

**Everything downloadable lands in the existing shared model folder** —
`QgsApplication.qgisSettingsDirPath()/bambi_deps/models/`, what
`BambiProcessor._get_default_model_dir()` already returns and where the YOLO
detection weights and the ReID model already live. One folder the user can point
at, back up, or clear:

```
bambi_deps/models/
    thermal_animal_detector.pt          (existing)
    rgb_animal_detector.pt              (existing)
    classification/
        occlusion/non_geo/occlusion_matched.pt
        sex/non_geo/sex_rgb.pt
        …
    hf_cache/                           ← the DINOv3 backbone
        models--facebook--dinov3-vith16plus-pretrain-lvd1689m/
```

- **Heads** are fetched with `huggingface_hub.hf_hub_download(...,
  local_dir=…/classification/<task>/<projection>/)`, keeping the repo's own
  `{projection}/{task}_{modality}.pt` layout so a manually downloaded file drops
  into the right place and is picked up without configuration.
- **The backbone** goes to `…/models/hf_cache` by passing `cache_dir=` explicitly
  to `from_pretrained` — **not** by setting `HF_HOME`. Mutating that environment
  variable inside the QGIS process would relocate the cache for anything else in
  the session that uses Hugging Face, which is not ours to do.
- The folder is shared across projects, so the 3.3 GB backbone is paid for once,
  not once per flight. ViT-H+ is ~840 M parameters — say so in the UI *before*
  the first run, with the resolved path, rather than after.
- Both are skipped when the file is already present and non-empty, matching the
  existing `download_default_model` behaviour (including its
  delete-and-retry-on-truncation guard).

### 5.2 C3 — heads, then quorum

Strictly ordered, as the paper requires:

```
occlusion  → per frame, no vote.        writes frame_predictions   [optional]
             voting frames C_i per track  ← §5.2a decides what this is
species    → per frame over C_i → vote. writes frame_ + track_predictions
sex        → per frame over C_i → vote. (per species, keyed on the species call)
```

- **Feature assembly per task**, from the task's configured modality:
  `rgb` → the RGB detection's 1280-d row; `thermal` → likewise; `matched` →
  `concat(rgb, thermal)` via `detection_matches`, in that order (the heads were
  trained on `[RGB, thermal]`).
- **Head loading**: `torch.jit.load`, then `m.classes` for the label order —
  read from the model rather than hardcoded.
- **Sex is per species**: a track's sex head is chosen by the species the vote
  just assigned it. A species with no configured sex model is left alone.
- **Quorum**: `label = argmax over labels of count(frame calls)`, subject to
  `fraction > quorum` and `n ≥ min_clear_frames`; otherwise no call. Never
  abstain on the individual by discarding it — abstain by leaving the attribute
  `unknown`, which is what the enum already has a value for.

### 5.2a Occlusion is optional — where the voting frames come from

Occlusion is a **quality filter, not a prerequisite**. Species and sex must run
whether or not it has. Setting its Model to `Off` in the mapping table skips the
head entirely; the frame-selection setting then decides what the votes run over:

| Setting | Frames each track votes over |
|---|---|
| `Visible frames only` *(default)* | the frames known to be clear, resolved below |
| `All frames` | every frame of the track |

**Resolving "known to be clear" has three sources, in order.** This matters
because the occlusion attribute is *detection*-scoped and already exists in the
schema, so the classifier is not its only possible author:

1. **This run's `frame_predictions`**, when the occlusion head ran.
2. Otherwise the **stored `detections.attributes["occlusion"]`** — which is what
   the labelling tool writes. A user who annotated occlusion by hand gets their
   annotations honoured for free, and a hand annotation is better evidence than
   the 78 %-accurate head would have been.
3. Otherwise **every frame counts as clear**, with a single log line saying so
   ("no occlusion labels available — voting over all N frames"). Silently
   voting over occluded frames while the UI claims "visible frames only" would
   be the wrong kind of quiet.

Consequences to keep straight:

- `min_clear_frames` applies to whatever set came out of the above, so it is a
  meaningful floor in all three cases.
- **Species and sex must vote over the *same* frame set** — the paper is
  explicit that sex "reuses exactly those frames" — so the set is computed once
  per track and passed to both heads, never recomputed per task.
- The set actually used, and which of the three sources produced it, goes into
  `track_predictions.evidence`. Two runs of the same flight can otherwise differ
  for reasons nothing on disk explains.
- With `All frames` and the occlusion head `Off`, C3 needs no occlusion model at
  all — which is the configuration a user with only the sex classifier will
  reach for, and it has to work with nothing else installed.

### 5.3 The occlusion enum, and label mapping in general

**Decision: the seeded `occlusion` enum becomes `("clear", "occluded")`**, so the
project vocabulary matches the classifier instead of translating to it.
`core/store.py:104`:

```python
SEEDED_ENUMS = (
    ("sex",       ("unknown", "female", "male"),    "sex",       "track"),
    ("age",       ("unknown", "adult", "juvenile"), "age",       "track"),
    ("occlusion", ("clear", "occluded"),            "occlusion", "detection"),
)
```

Consequences, each with its handling:

- **Existing projects are untouched.** `seed_vocabulary` uses `INSERT OR IGNORE`
  and the ids are append-only (`trg_enum_value_no_renumber`), so a project
  already carrying `0=none, 1=partially, 2=fully` keeps exactly that. Only new
  projects get the two-value enum. The label mapping below is what lets one
  build work with both.
- **5.x migration still works.** `core/migration.py::_resolve_enum_value` already
  *appends* an unseen label with a warning, so migrating a 5.x flight into a
  fresh store adds `none`/`partially`/`fully` as ids 2/3/4 and loses nothing.
- **`core/labelling.py:40`**: `OCCLUSION_LEVELS` becomes `["clear", "occluded"]`.
  It is only the fallback list for projects with no 6.0 store — the labelling
  tool refills the combo from the project vocabulary when one exists — but it is
  also the source of the `occlusion="none"` defaults at four call sites in
  `bambi_labelling_tool.py`, which become `OCCLUSION_LEVELS[0]` rather than a
  second hardcoded string.
- **Test churn** (bounded, and expected): `tests/test_exporters.py` (4 assertions
  keyed on `occlusion: 1 → "partially"`), `tests/test_label_store.py`
  (2 assertions), and comments in `tests/test_labelling_core.py`, whose
  occlusion values are free-form strings in the in-memory model and so still
  pass. `tests/test_store.py` gains a case pinning the new seed *and* one
  pinning that an existing three-value enum survives re-opening.

**Label mapping is generic, not an occlusion special case.** Each task in the
mapping table gets a **[Map…]** button opening a small dialog:

```
Occlusion — bambi-occlusion-classifiers/non_geo/occlusion_matched.pt

  model class      →  project value        detections
  clear            →  clear            ▾   14 744
  occluded         →  occluded         ▾    5 787
```

The dialog lists the project's enum values (or species, for the species task) in
the combo and defaults to an exact case-insensitive name match — so with the new
seed the occlusion mapping is filled in correctly with no user action, and on an
old project or a third-party `partially`/`fully` model it is one dropdown.
Stored per `(task, model)` in the classification config JSON.

**`m.classes` is a convenience, never a requirement.** A custom or third-party
head may not carry the attribute at all, and a head that has not been downloaded
yet cannot be asked. The dialog therefore fills its left-hand column from the
first of these that works, and says which one it used:

1. **`m.classes`** on the loaded TorchScript module — the normal case for our
   own heads and anything following the same convention.
2. **A probe forward pass** with a zero feature vector of the configured
   dimension, reading `probs.shape[1]`. That yields the class *count* without
   the names, so the rows appear as `class 0`, `class 1`, … with an editable
   label field. Enough structure that the user only supplies the names.
3. **Fully manual** — an *Add class* button. The user defines the number of
   classes, their index order and their labels by hand, then maps each onto a
   project value. Needed when the model is not present locally yet, and it is
   also the escape hatch when a head's own `classes` are wrong.

Class **index order is what the mapping is keyed on**, not the label text: a
head returns `probs[i]`, and position `i` is the only thing guaranteed
meaningful. Labels are for the user's benefit and go into
`frame_predictions.label` verbatim. A mapping is therefore valid as long as
every index has a project value, whether or not anything named them.

The dialog warns rather than silently truncating when a later actual load
disagrees with the manual definition (model returns 3 classes, mapping defines
2) — a silent mismatch here would mislabel every frame with an off-by-one.

**This applies to every mapping in the feature** — occlusion, species and sex
alike. One dialog class, three uses.

The sex head keeps a **fixed, documented** mapping (`female_juvenile → female`,
`male → male`) as its default, editable like the rest. `female_juvenile` really
means "female **or** juvenile" — the tooltip, the docs and the changelog all have
to say so, because C4 is what resolves the ambiguity.

### 5.4 C4 — life stage from box area

No models, no embeddings — pure geometry, per §3.3 and Eq. 4 of the paper.

1. Take each individual's **median box area** over its track. Use the thermal
   box where a match exists (the paper's reference box, being the looser and more
   reliably complete of the two).
2. Robust z-score over the individuals *of this flight*:
   `z_i = 0.6745 · (a_i − median_j a_j) / MAD_j a_j`.
3. Call individual *i* a juvenile only when **both** hold: `z_i < −2`, **and**
   its size gap to the next-smallest individual exceeds `2 × IQR` of the flight's
   areas. The second condition is what stops the rule firing on the merely
   smallest adult of a herd.
4. Below the configured minimum number of individuals, decline to run and say so.

Two constraints the paper is emphatic about, and which the UI must repeat:

- **Never compare areas across flights.** Box tightness varies between
  recordings; the paper's own `B` juvenile lands inside `A2`'s female range
  (§4.5). The computation is therefore strictly per flight, and there is no
  global threshold to configure.
- **Prefer the orthorectified area** when a GeoTIFF export exists, because it is
  metric; fall back to perspective pixels otherwise, and record which was used in
  `track_predictions.evidence` alongside `area`, `z`, `gap_iqr`.

Writes `tracks.attributes["age"] = juvenile | adult`. Individuals that fail the
test are `adult`, not `unknown` — the test is designed to be a low-outlier flag,
and every non-outlier in a surveyed herd is an adult by construction. Anything
the sex head already called `male` is an adult regardless.

### 5.5 Applying results to the canonical store

This is the part that makes everything downstream work for free. The 6.0 schema
already has the right slots:

| Prediction | Written to |
|---|---|
| occlusion (per frame) | `detections.attributes["occlusion"]` = enum value id |
| species (per track) | `tracks.species_id` |
| sex (per track) | `tracks.attributes["sex"]` = enum value id |
| life stage (per track) | `tracks.attributes["age"]` = enum value id |

which means the labelling tool, every exporter (Darwin Core, Camtrap DP, YOLO,
MOT…), the QGIS layers and the survey analytics all see the classifications
**with no changes to any of them**.

Guards: never touch tracks belonging to the **manual** (labelling-tool) run — a
hand annotation outranks a model. Writing is gated behind the "write results into
tracks and detections" checkbox, and `frame_predictions`/`track_predictions`
retain full provenance either way, so the write is reversible.

---

## 6. Dependencies

New group in `bambi_dependency_manager.py`, "Classification (optional)":

| Package | Range in `_VERSION_RANGES` | Note |
|---|---|---|
| `transformers` | `>= 4.56.0` | DINOv3 support landed in 4.56.0 |
| `huggingface_hub` | any | gated download + token |
| `torch` | already tracked (`2.5.1`–`2.11.0`) | already installed for detection |

The existing numpy/scipy pinning in `core/dependency_ops.py::_run_pip` covers the
ABI risk transformers brings. GPU stays the existing CUDA row — worth restating
in the tab that ViT-H+ on CPU is slow enough to matter (order minutes per hundred
crops).

---

## 7. Files touched

| File | Change |
|---|---|
| `core/store.py` | `MATCHES`, `CLASSIFICATION` kinds + DDL; `matches_path()`; `SEEDED_ENUMS` occlusion → `clear`/`occluded` |
| `core/stages.py` | four new stages, deps, store kinds |
| `core/frame_matching.py` | **new** — `_FrameMatcher` promoted out of `labelling.py` |
| `core/track_matching.py` | **new** — affine fit, cost, Hungarian, gating |
| `core/classification.py` | **new** — crops (perspective + geo), backbone, heads, quorum, life stage (torch lazy) |
| `core/embedding_files.py` | **new** — the per-frame `.npz` writer and reader (§3.4) |
| `bambi_processing.py` | `_get_default_model_dir()` gains the `classification/` and `hf_cache/` sub-paths |
| `core/classification_store.py` | **new** — read/write both stores, apply-to-canonical |
| `core/config_schema.py` | ~30 `Classification/*` entries + bindings; new `json` role |
| `core/labelling.py` | `OCCLUSION_LEVELS`; import `_FrameMatcher` from `core.frame_matching` |
| `bambi_labelling_tool.py` | four `occlusion="none"` defaults → `OCCLUSION_LEVELS[0]` |
| `bambi_processing.py` | `run_track_matching`, `run_embeddings`, `run_classification`, `run_life_stage`; `ProcessingWorker` dispatch |
| `bambi_dock_widget.py` | Classification config sub-tab, A3 rows, statuses, `get_config`, QGIS layer, info text |
| `bambi_classification_model_dialog.py` | **new** — per-species model choice for sex and life stage |
| `bambi_label_mapping_dialog.py` | **new** — head class → project value mapping (§5.3) |
| `bambi_dependency_manager.py` | Classification dependency group |
| `docs/pipeline.md`, `docs/tools.md`, `docs/installation.md`, `docs/results-and-export.md` | new steps, config tab, HF token, occlusion enum change |
| `metadata.txt` | changelog for 6.1.0 |

On `core/config_schema.py`: the classifier mapping is a table, not a scalar, so
it needs a new `json` role in `ROLES` (get/set serialise the table to a JSON
string in a `str` entry). The existing precedent —
`Correction/AdditionalCorrections` and `Input/ThermalVisCurve` — keeps such
things *out* of the schema and saves them by hand, but that is precisely the
drift the schema was built to stop. A `json` role is a few lines and stays
covered by `check_schema` and `test_config_schema.py`. The HF token stays out
entirely: QSettings, not the project file.

---

## 8. Testing

**Unit (`tests/`, qgis stubbed, no torch):**

- `test_track_matching.py` — affine recovery from synthetic pairs incl. noise;
  seed fallback when no unambiguous frames; Hungarian one-to-one; every gate
  rejects independently; the paper's Table-2 counters.
- `test_classification_core.py` — crop geometry for **both** projections
  (perspective box; geo box through a raster transform); padding, letterbox,
  thermal-anchored sizing; quorum voting incl. ties, below-quorum, too-few
  frames; feature assembly order for `matched`; head class order read from the
  model, not assumed; label mapping applied.
- `test_label_mapping.py` (§5.3) — all three fallbacks: `m.classes` used when
  present; the probe recovers the class count from a stub head without the
  attribute; a fully manual mapping with no model available at all round-trips
  through the project config; the mapping is keyed on class *index*, so
  renaming a label does not re-target it; a mapping shorter than the model's
  actual output is rejected rather than truncated.
- `test_classification_frame_selection.py` (§5.2a) — each of the three sources
  selected in the right precedence; `All frames` ignores occlusion entirely;
  species and sex receive the identical frame set; the "no labels available"
  path logs and votes over everything; C3 runs end to end with the occlusion
  model `Off` and no occlusion weights present anywhere.
- `test_life_stage.py` — the paper's own numbers as the fixture: a cohort where
  the juvenile sits at 0.53× / 0.60× the median fires; a smooth size continuum
  (their `A1`, `C`) does **not**; the IQR-gap condition rejects on its own;
  below the minimum cohort size it declines.
- `test_classification_store.py` — round-trip both stores; incremental embedding
  skip; a projection change starts a new run rather than overwriting;
  apply-to-canonical writes the right enum ids; manual tracks untouched.
- `test_embedding_files.py` (§3.4) — `.npz` per frame named off the poses
  `imagefile`, including photo mode where that is the original photo name and
  the frame with no `imagefile` where the `frame_%06d` fallback applies;
  a detection resolves to its vector from `(frame, detection_id)` alone;
  the run is fully described by `embedding_runs` (no sidecar written, asserted
  by listing the folder); a rerun over a partially embedded flight appends
  without rewriting untouched frames; a membership row whose `.npz` was deleted
  by hand reconciles rather than raising ("the files win", `core/stages.py`).
- `test_model_paths.py` — heads and backbone resolve under
  `_get_default_model_dir()`; an already-present file is not re-downloaded;
  `HF_HOME` is left unmodified after a backbone load (asserted explicitly —
  this is a side effect on the whole QGIS process, so it needs a test, not a
  code comment).
- `test_store.py` — new occlusion seed; an existing three-value enum survives
  re-opening unchanged.
- `test_migration.py` — 5.x `none/partially/fully` migrating into a
  `clear/occluded` store appends rather than drops.
- `test_stages.py`, `test_config_schema.py` — new cascade, `json` role, parity.
- `test_architecture.py` — passes unchanged: `torch`/`transformers` imports in
  the new core modules must be function-local (`transformers` joins
  `HEAVY_TOP_LEVEL_PACKAGES`).
- Updated for the enum change: `test_exporters.py`, `test_label_store.py`.

Head inference is tested against a **tiny TorchScript stub** built in the test
(`nn.Linear(1280, 2)` scripted, with a `classes` attribute) — that exercises the
real `torch.jit.load` contract without a 3 GB download. Marked to skip when torch
is absent so the unit tier stays torch-free by default.

**QGIS (`tests_qgis/`):** config tab builds; mapping table saves and reloads
through a project round-trip; per-species and label-mapping dialogs populate from
`project.gpkg`; A3 rows exist and the statuses refresh.

**Integration:** the existing flight-6 fixture gains a matching run and a
life-stage run (real tracks, both modalities) — neither needs a model.
Backbone/head inference stays out of CI.

---

## 9. Out of scope for 6.1.0

**Appearance re-identification / duplicate suppression** (§3.4 of the paper).
Needs the 256-d head embeddings (which we would already be storing) plus a
geo-gated candidate search. It changes what a "track" means for every survey
analytic downstream, so it belongs in its own version.

---

## 10. Phasing

| Phase | Content | Independently shippable |
|---|---|---|
| 0 | Deps group, HF token + QSettings, **Check access** button, config tab skeleton | yes — user can verify gated access before anything else exists |
| 1 | `store.py` kinds + occlusion seed change + `stages.py` graph + both store modules, with tests and the doc/test churn | no user-visible change beyond the enum |
| 2 | `core/frame_matching.py` promotion, `core/track_matching.py`, C1 step + QGIS layer | yes — cross-modal matching is useful on its own |
| 3 | Perspective crops + DINOv3 + `embeddings.npy`, C2 step | yes |
| 4 | Geo crops (`geo_1k`/`geo_2k`) via the GeoTIFF transform | yes |
| 5 | Heads, frame predictions, quorum, label-mapping dialog, C3 step | yes |
| 6 | Life stage, C4 step | yes |
| 7 | Apply-to-canonical, per-species sex dialog, statuses/reset wiring | yes |
| 8 | Docs, **changelog in `metadata.txt`**, Qt6 check (`run_qt6_check.sh`), full test run, packaging | ship |

**The `metadata.txt` changelog is written last, in one pass.** Writing it
per-phase would describe features as they were half-built and leave the entry
contradicting itself; the version is only describable once it is whole. It has
to cover, at minimum: the new Classification tab and its steps, the gated
Hugging Face requirement, the occlusion enum change (with the note that
existing projects keep their values), and where models and embeddings are
stored.

Phases 2–6 each end with a working, testable step, so the version can be cut
early if the backbone work turns out heavier than expected.

---

## 11. Settled decisions

1. **Occlusion enum** → seeded as `clear`/`occluded`, matching the classifier;
   every task gets an editable head-class → project-value mapping, so a
   `partially`/`fully` model is a configuration change (§5.3).
2. **Distance gate** → configurable, default 28 px (§4).
3. **Life stage** → in scope, as step C4 (§5.4).
4. **Projection** → both perspective and orthorectified selectable, driving both
   the crop source and which sub-folder of the head repos is used (§2, §5.1).
5. **Occlusion is optional** → species and sex run either over visible frames
   (default) or over all frames; visibility comes from the classifier, else from
   hand annotations, else everything counts as visible (§5.2a).
6. **Embeddings are files on disk**, one `.npz` per frame mirroring
   `frames_{m}/`. Everything *about* them stays in the GeoPackage — no sidecar
   `.json` or `.csv`. A flat joined table, if wanted, is an export format, not
   a second copy of the store (§3.4).
7. **All downloads share `bambi_deps/models/`** — heads *and* the DINOv3
   backbone, alongside the existing detection weights (§5.1).
8. **The species classifier follows the same layout** as the other two heads,
   so it needs no special handling. Every mapping dialog nonetheless works
   without `m.classes`: probe the class count, or define the classes by hand
   (§5.3).

## 12. No open questions

All decisions are settled; §11 is the record. Implementation starts at Phase 0.
