# Exchange Format Rework Plan (6.0.0)

**Goal:** replace the positional text files used between pipeline stages with a
relational, self-describing, extensible store — so that user-defined attributes
survive the whole pipeline, stage outputs can be converted to standard formats
(MOT, YOLO, COCO, TRex), and the detection↔track linkage stops being
reconstructed from rounded coordinates.

**Non-goal:** changing what any stage *computes*. This is a data-plumbing
release. Geometry, projection maths and the correction model stay untouched.

---

## 1. Status quo

### 1.1 Current exchange files

| File | Format | Written by |
|---|---|---|
| `detections_{m}/detections.txt` | `frame x1 y1 x2 y2 confidence class_id` (space) | detection stage, TRex import, labelling tool |
| `georeferenced_{m}/georeferenced.txt` | `idx frame x1 y1 z1 x2 y2 z2 confidence class_id` (space) | georeference stage |
| `tracks_{m}/tracks.csv` | `frame,track_id,x1,y1,z1,x2,y2,z2,confidence,class_id[,interpolated]` | tracking stage |
| `tracks_{m}/tracks_pixel.csv` | `frame,track_id,x1,y1,x2,y2,conf,cls,interpolated` | tracker backends, `core/track_export.py` |
| `fov_{m}/fov_polygons.txt` | ragged `frame_idx num_points x1 y1 z1 …` | FoV stage |
| `labels_{m}/labels.json` | JSON, extensible | labelling tool |
| `labels_{m}/labels.csv` | fixed 11 columns | labelling tool |
| `segmentation_{m}/segmentation_{pixel,georef}.json` | JSON | SAM3 stages |
| `poses_{m}.json` | JSON (alfspy-shaped) | frame extraction |

### 1.2 The three structural defects

**(a) No identity.** A detection is a line in a file. Nothing downstream can
refer to it, so downstream code re-derives the link from the data itself.
`core/track_export.py` rounds geo coordinates to 3 decimals and uses them as a
hash key (`_key()`, line 20; `trk_lookup.setdefault`, line 96), falling back to
file ordering — which the module docstring admits: *"relies on `detections.txt`
and `georeferenced.txt` sharing the same per-frame ordering."* When
georeferencing drops a detection, positional alignment breaks and the **entire
frame** is discarded (`track_export.py:104-107`). Two animals within a
millimetre collide silently.

**(b) No extensibility.** `CustomField` values are deliberately confined to
`labels.json`; `LabelStore._export_csv` (`core/labelling.py:930`) documents why:
*"downstream consumers of this file (and of `detections.txt`) parse it
positionally."* So the one part of the system that already supports arbitrary
attributes cannot pass them to any other part.

**(c) No metadata.** No file records its CRS, units, schema version, or which
config produced it. `output_inventory.check_existing_outputs` infers stage
completion from `os.path.isdir(...) and os.listdir(...)`
(`core/output_inventory.py:73`) — a non-empty folder means "done", regardless of
whether it is stale, partial, or from a different configuration.

### 1.3 Three latent bugs — in scope for 6.0

- **Unstable class ids.** `LabelStore.species_class_ids()`
  (`core/labelling.py:821-835`) rebuilds the species→id mapping on every export:
  built-ins keep their list index, custom species are appended *alphabetically*.
  Labelling a "badger" after having exported "wolf" renumbers "wolf" — silently
  changing the meaning of already-exported detections. *Fixed by the persisted
  `species` table, §3.1 / Phase 2.*
- **`class_id 0` means two different things.** The detector emits `0` for a
  generic animal detection. The labelling tool's taxonomy has
  `SPECIES_CLASSES[0] == "unknown"` (`core/labelling.py:34`), so it reads and
  writes `0` as "unknown". Same integer, two meanings, and `detections.txt`
  carries nothing that distinguishes them **except which side of
  `DETECTIONS_MARKER` the row sits on** — which is exactly what the migration
  uses to separate them (§9). *Fixed by the three base classes, §3.1 / Phase 2.*
- **Hand-maintained cascade.** `LabelStore.replace_detections()`
  (`core/labelling.py:881-887`) `rmtree`s `tracks_{m}` and `tracks_pixel_{m}`
  because it knows they are invalidated. Nothing does the same for
  `perpendicular_tracks{m}.json`, `population_estimate.json`, exported videos or
  GeoTIFFs, which are equally stale afterwards. *Fixed by the stage cascade,
  §7 / Phase 5.*

---

## 2. Target layout

One SQLite/GeoPackage file **per stage per modality**, so `rm` remains a valid
reset and stage-completion stays file-existence-based:

```
<target_folder>/
    project.gpkg                # vocabulary + stage state (see below)
    bambi_t/                    # thermal
        detections.gpkg
        georeferenced.gpkg
        tracks.gpkg
        fov.gpkg
        labels.gpkg
        segmentation.gpkg
    bambi_w/                    # RGB — same set
        ...
    frames_t/  frames_w/        # unchanged (image data)
    poses_t.json poses_w.json   # unchanged (alfspy-shaped, consumed as-is)
    flight_route_t/ …           # unchanged for now (see §9, Phase 7)
```

Cross-stage queries use `ATTACH`:

```sql
ATTACH 'bambi_t/detections.gpkg'   AS det;
ATTACH 'bambi_t/georeferenced.gpkg' AS geo;
ATTACH 'bambi_t/tracks.gpkg'        AS trk;

SELECT d.frame, d.x1, d.y1, g.gx1, g.gy1, m.track_id
FROM det.detections d
LEFT JOIN geo.detections_geo g USING (detection_id)
LEFT JOIN trk.track_members  m USING (detection_id);
```

Foreign keys are not enforced across attached databases. That is acceptable —
what we need is stable ids, not enforcement. Referential integrity is the
cascade's job (§7).

**The vocabulary lives at project level, not in the detections stage.**
Species, enums, custom fields, detection sources and class mappings sit in
`project.gpkg` alongside the stage state, because they must outlive any single
stage: `rm bambi_t/detections.gpkg` is a supported reset (§7) and must not take
the project's species list with it. They are also modality-independent — thermal
and RGB share one vocabulary, which is what the cross-modality label copy
already assumes. *(Corrected during Phase 0; §3.1 originally placed these tables
in `detections.gpkg`, which would have destroyed the taxonomy on every
detections reset.)*

### 2.1 GeoPackage or plain SQLite?

Stage files hold **plain typed columns, no GPKG geometry**. They are still
written as valid GeoPackages using the *aspatial* profile (`gpkg_contents`
rows with `data_type='attributes'`, plus the `application_id`/`user_version`
pragmas), so QGIS's Browser panel opens them directly and the attribute table +
"toggle editing → delete rows" workflow works without any plugin code.

A small `core/gpkg.py` writes that boilerplate with **stdlib `sqlite3` only** —
no GDAL, no fiona, no new dependency, and fully unit-testable under the
QGIS-stubbed `tests/` suite. Encoding real GPKG geometry blobs is deliberately
avoided; QGIS map layers keep being constructed in code from the tables, as they
are today from the text files. A later "export spatial layers" action can emit
true spatial GeoPackages via fiona if that proves worth it.

> **Settled in Phase 0.** The aspatial layout works — every stage file opens in
> QGIS as an ordinary attribute table, and the `.sqlite` fallback is not needed.
> One non-obvious requirement came out of the spike: an **empty
> `gpkg_geometry_columns` table must exist**. The specification requires it only
> for files containing geometry, but GDAL (verified against 3.4.1, the QGIS test
> image) refuses to open a GeoPackage without it — `ogr.Open()` returns `None`
> and every layer is invalid, with no error message. Creating it empty costs
> nothing. See `tests_qgis/test_store_layers.py`.

---

## 3. Core schema

Common to every stage file:

```sql
CREATE TABLE bambi_meta (key TEXT PRIMARY KEY, value TEXT);
-- schema_version, plugin_version, modality, crs, dem_origin_x, dem_origin_y,
-- generation, created_at, created_by_stage
```

### 3.1 Detections and the project vocabulary

The `detections` table below lives in `bambi_{m}/detections.gpkg`. Everything
after it — `species`, `enums`, `enum_values`, `field_schema`,
`detection_sources`, `class_mapping` — lives in the project-level
`project.gpkg`, so that resetting the detections stage cannot destroy the
taxonomy (§2). Stage files reach it by `ATTACH`.

```sql
CREATE TABLE detections (
  detection_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  frame          INTEGER NOT NULL,
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,     -- pixel space, extracted-frame coords
  confidence     REAL,
  species_id     INTEGER NOT NULL DEFAULT 0
                 REFERENCES species(species_id),   -- resolved, 0 = animal
  source_id      INTEGER NOT NULL REFERENCES detection_sources(source_id),
  source_class   TEXT,                    -- raw class as emitted, unresolved
  label_track_id INTEGER,                 -- set for manual detections (§6)
  attributes     TEXT                     -- JSON1 object, may be NULL
);
CREATE INDEX ix_det_frame   ON detections(frame);
CREATE INDEX ix_det_source  ON detections(source_id);
CREATE INDEX ix_det_species ON detections(species_id);
CREATE UNIQUE INDEX ux_det_label ON detections(label_track_id, frame)
  WHERE label_track_id IS NOT NULL;       -- the materialisation key, §6.2
```

#### Species

```sql
CREATE TABLE species (
  species_id INTEGER PRIMARY KEY,         -- assigned once, never reordered
  name       TEXT UNIQUE NOT NULL,
  protected  INTEGER NOT NULL DEFAULT 0   -- base class: no rename, no delete
);
-- the three base classes, present in every project
INSERT INTO species VALUES ( 0, 'animal',        1);   -- the fallback
INSERT INTO species VALUES (-1, 'unknown',       1);   -- not yet determined
INSERT INTO species VALUES (-2, 'not-an-animal', 1);   -- false positive
```

**Three base classes exist in every project**, created at store initialisation
and protected against rename and deletion. Everything else is user-defined.

The sign carries the distinction, which makes it self-describing and
collision-proof:

| Range | Meaning |
|---|---|
| `species_id <= 0` | base classes — protected, fixed, present in every project |
| `species_id >= 1` | concrete species — user-managed, renameable, deletable when unused |

`animal` keeps id **0** because that is what the current detector emits and what
every unresolved or species-agnostic detection falls back to. Any code path that
cannot determine a species writes 0 rather than NULL — hence
`species_id NOT NULL DEFAULT 0`, so "no species" is never representable.

Negative ids for the other two are deliberate: legacy projects already contain
class ids `0–9` (built-in taxonomy) *and* `10+` (custom species appended by
`species_class_ids()`), so any non-negative reservation would collide with real
5.x data. Exporters remap to contiguous non-negative ids anyway (§8.1), so nothing
downstream sees a negative class.

Concrete species are seeded from `SPECIES_CLASSES[1:]` at their existing indices
(`roe deer` = 1 … `other` = 9), so ids written by 5.x keep their meaning. New
species get `max(species_id)+1` at first use and are never renumbered.

#### Where species are configured

The `species` table is **authoritative for a project**, and it is edited in
exactly one place: the **Detection configuration tab**. Species are a property of
the survey, not of whichever tool happens to be open, and confining creation to
one editor is what actually fixes latent bug 1 — `species_class_ids()` can no
longer mint an id from an arbitrary string typed into a combo box.

- The editor is a standalone dialog (`bambi_species_editor.py`) over a headless
  model (`core/species.py`), hosted by the Detection tab and openable from
  anywhere else that needs it.
- The detector's `class_mapping` (§3.1) is edited in the same tab — it is
  detector configuration, so it belongs next to the model selection.
- `core/config_schema.py` carries the species list only as a **preset** for
  seeding new projects or sharing a taxonomy between flights. The store wins for
  an existing project; the config never silently overwrites it.
- Renaming a species keeps its id (that is the point of stable ids). Deleting one
  is rejected while detections or label tracks still reference it, and always
  rejected for the base classes.

`AUTOINCREMENT` on `detections` matters: plain `INTEGER PRIMARY KEY` recycles the
rowids of deleted rows, which would let a stale reference in `track_members`
silently point at a different animal.

#### Detection sources and class mapping

Detections may come from several producers, now and in future — the current
detector, a species-agnostic detector, TRex, the labelling tool. Each is
registered, and each keeps its own class vocabulary mapped onto `species`:

```sql
CREATE TABLE detection_sources (
  source_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  kind        TEXT NOT NULL,              -- 'detector' | 'manual' | 'trex' | …
  model       TEXT,                       -- model / weights identifier
  version     TEXT,
  generation  INTEGER NOT NULL DEFAULT 1, -- bumped when this source is re-run
  config_hash TEXT,
  created_at  TEXT
);

CREATE TABLE class_mapping (
  source_id      INTEGER NOT NULL REFERENCES detection_sources(source_id),
  source_class   TEXT NOT NULL,           -- as the producer emits it
  species_id     INTEGER NOT NULL DEFAULT 0 REFERENCES species(species_id),
  PRIMARY KEY (source_id, source_class)
);
```

Rules:

- Resolution is `class_mapping` lookup, **falling back to species 0 (`animal`)**
  for anything unmapped. A species-agnostic detector simply has no mapping rows
  and everything it produces is `animal` — no special-casing anywhere.
- `detections.source_class` keeps the **raw** value alongside the resolved
  `species_id`. Correcting a mapping is then an `UPDATE … FROM class_mapping`
  over existing rows rather than a re-run of the detector. This is the reason to
  store both instead of resolving at write time and discarding the original.
- Editing a mapping invalidates nothing upstream — boxes do not move — but it
  does make species-dependent analytics stale, so it marks the analytics stages
  (§7), not georeferencing or tracking.

#### Enums

Reusable, named value sets, so a categorical field is picked from a list instead
of typed — no typos, no `"Female"` / `"female"` / `"femal"` drift:

```sql
CREATE TABLE enums (
  enum_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  name      TEXT UNIQUE NOT NULL,
  protected INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE enum_values (
  enum_id  INTEGER NOT NULL REFERENCES enums(enum_id),
  value_id INTEGER NOT NULL,              -- stable, append-only, like species
  label    TEXT NOT NULL,
  ordinal  INTEGER NOT NULL,              -- display order, freely changeable
  PRIMARY KEY (enum_id, value_id)
);
CREATE UNIQUE INDEX ux_enum_label ON enum_values(enum_id, label);
```

`value_id` follows the **same stability rules as `species_id`** (§4.1) and for
the same reason: if renaming or reordering a value could change what stored data
means, we have simply rebuilt `species_class_ids()` in another table. So
`value_id` is append-only and never renumbered, `label` is renameable, and
`ordinal` controls presentation independently of both.

#### Custom field schema

```sql
CREATE TABLE field_schema (               -- generalises labelling_fields.json
  name      TEXT NOT NULL,
  type      TEXT NOT NULL,                -- int|float|string|bool|datetime|enum
  scope     TEXT NOT NULL,                -- detection|track|frame|project
  enum_id   INTEGER REFERENCES enums(enum_id),   -- required iff type='enum'
  protected INTEGER NOT NULL DEFAULT 0,   -- seeded fields the UI renders itself
  PRIMARY KEY (name, scope)
);
```

`enum` joins the existing `FIELD_TYPES`. Everything else about `CustomField`
survives unchanged — name, scope, coercion, `_prune_to_schema` reconciliation.

### 3.2 `georeferenced.gpkg`

```sql
CREATE TABLE detections_geo (
  detection_id INTEGER PRIMARY KEY,       -- 1:1 with detections
  gx1 REAL, gy1 REAL, gz1 REAL,
  gx2 REAL, gy2 REAL, gz2 REAL
);

CREATE TABLE georef_failures (            -- replaces log-line diagnostics
  detection_id INTEGER PRIMARY KEY,
  reason       TEXT NOT NULL              -- 'no_dem_hit' | 'outside_mesh' | …
);
```

A detection that fails to project costs you that detection and records *why* —
not, as today, the pixel tracks of its entire frame. The existing
georeference-miss logging becomes queryable rows.

### 3.3 `tracks.gpkg`

```sql
CREATE TABLE track_runs (
  run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  kind        TEXT NOT NULL,        -- 'builtin' | 'boxmot' | 'trex' | 'manual'
  tracker     TEXT, version TEXT,
  generation  INTEGER NOT NULL DEFAULT 1,
  config_hash TEXT, created_at TEXT,
  is_active   INTEGER NOT NULL DEFAULT 1   -- which run the QGIS layers show
);

CREATE TABLE tracks (
  track_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id     INTEGER NOT NULL REFERENCES track_runs(run_id),
  species_id INTEGER NOT NULL DEFAULT 0 REFERENCES species(species_id),
  attributes TEXT
);
CREATE INDEX ix_tracks_run ON tracks(run_id);

CREATE TABLE track_members (
  track_id     INTEGER NOT NULL REFERENCES tracks(track_id),
  detection_id INTEGER NOT NULL,
  interpolated INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (track_id, detection_id)
);
CREATE INDEX ix_tm_det ON track_members(detection_id);
```

Tracks carry a **run**, for the same reason detections carry a source: the
built-in tracker, boxmot, TRex and the labelling tool can all produce tracks, and
re-running one of them must not disturb the others. `track_id` is unique across
runs (a single `AUTOINCREMENT` counter), so a `track_members` row is never
ambiguous even with several runs present. `is_active` picks the run the QGIS
layers and analytics use; keeping the others costs little and makes comparing
trackers possible, which the current one-file-per-stage layout cannot express at
all.

Interpolated track points have no detection of their own; they are stored as
`track_members` rows with a synthetic detection (a row whose source is the
tracker's own `detection_sources` entry, `source_class='interpolated'`) so the
join stays uniform.

**`core/track_export.py` is deleted.** `tracks_pixel.csv` becomes a view:

```sql
CREATE VIEW tracks_pixel AS
SELECT m.track_id, d.frame, d.x1, d.y1, d.x2, d.y2,
       d.confidence, d.species_id, m.interpolated
FROM track_members m JOIN detections d USING (detection_id);
```

### 3.4 `fov.gpkg`, `segmentation.gpkg`

```sql
CREATE TABLE fov_polygons (frame INTEGER PRIMARY KEY, n_points INTEGER);
CREATE TABLE fov_vertices  (frame INTEGER, seq INTEGER, x REAL, y REAL, z REAL,
                            PRIMARY KEY (frame, seq));

CREATE TABLE segments (
  segment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  detection_id INTEGER,                   -- SAM3 prompted from a detection
  frame        INTEGER NOT NULL,
  polygon_px   TEXT,                      -- JSON [[x,y],…]
  polygon_geo  TEXT,
  attributes   TEXT
);
```

---

## 4. The `detection_id` contract

1. **Minted at creation.** The detector, the TRex importer and the labelling
   tool each mint ids when they insert into `detections`. Nothing else does.
2. **Never re-derived.** No downstream stage matches on geometry, ordering or
   rounded coordinates. If a stage needs to know which detection a row belongs
   to, it stores the id.
3. **Stable within a source generation.** Generations are tracked **per
   source**, not per file (`detection_sources.generation`). Re-running the
   detector deletes and re-mints only `source_id`-matching rows and bumps that
   source's generation; manual and TRex detections keep their ids untouched.
   This is what makes several producers able to share one table — a global
   generation counter would mean any producer re-running invalidated all the
   others, which is exactly the behaviour we are trying to get rid of.
4. **Not content-hashed.** Deriving ids from `hash(frame, box, config)` was
   considered and rejected: it hides the invalidation problem and collides on
   genuinely duplicate boxes.
5. **Scope is one modality file.** Thermal and RGB have independent id spaces.
   Cross-modality links (the label copy feature) go in an explicit table, §6.4.

### 4.1 Stability guarantees

Because detections and tracks now have several possible producers, "stable" has
to be stated per operation rather than in general:

| Operation | `detection_id` | `track_id` | `species_id` |
|---|---|---|---|
| Re-run detector | re-minted for `kind='detector'` rows only | tracking invalidated | unaffected |
| Re-run tracker | unaffected | re-minted for that run only | unaffected |
| Edit a label keyframe | preserved for unchanged `(label_track_id, frame)` pairs (§6.2) | preserved | unaffected |
| Add/replace label detections | manual rows upserted; detector rows untouched (or deleted wholesale on *Replace*) | manual tracks upserted | unaffected |
| Import TRex tracklets | re-minted for `kind='trex'` rows only | new run | unaffected |
| Add a new species | unaffected | unaffected | **never renumbers existing ids** |
| Change a class mapping | unaffected | unaffected | re-resolved from `source_class`; analytics stale |
| Rename a species | unaffected | unaffected | id kept; base classes rejected |
| Delete a species | rejected if in use | — | base classes rejected always |

Three invariants hold this together, and each deserves a test in Phase 0:

- ids are minted by `AUTOINCREMENT` and **never reused**, across all sources;
- species ids are **append-only** — nothing renumbers, nothing is reordered,
  and the three base classes always exist;
- a producer may only delete or re-mint rows it owns (`source_id` / `run_id`).

`track_id` is durable *within* a run but carries no meaning across runs —
trackers renumber freely, so comparing track 7 of two runs is meaningless.
`detection_id` is the identity that persists across everything.

---

## 5. Custom fields end-to-end

`field_schema` replaces `labelling_fields.json` and is promoted from a
labelling-only concept to a pipeline-wide one. Existing scopes map directly:

| `CustomField.scope` (5.x) | 6.0 destination |
|---|---|
| `track` — one value per label track | `label_tracks.attributes` → `tracks.attributes` |
| `keyframe` — inherited by following interpolated frames | materialised into `detections.attributes` per frame |

Two new scopes become possible because the store has somewhere to put them:
`frame` (per extracted frame) and `project` (per flight).

Because the values live on `detections` rows, they reach the exporters for free:
a COCO annotation gets them as extra keys, a GeoJSON feature as properties, a
MOT sidecar as extra columns. That is the original complaint, solved
structurally rather than by threading a parameter through each stage.

`coerce_attributes` / `_prune_to_schema` (`core/labelling.py:804-815`) keep
their current job — the store never holds attributes the schema cannot describe.

### 5.1 How values are stored

`attributes` is a JSON object keyed by **field name**, so a row stays readable
without joining anything. Enum-typed fields store the **`value_id`, not the
label**:

```json
{"sex": 2, "age": 1, "collar_id": "R-114", "first_seen": "2026-08-04T06:12:00"}
```

That asymmetry is deliberate:

- **Enum values** are stored by id, so renaming `female` → `Female` (or fixing a
  spelling) rewrites nothing and cannot orphan data. Exporters resolve ids to
  labels on the way out.
- **Field names** are stored as keys, so *renaming a field* does have to rewrite
  the JSON keys of affected rows. That is an `UPDATE` issued by the schema
  editor — the same controlled reconciliation `set_custom_fields` already
  performs today (`core/labelling.py:775-785`) — and it buys human-readable
  attribute blobs, which matters a great deal when debugging by opening the
  table in QGIS.

### 5.2 Seeded enums — `sex`, `age`, `occlusion`

`SEX_CLASSES`, `AGE_CLASSES` and `OCCLUSION_LEVELS` (`core/labelling.py:38-40`)
stop being hardcoded Python lists and become **seeded enums with seeded fields**:

| Enum | Values | Field | Scope |
|---|---|---|---|
| `sex` | unknown, female, male | `sex` | track |
| `age` | unknown, adult, juvenile | `age` | track |
| `occlusion` | none, partially, fully | `occlusion` | detection |

They are ordinary rows in `enums` / `field_schema`, which makes them exactly the
worked examples a user needs when defining their own — "add `subadult` to `age`"
and "define a new enum" are then the same operation, not two different concepts.

The fields are `protected` because the labelling tool renders dedicated widgets
for them, but their **value sets are freely extensible**. Value deletion is
blocked while any row references it (same rule as species), so extending a
taxonomy can never dangle existing annotations.

This supersedes the dedicated `sex` / `age` columns on `label_tracks` and
`occlusion` on `label_keyframes`: they live in `attributes` like everything else,
so there is one mechanism rather than two. The UI is unaffected — it still shows
three fixed combo boxes, it just reads their contents from the store.

### 5.3 Where the schema is edited

Species (§3.1), enums and custom fields are all project-level vocabulary, so
they share **one editor** — a *Project Schema* dialog with three tabs
(Species | Enums | Custom fields) over a headless `core/schema_editor.py`.

It has two entry points and no duplicated logic:

- the **Detection configuration tab** hosts it (species and `class_mapping` are
  detection configuration);
- the **labelling tool's gear button** opens the same dialog, so an annotator can
  add a species, extend an enum or define a field without switching tools —
  the requirement that already applied to species, generalised.

`labelling_fields.json` import/export (`read_custom_fields` /
`write_custom_fields`) is kept and widened to carry enums alongside fields, so a
labelling setup remains shareable between projects and colleagues. Enum
`value_id`s travel with it; on import into a project that already has an enum of
that name, values are matched by label and unmatched ones appended.

---

## 6. The labelling tool

The labelling tool is the one component that **writes backwards into upstream
stages**, so it gets an explicit contract rather than being treated as a
consumer. It writes **detections and tracks**, and only ever *reads* pipeline
tracks — it must never contain tracking logic of its own (§6.6).

### 6.1 What it does today

| Action | Effect |
|---|---|
| "Add detections to project" | `export_to_detections()` — appends a marker-delimited block to `detections.txt`, preserving detector rows above it |
| "Replace detections in project" | `replace_detections()` — overwrites with label boxes only, `rmtree`s `tracks_{m}` and `tracks_pixel_{m}` |
| "Import as label track" / "Import all" | reads `tracks_pixel.csv` into new label tracks with ids from `next_track_id()` (`max+1`, an id space independent of the pipeline's) |

The marker comment (`DETECTIONS_MARKER`, `core/labelling.py:819`) exists purely
because a text file has no way to record *where a row came from*. With
`source_id` it disappears — the labelling tool owns one `detection_sources` row
(`kind='manual'`) and one `track_runs` row (`kind='manual'`), and touches only
rows carrying them:

- **Add** → upsert manual detections and manual tracks; detector rows untouched
- **Replace** → `DELETE FROM detections WHERE source_id IN (SELECT source_id
  FROM detection_sources WHERE kind <> 'manual')`, then upsert
- the hand-coded `rmtree` cascade is replaced by the stage cascade (§7)

Nothing is preserved by *file position* any more, so hand-edited rows can no
longer be silently wiped, and *Replace* stops being a structurally different
code path from *Add* — it is the same upsert plus one delete.

### 6.2 Materialisation and id stability — the subtle part

Label boxes are *generated*: `labels.gpkg` stores sparse keyframes, and
`LabelTrack.box_at()` interpolates everything between them. Every export
re-materialises the full frame range.

If export re-minted ids each time, editing one keyframe on one track would
invalidate georeferencing and tracking for the **whole flight** — and labelling
is inherently iterative, so that would make the tool unusable.

The fix is the unique index `ux_det_label (label_track_id, frame)`. Export is an
upsert keyed on that pair:

| Situation | Result |
|---|---|
| `(track, frame)` unchanged | keeps its `detection_id`; box updated in place if it moved |
| new frame in range | new `detection_id` |
| frame no longer covered (keyframe removed, stop frame added, track deleted) | row deleted |

and — crucially — **when a box moves, its `detections_geo` row is deleted**.
Georeferencing then becomes:

```sql
SELECT * FROM detections d
LEFT JOIN geo.detections_geo g USING (detection_id)
LEFT JOIN geo.georef_failures f USING (detection_id)
WHERE g.detection_id IS NULL AND f.detection_id IS NULL;
```

Re-running a stage becomes a left join instead of an all-or-nothing rebuild.
Georeferencing and segmentation are incremental this way. **Tracking is not** —
it is sequential and global, so any change to its input detections invalidates
the whole tracking stage. That asymmetry is intentional and should be stated in
the UI ("re-georeference: 12 detections; re-track: all").

### 6.3 Label tracks vs pipeline tracks

`label_tracks.track_id` stays its own id space (`next_track_id()` semantics are
unchanged), but the link to the pipeline becomes explicit:

```sql
CREATE TABLE label_tracks (
  label_track_id  INTEGER PRIMARY KEY,
  species_id      INTEGER NOT NULL DEFAULT 0,   -- 0 = animal, never NULL
  origin_track_id INTEGER,       -- set when imported from a pipeline track
  track_id        INTEGER,       -- the materialised manual track, §6.5
  attributes      TEXT
);
CREATE TABLE label_keyframes (
  label_track_id INTEGER NOT NULL,
  frame          INTEGER NOT NULL,
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  stop           INTEGER NOT NULL DEFAULT 0,
  origin_detection_id INTEGER,   -- set when imported from a pipeline detection
  attributes     TEXT,
  PRIMARY KEY (label_track_id, frame)
);
```

`sex`, `age` and `occlusion` are deliberately absent as columns — they are
seeded enum fields living in `attributes` like every other user-defined
attribute (§5.2).

`origin_track_id` / `origin_detection_id` make "this label came from pipeline
track 7" answerable, which today is lost the moment the import happens.

### 6.4 Cross-modality copy

The RGB↔thermal label copy currently matches frames by capture time and
reprojects through the DEM, producing tracks with no recorded relationship to
their source. Add, in the *destination* modality's `labels.gpkg`:

```sql
CREATE TABLE label_track_origin_xmodal (
  label_track_id     INTEGER PRIMARY KEY,
  source_modality    TEXT NOT NULL,
  source_label_track INTEGER NOT NULL
);
```

so a copied track can later be re-synced or diffed instead of silently drifting.

### 6.5 Labelling writes tracks too

A label track *is* a track: it has an identity across frames, given by the
annotator instead of computed by a tracker. So exporting labels materialises
both halves at once — manual detections into `detections`, and their grouping
into `tracks` / `track_members` under the tool's own `kind='manual'` run:

```sql
-- one row per label track, in the labelling tool's run
INSERT INTO tracks (run_id, species_id, attributes) …
-- membership follows the materialisation key of §6.2
INSERT INTO track_members (track_id, detection_id, interpolated)
SELECT :track_id, detection_id, 0
FROM detections WHERE label_track_id = :label_track_id;
```

The payoff is that **downstream consumers stop special-casing manual data**.
Geo-referencing, the geo-track layers, distance sampling, population estimation
and every exporter read `tracks`/`track_members` and get manual tracks for free,
because they are structurally identical to tracker output and merely carry a
different `run_id`. Today a manually labelled animal only reaches those consumers
if the user remembers to re-run tracking over the exported detections.

Manual tracks still depend on **georeferencing** for their world positions — the
geo coordinates come from `detections_geo` of their member detections, on the
normal path. They just never pass through the tracking stage.

### 6.6 …but implements no tracking logic

Writing tracks must not turn into owning tracking. The rules:

- The labelling tool only ever writes tracks whose membership comes **directly
  from annotator input** — the keyframes of one label track. It performs no
  association, no motion model, no re-identification, no gap closing beyond the
  existing linear keyframe interpolation.
- It never writes into another run's tracks, and never re-numbers them.
- The import direction ("Import as label track", `_load_pixel_tracks`) stays a
  **pure read** of `tracks_pixel`, copying boxes into keyframes. It creates no
  dependency and writes nothing back to the originating run.
- Anything that looks like inferring associations belongs in the tracking stage,
  and the labelling tool should call it rather than reimplement it.

### 6.7 Labelling is still not a DAG node

The tool writes `detections` and `tracks` and reads `tracks`. Modelling it as a
stage would put a cycle in the dependency graph and deadlock the cascade.

It is therefore **an editor, not a stage**: the DAG edges stay
`detections → georeferenced → tracks`, and labelling is an out-of-band mutation
of the `detections` and `tracks` nodes that triggers the normal downstream
invalidation. Its read of `tracks` creates no dependency, and its writes are
confined to rows it owns — which is precisely what makes the cycle harmless.

### 6.8 Species selection in the labelling tool

The species combo is populated **from the project's `species` table**, not from
the module-level `SPECIES_CLASSES` list, and it is a closed vocabulary: the tool
selects species, it does not create them. Free-text entry goes away, and with it
the mechanism that made class ids unstable.

Because that would otherwise force a round trip through another tab
mid-annotation, the combo gets an adjacent **"Manage species…"** button that
opens the shared *Project Schema* dialog (§5.3) on its Species tab — the same
dialog the gear button opens on Custom fields, and the same one the Detection tab
hosts. One editor, three entry points. It is modal over the labelling dialog, and
on close every combo repopulates with the current selection preserved by id.

The same applies to enum-typed fields: the `sex`, `age` and `occlusion` combos
are populated from `enum_values` ordered by `ordinal`, and adding a value is the
same dialog away.

Consequences worth handling explicitly:

- A label track whose species or enum value is renamed follows the rename
  automatically — it stores ids, not names.
- Deletion is blocked while any label track or detection references the species
  or enum value, so a stale reference cannot arise.
- Free text remains available as the `string` field type — notes, collar ids and
  similar are genuinely free-form, and forcing them into an enum would be worse.
  What changes is the **default setup**: every field seeded by BAMBI is an enum,
  so typos are impossible unless a user deliberately chooses a string field.

---

## 7. Stage state and cascade

The `stages` table, in the project-level `project.gpkg` (§2):

```sql
CREATE TABLE stages (
  stage             TEXT NOT NULL,
  modality          TEXT NOT NULL,
  state             TEXT NOT NULL,   -- pending|running|complete|failed|stale
  row_count         INTEGER,
  started_at        TEXT, finished_at TEXT,
  generation        INTEGER,
  input_fingerprint TEXT,            -- hash of config subset + upstream generations
  PRIMARY KEY (stage, modality)
);
```

Dependency graph (per modality):

```
frames ──> detections ──> georeferenced ──> tracks ──> perpendicular_tracks
   │            │                              │
   │            └──> segmentation              └──> population / density / coverage
   │
   └──> fov, alfs, geotiffs, orthomosaic          (poses + DEM, independent of detections)

labels ──(edits)──> detections, tracks         [editor, not a node — §6.7]
class_mapping ──(edits)──> analytics only      [boxes do not move — §3.1]
```

Rules:

- Completing a stage writes `state='complete'` plus counts and fingerprint.
- Invalidation is **per source / per run** where the stage supports it: a
  re-run of the detector marks tracking stale, but does not touch manual
  detections or their manual tracks.
- Invalidating a node marks every transitive dependent `state='stale'`. Stale
  results are **not** deleted automatically — they are flagged, and the UI
  offers "Reset stage" (which deletes and cascades) so a destructive action
  stays a deliberate click.
- If the `stages` table disagrees with what is on disk, **the files win**. Someone
  deleting `bambi_t/tracks.gpkg` in Explorer must still get the right answer;
  `output_inventory` reconciles on load.
- **Reset-stage must survive a locked file.** Because the plugin holds no
  handle on stage files (§11), a reset normally just deletes them. But a user
  who opened the file in the QGIS Browser holds one on Windows, and the delete
  will fail. Catch it and say so — "close the layer in the Browser and try
  again" — rather than surfacing a raw `PermissionError`. The same applies to
  `rm` in Explorer, which Windows will simply refuse; that is a platform fact to
  document, not a bug to fix.
- `input_fingerprint` answers "was this really recomputed?" directly, replacing
  folder mtime guesswork.

`output_inventory.check_existing_outputs` changes from
`isdir(folder) and listdir(folder)` to `isfile(stage_file)` plus a reconcile
against `stages` — close to a rename, which is the point of the per-stage split.

---

## 8. Consumers of the store

Everything downstream reads the store and writes throwaway products. Two
families: the exporters (§8.1) and the survey analytics tools (§8.2). Both are
strictly read-only with respect to the pipeline stages.

### 8.1 Exporters

`core/exporters/`, strictly one-directional, reading the store and writing
throwaway files. None of them is an input format.

| Target | Notes |
|---|---|
| **COCO** | reference detection export; the spec tolerates extra keys, so `attributes` maps straight onto each annotation |
| **MOT17/MOT20** | `frame,id,bb_left,bb_top,bb_w,bb_h,conf,x,y,z`; MOT20 carries class. Custom fields go to a sidecar CSV keyed by `(frame,id)` |
| **YOLO** | per-frame `.txt` with normalised `cls cx cy w h` + `data.yaml`; class map from the `species` table |
| **TRex `.npz`** | write as well as read, closing the loop with the existing importer |
| **GeoJSON / GPKG** | geo tracks, detections, FoV, transects — as today but from the store |
| *(later)* **Camtrap DP**, **Darwin Core Archive** | wildlife-survey exchange / GBIF publishing; revisit once the core lands |

**Enum resolution.** Attribute values stored as `value_id` (§5.1) are resolved to
their labels on export — no consumer should be handed an integer whose meaning
lives in a table it does not have. Where the target format supports a controlled
vocabulary (COCO attribute definitions, Camtrap DP vocabularies), the enum is
emitted as such rather than flattened to free strings.

**Class-id remapping.** Every exporter emits **contiguous, non-negative** class
ids and ships the mapping alongside (`data.yaml` for YOLO, `categories` for COCO,
a sidecar for MOT). Internal `species_id` values are never written verbatim: they
are sparse by construction (base classes are ≤ 0, user species accumulate gaps as
species are deleted), and no external consumer tolerates that. The exporters are
therefore the single place the internal id space is flattened — and the only
place a negative base class becomes visible as an ordinary category.

Each exporter also decides what to do with `not-an-animal` (-2): YOLO and COCO
drop those rows by default (they are labelled false positives, useful as hard
negatives only if the target training setup wants them), while GeoJSON and
Camtrap DP keep them. Make it an explicit per-export option, not a silent filter.

### 8.2 Survey analytics

The Survey Analytics group — density heat map, line-transect distance sampling,
coverage map, transect splitting and population estimation
(`core/population.py`, `core/transects.py`) — is the most consequential consumer,
because its outputs are scientific results rather than files. It gets explicit
treatment rather than "port the readers".

#### What changes for them

Today each tool parses `tracks.csv`, `perpendicular*.json` and
`population_tracks.csv` positionally and takes whatever it finds. In the new
model every analytic must state its **population filter** — which is currently
implicit and therefore unauditable:

```sql
-- the default selection every analytic starts from
SELECT … FROM tracks t
JOIN track_runs r USING (run_id)
WHERE r.is_active = 1              -- one tracker run, never several pooled
  AND t.species_id > -2            -- exclude 'not-an-animal'
```

Three rules, all of which are silent-bias risks if left to chance:

- **`not-an-animal` (-2) is excluded from every analytic.** A labelled false
  positive entering a density estimate biases it upward. `unknown` (-1) and
  `animal` (0) both *are* animals and both count — the distinction is
  determinacy, not presence.
- **Exactly one *tracker* run contributes.** Builtin, boxmot and TRex are
  alternative descriptions of the same animals, so pooling two would double-count
  everything. Analytics read `is_active` unless the user picks a run explicitly.
- **The manual run is pooled, not chosen between.** Label tracks are usually
  animals the detector *missed* — false negatives the annotator added — so
  excluding them undercounts. The labelling tool's run is therefore additive:
  it contributes alongside whichever tracker run is active.

  The exception is a label track created by "Import as label track", which
  copies a tracker track and refines it. Both then describe the same animal, and
  pooling would double it. `label_tracks.origin_track_id` records exactly that
  provenance, so the original is **superseded** — excluded once the label track
  has been materialised. A track drawn from scratch has no `origin_track_id`
  and supersedes nothing.

  *(Revised after Phase 5. The plan originally said manual and detector tracks
  must never be mixed, on the assumption they described the same animals. That
  is only true for imported tracks; treating every label that way silently drops
  the annotator's corrections, which is the more likely mistake. The provenance
  added in §6.3 is what makes the distinction decidable rather than a guess.)*

#### What they gain

- **Species filtering and stratification become first-class.** `species_id` now
  has a stable meaning, so "roe deer only" or "per-species density" stops being a
  fragile integer comparison.
- **Custom fields reach the analytics.** Stratifying a density estimate by `sex`
  or `age`, or filtering by a user-defined field, needs no new plumbing — the
  attributes are on the rows the analytics already read. This is the clearest
  payoff of the whole rework for the science side.
- **Perpendicular distances key on ids.** `perpendicular{m}.json` and
  `perpendicular_tracks{m}.json` become tables keyed by `detection_id` /
  `track_id` instead of implicitly ordered arrays, so a dropped detection no
  longer shifts every subsequent distance.

#### Multi-project pooling

Pooling several target folders into one estimate is where per-project id spaces
bite. The rule:

> **Across projects, join species and enum values by `label`, never by id.**

Ids are assigned per project (§3.1), so project A's species 12 and project B's
species 12 are unrelated. The pooling code resolves by name, reports any
vocabulary mismatch (a species present in one project and not the other) rather
than silently dropping it, and refuses to pool if two projects disagree on what a
shared enum value means.

Pooling itself is `ATTACH` over each project's stage files, which is materially
simpler than today's parse-and-merge across folders.

#### Consumer audit (done during Phase 4)

Every tool that reads a pipeline output was checked against the legacy file
list of §1.1. All of them still work, because each stage keeps writing its text
file — but two findings are worth carrying forward:

| Consumer | Reads | State |
|---|---|---|
| Video Creator (`core/video_export.py`) | `tracks_pixel.csv`, `tracks.csv`, `detections.txt`, `georeferenced.txt`, `fov_polygons.txt` | works; **positional alignment**, see below |
| Click tool (`core/inspection.py`) | `detections.txt`, `georeferenced.txt`, `tracks_pixel.csv` | works |
| Box projector | `georeferenced.txt` | works |
| Population / transects | `fov_polygons.txt`, `tracks.csv` | works — moves in Phase 7 (§8.2) |
| QGIS layer builders | `georeferenced.txt`, `tracks.csv`, `fov_polygons.txt`, segmentation JSON | works — moves in Phase 5 |
| Labelling tool | `detections.txt`, `tracks_pixel.csv` | reads the store when present (Phase 4) |

**`video_export.load_track_id_rows` pairs `tracks_pixel.csv` rows with
`detections.txt` rows by position** — the same assumption `track_export.py`
made, and the same one that breaks when a detection is dropped. It predates
this rework and is not made worse by it, but it is the last surviving instance
of the defect §1.2a describes. Move the Video Creator onto the store in Phase 6
alongside the exporters; a join on `detection_id` replaces the alignment
outright.

**Legacy projects need `detections.txt` adopted.** Everything downstream of
detection now resolves through the store, so a `detections.txt` produced by 5.x
— or by hand, or by an external tool — would reach geo-referencing with nothing
to work against and silently yield no `tracks_pixel.csv`.
`detection_store.adopt_legacy_detections()` ingests the detector block on the
way into geo-referencing. Only that block: rows below `DETECTIONS_MARKER`
describe label tracks whose identity the text file does not record, and
adopting them unlinked would strand manual detections the labelling tool does
not manage. A project tracked without ever passing through a 6.0 stage says so
and points at "Migrate 5.x…" rather than writing nothing.

#### What does *not* move

Density and coverage rasters stay raster outputs — only their inputs change.
`population_estimate.json` stays a JSON result document; it is a report, not an
exchange format, and nothing consumes it programmatically. Resisting the urge to
migrate these keeps Phase 7 small.

---

## 9. Migration from 5.x

A one-shot importer, offered when a 5.x target folder is opened. Reuses the
existing parsers in `core/pipeline_outputs.py` and `core/labelling.py`.

| Legacy | New |
|---|---|
| *(implicit)* | `species` seeded: base classes `0 animal`, `-1 unknown`, `-2 not-an-animal`; concrete species `1–9` from `SPECIES_CLASSES[1:]` unchanged; custom species from `labels.json` appended above 9 in exactly the alphabetical order `species_class_ids()` would produce — computed once at migration and then **frozen**, which is the last moment that mapping is ever derived rather than stored |
| *(implicit)* | `detection_sources` rows synthesised: one `detector`, one `manual`, one `trex` where the corresponding rows exist; `class_mapping` seeded identity (`source_class = species name`) |
| `detections_{m}/detections.txt` | `detections` — ids assigned in file order; rows below `DETECTIONS_MARKER` get the `manual` source, above it the `detector` source; `source_class` set from the legacy `class_id`. **`class_id 0` resolves by block**: above the marker → `animal` (0), below → `unknown` (-1), which is the only place that distinction is recoverable |
| `georeferenced_{m}/georeferenced.txt` | `detections_geo` by per-frame order; rows with `x1<0 or y1<0` (`pipeline_outputs.py:182`) become `georef_failures` instead of being dropped |
| `tracks_{m}/tracks.csv` | `tracks` + `track_members` |
| `tracks_{m}/tracks_pixel.csv` | preferred linkage source when present (it already carries the pixel↔track pairing) |
| `fov_{m}/fov_polygons.txt` | `fov_polygons` + `fov_vertices` |
| `labels_{m}/labels.json` | `label_tracks`, `label_keyframes`, `field_schema`, `species` |
| *(implicit)* | `enums` seeded from `SEX_CLASSES` / `AGE_CLASSES` / `OCCLUSION_LEVELS` at their list indices, with `sex` / `age` / `occlusion` seeded as protected fields (§5.2). Stored label strings resolve to `value_id` by label; anything unmatched is **appended as a new enum value** rather than dropped, so hand-edited or older data survives |
| `segmentation_{m}/*.json` | `segments` |
| `poses_{m}.json`, route/transect/population files | unchanged in 6.0 (Phase 7) |

The importer runs the coordinate-rounding match **one final time** to recover
the detection↔track linkage. Where it fails, the detection is imported without a
track membership rather than the whole frame being discarded — strictly better
than today's behaviour, and it never runs again.

> **Phase 1 notes.** Two limits of the legacy data are worth recording, both
> reported by `MigrationReport` rather than papered over:
>
> * **TRex detections cannot be told apart from detector output.** 5.x wrote
>   them into the same `detections.txt` in the same format with no marker, so
>   everything above `DETECTIONS_MARKER` is imported as `kind='detector'`. When
>   a `tracks_pixel_{m}/` folder is present the report warns that the project's
>   provenance is ambiguous.
> * **A geo/detection count mismatch is left unlinked, not guessed.** 5.x
>   aligned `georeferenced.txt` to `detections.txt` positionally; where the
>   counts disagree that alignment is unrecoverable, so those rows are reported
>   and skipped rather than matched by proximity.

Legacy *writers* stay available for one release behind an "Also write legacy
text outputs" toggle (`Input/WriteLegacyTextOutputs`, on by default), so users
with external scripts are not stranded.

**Only outputs the store fully replaces are routed through it** —
`detections.txt`, `georeferenced.txt`, `tracks.csv`, `tracks_pixel.csv`.
`fov_polygons.txt`, the segmentation JSON and `labels.csv` are deliberately
*not* gated: no stage writes them to the store yet, so the toggle would let a
user delete their only copy. A test guards that rule rather than the wiring, so
gating one of them later has to be a deliberate change.

Two honest limits while the toggle exists:

* **Turning it off breaks internal consumers**, not just external scripts. The
  Video Creator, the click tool and the QGIS layer builders still read the text
  files, so the checkbox says so. It becomes genuinely safe once those move
  (phases 5–6), which is when this stops being an expert-only switch.
* **Backend-native outputs are not gated.** The advanced tracker and the TRex
  importer write their own `tracks_pixel.csv`; only the store-derived export is
  suppressed. Gating those means changing the backends, which belongs with the
  Phase 6 exporter work.

---

## 10. Phasing

Each phase leaves the plugin working.

| Phase | Content |
|---|---|
| **0** ✅ | `core/gpkg.py` (aspatial GPKG boilerplate, stdlib sqlite3) + `core/store.py` (schema, vocabulary seeding, invariant triggers). No stage uses it yet. **Done:** 95 unit tests + 14 QGIS spikes; GPKG layout confirmed; ratchet 20 → 22. |
| **1** ✅ | 5.x importer (`core/migration.py`) + "Migrate 5.x…" action on the dock widget. Read-only with respect to the legacy files. **Done:** 33 unit tests on golden fixtures, 5 QGIS widget tests, 5 integration tests on flight 6's real output; ratchet 22 → 24. |
| **2** ✅ | Detection stage, TRex import → write the store (`core/detection_store.py`). `detection_id`, `species` (base classes 0 / -1 / -2), `enums`, `field_schema`, `detection_sources` and `class_mapping` live, plus `core/schema_editor.py` + the *Project Schema* dialog on the Detection tab; **latent bugs 1 and 2 fixed.** Dual-write with a parity check on every detection run, behind `Input/WriteLegacyTextOutputs` (§9). **Done:** +86 unit, +17 QGIS, +11 toggle; ratchet 24 → 25. |
| **3** ✅ | Georeference + tracking read/write the store (`core/track_store.py`), `track_runs` introduced, `georef_failures` populated. **`core/track_export.py` deleted.** **Done:** +28 unit, +6 integration on flight 6; ratchet held at 25 (see below). |
| **4** ✅ | Labelling tool onto the store (`core/label_store.py`): upsert materialisation (§6.2), manual tracks (§6.5), custom fields into `detections.attributes`, closed vocabulary in every categorical combo (§6.8), `origin_*` provenance on import. Consumer audit done (§8.2). **Done:** +46 unit; ratchet 25 → 26. |
| **5** ✅ | `core/stages.py`: dependency graph, cascade (**latent bug 3 fixed**), reconciliation, reset; stages record their own completion; `output_inventory` and the QGIS layer readers prefer the store; "Reset Step…" with locked-file handling. **Done:** +50 unit, +4 inventory, +15 reader, +7 QGIS; ratchet 26 → 27. |
| **6** | `core/exporters/` — COCO, MOT, YOLO, TRex npz, GeoJSON. Video Creator onto the store, removing the last positional alignment (§8.2). |
| **7** | Survey analytics onto the store (§8.2): explicit population filter, species/attribute stratification, perpendicular distances keyed by id, multi-project pooling by label. Route/transect files follow. |
| **8** | UI reorganisation: split the Processing tab into **Pre-Processing** and **Processing** (§10.1). Last, because it is the only phase that moves things the user has learned where to find. |

### 10.1 Phase 8 — splitting the Processing tab

The Processing tab currently runs ten numbered steps in one column, from frame
extraction to SAM3 geo-referencing. Split it in two:

| Tab | Steps |
|---|---|
| **Pre-Processing** | 1 Extract Frames · 2 Generate Flight Route · 5 Calculate Field of View · 6 Generate ALFS · 7 Export Frames as GeoTIFF · 8 Generate Orthomosaic |
| **Processing** | 3 Detect Animals · 3b Import TRex Tracklets · 4 Track Animals · 9 Run SAM3 Segmentation · 10 Geo-Reference Segmentation |

The split is not cosmetic — it is the dependency graph of §7 made visible:

```
frames ──> detections ──> georeferenced ──> tracks ──>     [Processing]
   └────> fov, alfs, geotiffs, orthomosaic                 [Pre-Processing]
```

Everything in Pre-Processing derives from poses and the DEM and is *independent
of any animal*; everything in Processing depends on detections. That is also why
the two can be re-run independently, and why a detector re-run marks tracking
stale but leaves the ALFS alone. Presenting them as one list has always implied
a sequence that does not exist.

Three things to settle when it lands:

- **Renumber or keep the numbers?** Keeping `1, 2, 5, 6, 7, 8` in one tab and
  `3, 4, 9, 10` in the other is confusing; renumbering breaks every screenshot,
  tutorial and support answer that refers to "step 7". Suggest renumbering
  within each tab (`P1…P6`, `A1…A5`) so no bare number means two things.
- **Where does geo-referencing appear?** It has no button today — it runs as
  part of the detection flow — but it is a distinct stage in the store, with
  its own `georef_failures`. Phase 3 gives it real status; Phase 8 should
  surface it in Processing rather than leave it invisible.
- **SAM3 sits in Processing.** It is genuinely dual-use — the same prompts can
  segment animals or scene content — but it is placed with the animal steps
  because that is its dominant use here, and because a step that could sit in
  either tab is better in the one where it is usually needed than duplicated
  across both. Worth revisiting only if scene segmentation becomes routine.

Config keys, step ids and `output_inventory` keys stay unchanged — this is a
presentation change only, so a saved configuration keeps working.

Phases 2–4 are the bulk. Phase 3 is where the payoff lands.

---

## 11. Ground rules

- `core/` stays QGIS-free and dependency-free. `sqlite3` is stdlib, so the
  entire store layer is testable under the existing QGIS-stubbed `tests/` suite.
- Every schema change bumps `bambi_meta.schema_version`. Readers refuse
  newer-than-known versions with an actionable message — mirroring what
  `read_custom_fields` already does (`core/labelling.py:238`).
- No positional parsing in new code. Anything reading by column index is a bug.
- Every write path goes through `core/store.py`. No stage opens sqlite directly.
- **Parameterised SQL only** — never f-string or `%`-formatted query text, not
  even for "safe" internal values. Correctness first, and it also keeps bandit's
  B608 quiet in a release that adds a great deal of SQL (§12).
- **Never hand a stage file path to QGIS as a layer source.** Windows locks a
  file that a layer holds open, which would block the next run of that stage —
  the same problem the plugin already has with added layers and GeoTIFF exports.
  Layers are built in memory from rows read through `core/store.py`, exactly as
  they are built from the text files today, so no QGIS handle ever touches a
  stage file. The GeoPackage container remains for the user's own inspection in
  the Browser, which is a deliberate, temporary act rather than something the
  plugin does behind their back.
- **Journal mode stays `DELETE`, not WAL.** With no concurrent QGIS reader to
  serve there is nothing to gain, and WAL needs shared memory (so it fails on
  network shares) and leaves `-wal`/`-shm` sidecars that a user copying "the
  .gpkg" would silently drop.
- Writes are transactional per stage: a cancelled run leaves the previous state
  intact, never a half-written table.
- `VACUUM` after a stage reset — `DELETE` alone does not shrink the file, and a
  monotonically growing project folder is a confusing thing to debug.

---

## 12. Testing and quality gates

The existing apparatus stays exactly as it is — this rework adapts to it, not the
other way round. All six compose services must be green at the end of **every**
phase; that is the operational meaning of "leaves the plugin working" in §10.

| Service | Runs | Gate |
|---|---|---|
| `checks` | bandit (all severities), detect-secrets, flake8 | must pass |
| `qt6-check` | pyqgis4-checker | must pass |
| `tests` | unit, QGIS stubbed | pass + coverage ratchet |
| `integration` | real pipeline on flight 6 | must pass |
| `qgis-tests` | real-QGIS smoke | must pass |
| `coverage-combine` | merged unit + qgis coverage | must not regress |

### 12.1 Unit tests (`tests/`, QGIS stubbed)

`sqlite3` is stdlib, so `core/store.py`, `core/gpkg.py`, `core/species.py`,
`core/schema_editor.py` and every exporter are fully unit-testable under the
stub. This release should **raise** the ratchet, not strain it — the logic being
added is exactly the kind the unit tier can reach, unlike the Qt-bound code that
holds coverage down today.

Required coverage, beyond the obvious happy paths:

- **The three §4.1 invariants**, one test each — ids never reused across sources,
  species/enum ids append-only, a producer only deletes rows it owns.
- **The §4.1 stability table row by row.** Each operation ("re-run detector",
  "edit a label keyframe", "change a class mapping", …) becomes a test asserting
  precisely which ids moved and which did not. That table is the contract; if it
  is not executable it will rot.
- **The §6.2 upsert matrix** — unchanged `(label_track_id, frame)` keeps its id;
  new frame mints; vanished frame deletes; **moved box drops its
  `detections_geo` row**. The last one is the load-bearing case for incremental
  re-georeferencing.
- **Base classes and fallbacks** — the three base species always exist, are
  rejected for rename/delete, `class_mapping` misses resolve to 0, and
  `species_id` can never be NULL.
- **Enum semantics** — `value_id` append-only, label rename does not touch
  stored rows, field rename rewrites JSON keys, deletion blocked while
  referenced.
- **Migration golden files.** Small legacy fixture folders committed to the
  repo: a `detections.txt` *with* a marker block, a `georeferenced.txt` with
  dropped rows, a `labels.json` with custom fields and custom species. Assert
  the resulting tables — especially the `class_id 0`-by-block disambiguation
  (§9), which has no second chance to be correct.
- **Exporters** — TRex npz round-trip, and COCO/YOLO/MOT output diffed against
  committed fixtures including the enum-label resolution and class-id remapping.

`tests/test_architecture.py` needs no change: it globs `core/*.py`, and `sqlite3`
being stdlib means the new modules may import it at module level without
touching `HEAVY_TOP_LEVEL_PACKAGES`. Confirm the guard still passes rather than
assuming it — the new modules are the first in `core/` to own persistent state.

### 12.2 Integration tests (`tests_integration/`, real flight 6)

This tier is where the rework's actual claims get proven, because they are claims
about whole-pipeline consistency:

- **Total accounting.** After a full run, every detection has *either* a
  `detections_geo` row *or* a `georef_failures` row — never neither. The old
  format could not express this assertion at all, which is precisely why
  detections went missing silently.
- **No orphans.** Every `track_members.detection_id` resolves; counts agree
  across stages.
- **The `track_export.py` regression.** With a detection deliberately made
  unprojectable, the *other* detections in that frame must still receive pixel
  tracks. The old code discarded the whole frame (`track_export.py:106`); a test
  should keep that behaviour from creeping back.
- **Dual-write parity (phases 2–3).** While the compat toggle exists, legacy text
  output regenerated from the store must be semantically identical to what 5.x
  wrote for the same input. This is the strongest available safety net for a
  format migration and it is nearly free while both paths coexist — it should
  gate the phase, not be an afterthought.
- **Migration on real data.** Migrate flight 6's 5.x outputs and assert the table
  contents against the text files parsed independently.

### 12.3 QGIS smoke tests (`tests_qgis/`, real QGIS)

Two of these are **de-risking spikes that belong in Phase 0**, because layout
decisions rest on their answers:

- **Aspatial GeoPackage opens in QGIS.** A stage file with attribute-only tables
  must be loadable and readable — this validates the §2.1 decision, and if it
  fails we fall back to plain `.sqlite` before any stage depends on it.
- **Layer locking** (§13, Q1). Load a layer from a stage file, rewrite that
  stage, assert no lock error and that a refresh shows the new rows. Note the
  limit of this tier: it runs on Linux, where the answer is permissive and
  therefore *not* the answer that matters. Windows locking is what drove the
  memory-layer rule in §11, and these tests only document the contrast.

Then, later: the *Project Schema* dialog round-trips species/enums/fields, and
the labelling tool's combos populate from the store.

### 12.4 Quality gates

- **flake8 / bandit / detect-secrets** stay green throughout. The one new risk
  this release introduces is B608 (hardcoded SQL) — avoided by the
  parameterised-SQL ground rule in §11, not by `# nosec`.
- **pyqgis4-checker** applies to the new dialogs (Project Schema, Reset stage):
  fully qualified enums, `.exec()`, no BOMs.
- **Coverage** is checked by `run_all_tests.sh` merging all three tiers. The
  ratchet moves up when a phase lands, never down; a phase that lowers combined
  coverage is not finished.

### 12.5 Per-phase gates

| Phase | What must be proven before it is called done |
|---|---|
| 0 ✅ | Aspatial-GPKG and locking spikes answered; §4.1 invariants under test |
| 1 ✅ | Migration golden files + real 5.x flight-6 folder migrate correctly |
| 2 ✅ | Dual-write parity for `detections.txt`; species/enum editors round-trip |
| 3 ✅ | Total accounting + no-orphans + `track_export` regression tests pass |
| 4 ✅ | Upsert matrix; manual tracks never touch another run's rows |
| 5 ✅ | Cascade correctness; `output_inventory` reconciles a hand-deleted file |
| 6 | Exporter fixtures, incl. enum resolution and class-id remapping |
| 7 | Analytics results unchanged vs. the pre-rework baseline on flight 6 |
| 8 | Every step still reachable and runnable; saved configs load unchanged |

Phase 7's gate is worth stating plainly: the Praschl2026 R validation baseline
must still reproduce. A data-plumbing release that quietly changes a population
estimate has failed, however clean the schema is.

### 12.6 Test growth

Baseline at the start of 6.0: **721 unit** tests (43 files), **47 QGIS smoke**
(4 files), **12 integration** (2 files), unit ratchet `--cov-fail-under=20`,
combined unit+qgis coverage ~41%.

New code gets new test modules rather than being appended to existing ones, so
the mapping from module to suite stays legible:

| New test module | Covers | ≈ |
|---|---|---|
| `test_gpkg.py` | aspatial boilerplate, pragmas, create/open, corrupt-file handling | 15 |
| `test_store.py` | schema creation, version guard, transactions, `ATTACH` helper | 30 |
| `test_store_ids.py` | §4.1 invariants + the stability table, row by row | 25 |
| `test_species.py` | base classes, protection, rename/delete rules, `class_mapping` fallback | 25 |
| `test_enums.py` | `value_id` append-only, ordinal, rename, delete-blocked, field-rename key rewrite | 25 |
| `test_field_schema.py` | types incl. `enum`, scopes, coercion, prune-to-schema | 15 |
| `test_migration_detections.py` | golden files, `class_id 0` marker-block split | 20 |
| `test_migration_labels.py` | `labels.json` → tables, enum resolution, unmatched-value append | 15 |
| `test_stages_cascade.py` | dependency graph, stale marking, reconcile-with-disk | 25 |
| `test_labelling_materialisation.py` | §6.2 upsert matrix, manual tracks, run isolation | 25 |
| `test_exporters_*.py` | COCO / MOT / YOLO / TRex fixtures, enum + class-id remapping | 40 |
| `test_analytics_filters.py` | population filter: `-2` excluded, single run, manual/detector choice | 20 |
| `test_multi_project_pooling.py` | join by label, vocabulary-mismatch reporting | 15 |

≈ **+295 unit tests → ~1000**. Plus roughly **+30 QGIS smoke** (aspatial-GPKG
layers, locking spike, Project Schema dialog, labelling combos, Reset stage →
~77) and **+13 integration** (total accounting, no-orphans, `track_export`
regression, dual-write parity, real-folder migration, analytics baseline → ~25).

Ratchet targets, raised **in the same commit as the phase that earns them** —
never retroactively, never lowered:

| After phase | `--cov-fail-under` | Combined |
|---|---|---|
| 0 ✅ | 22 | 42 |
| 2 ✅ | 24 | 44 |
| 3 | 27 → held at 25 | 46 |
| 4 ✅ | 30 → held at 26 | 47 |
| 5 ✅ | 32 → held at 27 | 48 |
| 6–7 | 35 | ~50 |

**Phase 3 fell short: 25.7% against a 27% target, so the ratchet stayed at 25.**
The reason is worth recording rather than quietly restating the goal. Phase 3's
new logic is well covered (`core/track_store.py` is exercised by 28 unit tests),
but the phase also *added* lines to `bambi_processing.py` — the store write-backs
inside `run_georeference` and `_run_builtin_tracking` — and those sit behind
alfspy, a real DEM and a GPU-less render context, so the unit tier cannot reach
them. They are covered by the integration tier instead, which the ratchet does
not count. Expect the same drag in phases 4–5, and read the combined figure
(`run_all_tests.sh`) rather than the unit one where a phase touches the monolith.

These are targets, not predictions — the point is that the new `core/` code is
plain-Python and stdlib-only, so unlike the Qt-bound modules that hold coverage
down today it is *all* reachable from the unit tier. A phase whose code lands
without moving the ratchet is a phase that was not tested, and should not be
called done.

The counts are deliberately not a goal in themselves: a phase adding 40 tests
that all assert the happy path is worth less than one adding 15 that pin the
§4.1 stability table. Where the two conflict, the table wins.

---

## 13. Open questions

1. ~~**QGIS layer locking.**~~ **Resolved — Windows locks, so the plugin never
   file-backs a layer.** The Phase 0 spike showed rewrite-and-delete working
   under a loaded layer, but only because it runs on Linux; on Windows a layer
   holds the file, which is already a known pain with added layers and GeoTIFF
   exports. Rather than build an unload/reload dance around every stage run, the
   plugin never hands a stage file path to QGIS at all (§11) — layers are
   memory layers built from rows, exactly as they are built from the text files
   today. Ad-hoc inspection in the Browser stays available and is the user's own
   deliberate act; §7 covers what happens if a stage is reset while they have it
   open.
2. **Interpolated track points.** Storing them as synthetic
   `source_class='interpolated'` detections keeps joins uniform but inflates the
   detections table. The alternative is coordinates directly on `track_members`.
   Decide in Phase 3.
3. **Do `poses_{m}.json` move into the store?** They are consumed in
   alfspy-shaped form by several stages; converting them buys little and touches
   a lot. Currently deferred to Phase 7 / never.
4. **Compat-toggle lifetime.** One release (6.x → dropped in 7.0) is proposed;
   confirm against how many users script against the text outputs.
5. ~~**Who edits `class_mapping`?**~~ **Resolved:** the Detection configuration
   tab, alongside the species editor (§3.1). Still open within that: whether a
   mapping is stored per project or as a reusable preset shared across flights.
   It is a property of the *model*, not the flight, which argues for a preset
   with a per-project override — same shape as the species preset.
6. **Is `track_runs.is_active` user-facing?** Keeping several tracker runs is
   nearly free and useful for comparison, but it needs a run selector in the
   dock widget to be worth anything. Fallback: always activate the newest run
   and leave the table ready for a later UI.
