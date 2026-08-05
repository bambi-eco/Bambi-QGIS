# Results, Flights and Export

New in 6.0. Where earlier versions wrote each processing step to a text file,
results now live in GeoPackages that record what they mean — which detection a
track point came from, why a detection could not be placed on the DEM, what a
density figure counted. This page covers what that changes for you.

The 5.x text files are still written alongside, so external scripts keep
working. See [Legacy text outputs](#legacy-text-outputs).

---

## Where results live

Each flight's target folder contains:

```
<target folder>/
    project.gpkg              species, enums, custom fields, configuration,
                              step status
    bambi_t/                  thermal results, one file per step
        detections.gpkg
        georeferenced.gpkg
        tracks.gpkg
        fov.gpkg
        labels.gpkg
        segmentation.gpkg
    bambi_w/                  RGB, same set
    frames_t/  frames_w/      extracted frames (unchanged)
    poses_t.json …            camera poses (unchanged)
```

These are ordinary GeoPackages: open one from the QGIS **Browser** panel to
look at a table, or delete a file to make that step run again from scratch.
Deleting `bambi_t/tracks.gpkg` in Explorer is a valid way to reset tracking —
the plugin reconciles what it recorded against what is actually on disk, and
the files win.

> **On Windows**, a file you have opened in the Browser is locked. Close the
> layer before deleting the file or resetting the step, or the deletion is
> refused with an explanation.

### Why it is not one file

One file per step means `rm bambi_t/tracks.gpkg` still works as a reset, that
thermal and RGB never contend, and that a corrupt file costs you one step
rather than the project. Queries across steps use SQLite's `ATTACH`.

---

## Migrating an existing project

Open a 5.x target folder and **Migrate 5.x…** appears next to the target
folder on the Input tab. It reads the existing outputs and writes the
equivalent rows into the new format. **The existing files are only read**,
never modified or deleted, so a migration that goes wrong is fixed by deleting
the generated `.gpkg` files and trying again.

Migration refuses to run over a folder that already has a store, because it
adds rows rather than reconciling them — it would duplicate what is there.

Two limits of 5.x data, both reported when they apply:

- **TRex detections cannot be told apart from detector output.** 5.x wrote both
  into the same `detections.txt` in the same format, so everything above the
  labelling tool's marker is imported as detector output.
- **Where `georeferenced.txt` and `detections.txt` disagree on how many rows a
  frame has**, the original alignment is unrecoverable, so those rows are
  reported and left unlinked rather than matched by guesswork.

---

## Flights

A QGIS project can hold several flights. The **Flight** box at the top of the
Input tab shows the active one, `+` adds another, **Rename…** renames it and
🗑 removes it.

- Each flight has **its own target folder**. This is enforced: the result files
  are per folder, so two flights sharing one would overwrite each other's
  detections.
- Each flight has **its own configuration**, stored in its `project.gpkg`. Hand
  someone the folder and the settings that produced it come with it.
- Each flight has **its own QGIS layer group**, named after the flight.
  Renaming the flight renames the group.
- **Exactly one flight is active.** Everything else in the plugin reads that
  one flight, so nothing changes about how the steps work.

Removing a flight (🗑) takes it out of the project and removes its layer group,
after a confirmation. **Nothing on disk is deleted** — the target folder keeps
its frames, detections, tracks and configuration, so pointing a flight at the
folder again picks up exactly where it left off. To delete the results
themselves, delete the folder in a file manager, or reset individual stages
with **Reset Step**.

Switching and removing flights are blocked while a step is running.

Projects from earlier versions become a single flight named after their target
folder, and behave exactly as before.

### Flights in the survey analytics

Distance sampling and population estimation have always been able to pool
several target folders, and still can — flights and projects can stay separate
if that suits you. **+ Add Flight…** is the shortcut for flights this project
already knows about: the folder and the DEM are looked up rather than browsed
for.

---

## The project schema

**Edit Project Schema…** on the **Project** configuration tab (also reachable from the
labelling tool's gear button and from *Manage species…* beside any species
list) edits three things that belong to the whole project.

### Species

Every project has three base classes that cannot be renamed or removed:

| Species | Meaning |
|---|---|
| `animal` | an animal, species not determined — what the detector reports |
| `unknown` | not yet determined by a person |
| `not-an-animal` | a labelled false positive |

`not-an-animal` is **excluded from every survey analytic**: leaving it in would
bias a density estimate upward. `unknown` and `animal` do count — the
distinction is determinacy, not presence.

Everything else is yours to define. Species keep their identity when renamed,
so correcting a spelling never changes the meaning of data already recorded,
and adding a species never renumbers the existing ones.

For publishing, a species can carry a **scientific name**, a rank and a **GBIF
taxon key**. The name you work with (`roe deer`) is a vernacular name; Darwin
Core needs the scientific one (`Capreolus capreolus`), and a GBIF key lets GBIF
resolve the record exactly instead of matching a string. Scientific names are
pre-filled for the built-in species as editable starting points; GBIF keys are
not, because a wrong identifier publishes confidently wrong data — take them
from the species' page on gbif.org.

### Detector class mapping

A detector reports whatever classes its weights were trained on — `0`, `1`, `2`,
or names of its own — and those are not the project's species. **Edit Class
Mapping…** on the **Detection** configuration tab connects the two. It sits with
the model rather than with the species, because the mapping describes *these
weights*: a different model needs a different mapping even for the same species.

The table lists the classes the detector actually reported, with how often each
one occurred, so there is nothing to guess. Anything left unmapped counts as
`animal`, which is why a single-class detector needs no configuration at all.

Applying a change **re-reads the detections already stored** — every detection
keeps the raw class it was reported with, so correcting a mapping never means
running the detector again. The boxes do not move, so geo-referencing and
tracking stay valid; only the species-dependent analytics need re-running.

### Enums and custom fields

An **enum** is a reusable list of values — `sex`, `age` and `occlusion` are
ordinary enums you can extend. Values keep their identity when renamed, so
fixing a label never orphans the data using it.

A **custom field** adds an attribute to detections, tracks, frames or the
project. Enum-typed fields are picked from a list, which is how typos are kept
out; `string` fields remain available for genuinely free-form notes.

Custom fields now travel with the data through geo-referencing and tracking and
reach the exports — in 5.x they stayed in `labels.json`.

---

## Re-running steps

Steps know what they depend on. Re-running detection marks geo-referencing,
tracking, the perpendicular distances and the population estimate as **out of
date** — their files are kept, they are simply flagged.

**Reset Step…** (next to Refresh Status) deletes one step's outputs so it runs
from scratch. It names what depended on the step before doing anything.

---

## Export

The **Export** box on the Processing tab writes the current flight out in a
standard format. Species names and enum values are resolved, so nothing
receives an internal id, and custom fields travel where the format has room.

| Format | For | `not-an-animal` |
|---|---|---|
| **COCO** | detection training; carries custom fields as annotation attributes | dropped |
| **YOLO** | training labels plus `data.yaml` | dropped |
| **MOT** | tracking benchmarks, with a sidecar for what the columns cannot hold | dropped |
| **TRex `.npz`** | back to TRex | dropped |
| **GeoJSON** | detections or tracks in the project CRS | kept |
| **Camtrap DP** | survey package: deployment, media, observations | kept as `blank` |
| **Darwin Core Archive** | GBIF publishing | never included |

Two things worth knowing:

- **Darwin Core writes one occurrence per track**, not per detection — a track
  is one animal seen once, and publishing every detection would report the same
  roe deer hundreds of times. Tracks whose species has no scientific name are
  held back, and the export says how many.
- **GeoJSON, Camtrap DP and Darwin Core need the target CRS**, and refuse
  before asking where to save rather than writing coordinates in the wrong
  reference system.

---

## What the analytics counted

Density, coverage, distance sampling and population estimation now record the
filter they used beside the result — which tracking run, whether manual tracks
were included, which species, how many false positives were excluded. A number
can be traced back to the rows behind it.

Two rules apply to all of them:

- **One tracker run contributes.** Built-in, BoxMOT and TRex are alternative
  descriptions of the same animals; pooling two would double-count.
- **Manual tracks are added, not chosen between.** Labels are usually animals
  the detector missed. The exception is a label created with *Import as label
  track*, which replaces the tracker track it came from rather than adding to
  it.

---

## Legacy text outputs

`detections.txt`, `georeferenced.txt`, `tracks.csv` and `tracks_pixel.csv` are
still written beside the new files, controlled by **Also write legacy text
outputs** under Output Configuration. They will be removed in 7.0.

Leave it on unless you know you do not need them: the Video Creator, the
click tool and the QGIS layers can still fall back to reading them, and older
scripts of your own may depend on them.

`fov_polygons.txt`, the segmentation JSON and `labels.csv` are *not* covered by
the switch, because no step writes them to the store yet — turning it off would
otherwise delete your only copy.
