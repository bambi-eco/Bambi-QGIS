# -*- coding: utf-8 -*-
"""Unit tests for BoxProjectionWorker (the cross-modality projection worker).

The worker only needs ``pyrr`` vectors and the ``alfspy`` Camera at run time;
both are replaced with minimal fakes so the projection pipeline (origin
lookup, correction, georef matching, world-to-pixel maths) runs without the
heavyweight rendering stack. The fake camera uses identity view/projection
matrices, so expected pixel coordinates follow directly from the NDC mapping
``px = (x + 1) * w / 2`` and ``py = h - (y + 1) * h / 2``.
"""
import json
import sys

import pytest

from bambi_wildlife_detection.bambi_box_projector import BoxProjectionWorker
from tests.fakes import SignalRecorder, install_fake_render_stack

IMG_W, IMG_H = 640, 512  # worker fallback resolution (no frame files on disk)
ORIGIN = (1000.0, 2000.0, 300.0)


@pytest.fixture
def fake_render_stack(monkeypatch):
    """Install fake ``pyrr`` and ``alfspy`` modules into sys.modules."""
    install_fake_render_stack(monkeypatch)


def _write_pipeline_folder(tmp_path, with_georef=True, with_poses=True):
    """Create the minimal target-folder layout the worker reads."""
    # DEM metadata next to the (non-existent) mesh file
    (tmp_path / "dem.json").write_text(json.dumps({"origin": list(ORIGIN)}))

    if with_georef:
        georef_dir = tmp_path / "georeferenced_t"
        georef_dir.mkdir()
        # Local box corners (relative to DEM origin): (-0.5, -0.25) .. (0.5, 0.25)
        line = (
            f"0 0 {ORIGIN[0] - 0.5} {ORIGIN[1] - 0.25} {ORIGIN[2]} "
            f"{ORIGIN[0] + 0.5} {ORIGIN[1] + 0.25} {ORIGIN[2]} 0.9 1"
        )
        (georef_dir / "georeferenced.txt").write_text("# header\n" + line + "\n")

    if with_poses:
        poses = {"images": [{
            "imagefile": "frame_000000.jpg",
            "location": [0.0, 0.0, 10.0],
            "rotation": [0.0, 0.0, 0.0],
            "fovy": [50.0],
        }]}
        (tmp_path / "poses_w.json").write_text(json.dumps(poses))

    return str(tmp_path)


def _make_worker(folder, frames=None):
    if frames is None:
        frames = [{
            "frame_idx": 0,
            "boxes_green": [(10, 10, 20, 20, 0.9, 1)],
            "boxes_blue": [],
        }]
    return BoxProjectionWorker(
        target_folder=folder,
        dem_path=str(folder) + "/dem.glb",
        correction_path="",
        src_modality="t",
        frames=frames,
    )


# Expected projection of the local box (-0.5,-0.25)..(0.5,0.25) through the
# identity camera onto the 640x512 fallback image.
EXPECTED_BBOX = (
    (1 - 0.5) * IMG_W / 2.0,            # x1 = 160
    IMG_H - (1 + 0.25) * IMG_H / 2.0,   # y1 = 192
    (1 + 0.5) * IMG_W / 2.0,            # x2 = 480
    IMG_H - (1 - 0.25) * IMG_H / 2.0,   # y2 = 320
)


class TestProject:
    def test_projects_matched_boxes_to_other_modality(self, fake_render_stack, tmp_path):
        folder = _write_pipeline_folder(tmp_path)
        worker = _make_worker(folder)
        worker.progress = SignalRecorder()

        results = worker._project()

        assert set(results.keys()) == {0}
        assert results[0]["blue"] == []
        green = results[0]["green"]
        assert len(green) == 1
        assert green[0][:4] == pytest.approx(EXPECTED_BBOX)
        assert green[0][4] == 0.9   # confidence carried over
        assert green[0][5] == 1     # class id carried over
        assert worker.progress.calls[-1] == (100,)

    def test_origin_found_via_folder_scan_fallback(self, fake_render_stack, tmp_path):
        folder = _write_pipeline_folder(tmp_path)
        worker = _make_worker(folder)
        worker._dem_path = ""  # force the target-folder JSON scan
        results = worker._project()
        assert results[0]["green"][0][:4] == pytest.approx(EXPECTED_BBOX)

    def test_frame_index_beyond_poses_yields_empty(self, fake_render_stack, tmp_path):
        folder = _write_pipeline_folder(tmp_path)
        frames = [{"frame_idx": 7, "boxes_green": [(1, 1, 2, 2, 0.9, 1)], "boxes_blue": []}]
        worker = _make_worker(folder, frames)
        results = worker._project()
        assert results[0] == {"green": [], "blue": []}

    def test_unmatched_boxes_are_dropped(self, fake_render_stack, tmp_path):
        folder = _write_pipeline_folder(tmp_path)
        # Confidence differs from the georef entry beyond tolerance
        frames = [{"frame_idx": 0, "boxes_green": [(1, 1, 2, 2, 0.5, 1)], "boxes_blue": []}]
        worker = _make_worker(folder, frames)
        results = worker._project()
        assert results[0]["green"] == []

    def test_missing_georef_raises(self, fake_render_stack, tmp_path):
        folder = _write_pipeline_folder(tmp_path, with_georef=False)
        worker = _make_worker(folder)
        with pytest.raises(RuntimeError, match="No geo-referenced detections"):
            worker._project()

    def test_missing_poses_raises(self, fake_render_stack, tmp_path):
        folder = _write_pipeline_folder(tmp_path, with_poses=False)
        worker = _make_worker(folder)
        with pytest.raises(RuntimeError, match="Poses file not found"):
            worker._project()

    def test_missing_alfspy_raises_helpful_error(self, fake_render_stack, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "alfspy", None)
        monkeypatch.delitem(sys.modules, "alfspy.core", raising=False)
        monkeypatch.delitem(sys.modules, "alfspy.core.rendering", raising=False)
        folder = _write_pipeline_folder(tmp_path)
        worker = _make_worker(folder)
        with pytest.raises(RuntimeError, match="alfspy is not available"):
            worker._project()


class TestRun:
    def test_run_emits_finished_on_success(self, fake_render_stack, tmp_path):
        folder = _write_pipeline_folder(tmp_path)
        worker = _make_worker(folder)
        worker.progress = SignalRecorder()
        worker.finished = SignalRecorder()
        worker.error = SignalRecorder()

        worker.run()

        assert worker.error.calls == []
        assert len(worker.finished.calls) == 1
        (results,) = worker.finished.calls[0]
        assert results[0]["green"][0][:4] == pytest.approx(EXPECTED_BBOX)

    def test_run_emits_error_on_failure(self, fake_render_stack, tmp_path):
        folder = _write_pipeline_folder(tmp_path, with_georef=False)
        worker = _make_worker(folder)
        worker.progress = SignalRecorder()
        worker.finished = SignalRecorder()
        worker.error = SignalRecorder()

        worker.run()

        assert worker.finished.calls == []
        assert len(worker.error.calls) == 1
        assert "No geo-referenced detections" in worker.error.calls[0][0]
