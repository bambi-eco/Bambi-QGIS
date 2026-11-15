# -*- coding: utf-8 -*-
"""A light-field tile gets the frames that can see it - and only those.

The mosaic of *Generate ALFS* spans the whole DEM. For a tile no frame
reached, the shot filter fell back to **all** frames, "so tiles on the
boundary still receive at least one shot": every tile away from the flight
had every frame of the flight read, uploaded and rendered into it, to come
out empty and be thrown away.

Not rendering such a tile is only safe when a frame's reach is never
underestimated. The estimate it replaces - height times the tangent of half
the field of view, times 2.5 - was too small for a camera that does not look
straight down: at 20 degrees of tilt it stopped 25 m short of where the
frame's far corner lands from 100 m up. The reach now comes from the
matrices the renderer projects the frame with, and is checked here against
what the renderer draws.
"""
import math
import os

import numpy as np
import pytest

from bambi_wildlife_detection.core import ortho_tiling
from bambi_wildlife_detection.core.ortho_tiling import ShotReach, shot_reach

from .software_gl import (
    FOVY, HEIGHT_AGL, N_FRAMES, small_flight, software_gl)

ZERO = {"x": 0.0, "y": 0.0, "z": 0.0}
ASPECT = 160 / 96
#: Poses to try: straight down, tilted, tilted and turned.
ROTATIONS = ([0, 0, 0], [20, 0, 0], [0, 25, 40], [30, 10, 115], [-15, -20, 250])


class MatrixShot:
    """A shot as far as its reach is concerned: a camera and the matrices
    the renderer would project it with."""

    def __init__(self, rotation, location=(10.0, -20.0, 100.0), fovy=50.0,
                 aspect=ASPECT):
        pytest.importorskip("alfspy")
        pytest.importorskip("pyrr")
        from alfspy.core.rendering import Camera

        from bambi_wildlife_detection.core.camera_pose import shot_pose

        position, quaternion = shot_pose(
            {"location": list(location), "rotation": list(rotation)},
            ZERO, ZERO)
        self.camera = Camera(fovy=fovy, aspect_ratio=aspect,
                             position=position, rotation=quaternion)

    def get_view(self):
        return self.camera.get_view()

    def get_proj(self):
        return self.camera.get_proj()

    def get_correction(self):
        return np.eye(4)


def _footprint(shot, ground=0.0):
    """Where the four frame corners meet flat ground."""
    position = np.array([float(v) for v in shot.camera.transform.position])
    points = []
    for ray in ortho_tiling._corner_rays(shot):
        distance = (ground - position[2]) / ray[2]
        points.append(position[:2] + distance * ray[:2])
    return np.array(points)


# ---------------------------------------------------------------------------
# How far a frame sees
# ---------------------------------------------------------------------------

class TestShotReach:
    def test_straight_down_it_is_the_frames_half_diagonal(self):
        x, y, radius = shot_reach(MatrixShot([0, 0, 0]))

        half_height = 100.0 * math.tan(math.radians(25.0))
        diagonal = half_height * math.hypot(1.0, ASPECT)
        assert (x, y) == (10.0, -20.0)
        assert radius == pytest.approx(diagonal * 1.02, rel=1e-4)

    @pytest.mark.parametrize("rotation", ROTATIONS)
    def test_it_ends_where_the_furthest_corner_meets_the_ground(self, rotation):
        """Never short of it - a frame must not be missing from a tile it
        covers - and not far beyond it either."""
        shot = MatrixShot(rotation)
        x, y, radius = shot_reach(shot)

        corners = _footprint(shot)
        furthest = np.hypot(corners[:, 0] - x, corners[:, 1] - y).max()
        assert furthest <= radius <= furthest * 1.03

    def test_a_tilted_camera_reaches_further_than_was_assumed(self):
        """What was assumed: height x tan(fov / 2) x 2.5, whatever the
        tilt."""
        assumed = 100.0 * math.tan(math.radians(25.0)) * 2.5
        assert shot_reach(MatrixShot([0, 0, 0]))[2] < assumed
        assert shot_reach(MatrixShot([20, 0, 0]))[2] > assumed + 20.0

    def test_higher_ground_does_not_extend_it(self):
        """Ground above the lowest ground is met sooner."""
        shot = MatrixShot([20, 0, 0])
        _x, _y, radius = shot_reach(shot)
        for ground in (10.0, 40.0, 80.0):
            corners = _footprint(shot, ground)
            assert np.hypot(corners[:, 0] - 10.0,
                            corners[:, 1] + 20.0).max() <= radius

    def test_the_lowest_ground_is_where_heights_are_counted_from(self):
        shot = MatrixShot([0, 0, 0])
        assert shot_reach(shot, ground_z=50.0)[2] == pytest.approx(
            shot_reach(shot)[2] / 2)

    def test_a_camera_looking_at_the_horizon_sees_everywhere(self):
        assert shot_reach(MatrixShot([70, 0, 0]))[2] == math.inf

    def test_a_camera_not_above_the_ground_sees_everywhere(self):
        assert shot_reach(MatrixShot([0, 0, 0]), ground_z=99.5)[2] == math.inf
        assert shot_reach(MatrixShot([0, 0, 0]), ground_z=300.0)[2] == math.inf

    def test_without_matrices_it_is_estimated_as_before(self):
        position = type("P", (), {"x": 5.0, "y": 6.0, "z": 100.0})()
        camera = type("C", (), {
            "transform": type("T", (), {"position": position})(),
            "fovy": 50.0})()
        shot = type("S", (), {"camera": camera})()

        assert shot_reach(shot) == pytest.approx(
            (5.0, 6.0, 100.0 * math.tan(math.radians(25.0)) * 2.5))


class TestShotsOfATile:
    def test_each_tile_gets_the_shots_that_reach_it(self):
        near = MatrixShot([0, 0, 0], location=(0.0, 0.0, 100.0))
        far = MatrixShot([0, 0, 0], location=(1000.0, 0.0, 100.0))
        reach = ShotReach([near, far])

        assert reach.for_tile((-50, -50, 50, 50)) == [near]
        assert reach.for_tile((950, -50, 1050, 50)) == [far]
        assert reach.for_tile((-2000, -50, 2000, 50)) == [near, far]
        assert reach.for_tile((400, 400, 500, 500)) == []
        assert ShotReach([]).for_tile((0, 0, 1, 1)) == []

    def test_a_tile_beside_the_frame_is_reached_by_its_edge(self):
        shot = MatrixShot([0, 0, 0], location=(0.0, 0.0, 100.0))
        _x, _y, radius = shot_reach(shot)
        reach = ShotReach([shot])

        assert reach.for_tile((radius - 1, -5, radius + 50, 5)) == [shot]
        assert reach.for_tile((radius + 1, -5, radius + 50, 5)) == []

    def test_a_tilted_frame_is_not_missing_from_the_tile_it_looks_at(self):
        """Tilted by 20 degrees the frame lands up to 141 m away; with a
        reach of 117 m this tile was rendered without it."""
        shot = MatrixShot([20, 0, 0], location=(0.0, 0.0, 100.0))
        corners = _footprint(shot)
        far_corner = corners[np.argmax(np.hypot(corners[:, 0], corners[:, 1]))]
        tile = (far_corner[0] - 3, far_corner[1] - 3,
                far_corner[0] + 3, far_corner[1] + 3)

        assert math.hypot(*far_corner) > 130.0
        assert ShotReach([shot]).for_tile(tile) == [shot]

    def test_it_says_how_many_shots_a_tile_gets(self):
        near = MatrixShot([0, 0, 0], location=(0.0, 0.0, 100.0))
        far = MatrixShot([0, 0, 0], location=(1000.0, 0.0, 100.0))
        logs = []
        ShotReach([near, far]).for_tile((-50, -50, 50, 50), logs.append)
        assert logs == ["  Shot filter: 1/2 shots relevant for this tile"]


# ---------------------------------------------------------------------------
# Against what the renderer draws
# ---------------------------------------------------------------------------

def _lit_from_above(ctx, location, rotation, half=450.0, size=900):
    """Render one white frame onto flat ground, seen from straight above:
    ``(coverage, shot's reach)``. The view is the one the step has of the
    DEM."""
    from alfspy.core.rendering import (
        Camera, CtxShot, MeshData, Renderer, Resolution)
    from pyrr import Quaternion, Vector3

    from bambi_wildlife_detection.core.camera_pose import shot_pose
    from bambi_wildlife_detection.core.render_resources import (
        release_renderer, release_shot)

    mesh = MeshData(
        np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0],
                  [-half, half, 0]], np.float32),
        np.array([[0, 2, 1], [0, 3, 2]], np.uint32))
    above = Camera(orthogonal=True, orthogonal_size=(2 * half, -2 * half),
                   position=Vector3([0.0, 0.0, 500.0], dtype="f4"),
                   rotation=Quaternion(), near=0.1, far=10000.0)
    position, quaternion = shot_pose(
        {"location": list(location), "rotation": list(rotation)}, ZERO, ZERO)
    shot = CtxShot(ctx, np.ones((96, 160, 4), dtype=np.float32), position,
                   quaternion, 50.0, ASPECT, None)
    renderer = Renderer(Resolution(size, size), ctx, above, mesh)
    try:
        coverage = np.asarray(renderer.render_integral_raw([shot]).coverage)
        return coverage > 0, shot_reach(shot)
    finally:
        release_renderer(renderer)
        release_shot(shot)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_nothing_is_drawn_beyond_a_shots_reach(rotation):
    """A real shot rendered onto flat ground: every pixel it lights lies
    within its reach, and it lights pixels almost that far out."""
    from bambi_wildlife_detection.core.render_resources import release_context

    ctx, _library = software_gl()
    half, size = 450.0, 900
    try:
        # which way the rows run: a frame known to lie in the north
        northern, _reach = _lit_from_above(ctx, (0.0, 300.0, 100.0), (0, 0, 0))
        lit, (x, y, radius) = _lit_from_above(
            ctx, (10.0, -20.0, 100.0), rotation)
    finally:
        release_context(ctx)

    rows, columns = np.nonzero(lit)
    assert len(rows) > 500
    metre = 2 * half / size
    east = -half + (columns + 0.5) * metre
    north = -half + (rows + 0.5) * metre
    if np.nonzero(northern)[0].mean() < size / 2:      # row 0 is the north
        north = -north
    furthest = np.hypot(east - x, north - y).max()

    assert furthest <= radius
    assert furthest >= radius * 0.95
    if rotation != [0, 0, 0]:
        # ... which is beyond what was assumed for any camera at this
        # height: such a frame was left out of tiles it lights
        assert furthest > 100.0 * math.tan(math.radians(25.0)) * 2.5


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

def _tiles(folder):
    return sorted(name for name in os.listdir(folder) if name.endswith(".tif"))


def _extent(folder):
    """What the tiles of a folder cover together, and how many pixels of
    them show something."""
    import rasterio

    bounds, lit = [], 0
    for name in _tiles(folder):
        with rasterio.open(os.path.join(folder, name)) as dataset:
            bounds.append(tuple(dataset.bounds))
            lit += int((dataset.read(dataset.count) > 0).sum())
    return (min(b[0] for b in bounds), min(b[1] for b in bounds),
            max(b[2] for b in bounds), max(b[3] for b in bounds)), lit


class TestGenerateAlfs:
    @pytest.fixture
    def flight(self, tmp_path, monkeypatch):
        return small_flight(tmp_path, monkeypatch)

    def _run(self, flight, name, **settings):
        folder = os.path.join(flight["root"], name)
        os.makedirs(folder)
        log = []
        flight["processor"].run_alfs(dict(
            flight["config"], alfs_output_file=os.path.join(folder, "alfs.tif"),
            **settings), log_fn=log.append)
        return folder, log

    def test_tiles_no_frame_sees_are_not_rendered(self, flight):
        """A kilometre of DEM in tiles of 32 m: 1024 of them, the flight
        over a handful. Every one of the others had all six frames rendered
        into it."""
        folder, log = self._run(flight, "tiled", alfs_max_tile_size=64)

        rendered = len(flight["textures"])
        assert 4 <= rendered <= 30
        assert any("tile(s) lie where no frame sees the ground: not rendered"
                   in line for line in log)
        skipped = next(int(line.split()[0]) for line in log
                       if "lie where no frame sees" in line)
        assert skipped + rendered == 32 * 32
        assert len(_tiles(folder)) >= 4

    def test_no_frame_is_read_for_them(self, flight):
        """6144 times a frame was read and uploaded for this mosaic - six
        frames for each of 1024 tiles."""
        self._run(flight, "tiled", alfs_max_tile_size=64)

        rendered = len(flight["textures"])
        assert rendered <= 30
        assert len(flight["read"]) <= rendered * N_FRAMES

    def test_the_picture_is_the_same_as_in_one_piece(self, flight):
        """What is not rendered would have been empty: in tiles the mosaic
        covers the ground it covers as a single image."""
        tiled, _log = self._run(flight, "tiled", alfs_max_tile_size=64)
        whole, _log = self._run(flight, "whole", alfs_max_tile_size=8192)

        assert len(_tiles(whole)) == 1
        (tiled_extent, tiled_lit), (whole_extent, whole_lit) = \
            _extent(tiled), _extent(whole)
        assert tiled_extent == pytest.approx(whole_extent, abs=0.5)
        assert tiled_lit == pytest.approx(whole_lit, rel=0.02)
        # and it is the frames' footprint: six frames a metre apart
        half_height = HEIGHT_AGL * math.tan(math.radians(FOVY / 2))
        assert whole_extent[3] - whole_extent[1] == pytest.approx(
            2 * half_height, abs=1.5)

    def test_a_tilted_flight_keeps_all_of_its_picture(self, tmp_path,
                                                      monkeypatch):
        """25 degrees of tilt: the frames land well away from the drone,
        on tiles that are not below it."""
        flight = small_flight(tmp_path, monkeypatch, rotation=(25.0, 0.0, 0.0))
        tiled, _log = self._run(flight, "tiled", alfs_max_tile_size=32)
        whole, _log = self._run(flight, "whole", alfs_max_tile_size=8192)

        (tiled_extent, tiled_lit), (whole_extent, whole_lit) = \
            _extent(tiled), _extent(whole)
        assert tiled_extent == pytest.approx(whole_extent, abs=0.5)
        assert tiled_lit == pytest.approx(whole_lit, rel=0.02)
