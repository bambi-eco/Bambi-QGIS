# -*- coding: utf-8 -*-
"""Reconstruction of the 3D Flight Viewer (core/flight_reconstruction.py).

Frames are ray-traced in NumPy over flat ground with a 5 m high block, so
the expected heights are known exactly.
"""
import math
import os

import numpy as np
import pytest

from bambi_wildlife_detection.core import flight_reconstruction as fr
from bambi_wildlife_detection.core import flight_viewer as fv

cv2 = pytest.importorskip("cv2")

BLOCK = (-10.0, 10.0, -8.0, 8.0, 5.0)        # x0, x1, y0, y1, height
W, H = 320, 256


def texture(x, y):
    """Smooth, aperiodic colour pattern on the ground (0..255)."""
    r = 128 + 50 * np.sin(0.9 * x + 0.3 * y) + 40 * np.sin(2.3 * y - 0.7 * x + 1)
    g = 128 + 45 * np.sin(1.7 * x - 1.1 * y + 2) + 35 * np.cos(3.1 * x + 0.4)
    b = 128 + 60 * np.cos(0.5 * x + 2.9 * y) + 30 * np.sin(4.1 * y + 1.3 * x)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255)


def surface_height(x, y):
    x0, x1, y0, y1, h = BLOCK
    return np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1), h, 0.0)


def camera(x, y, z=60.0, tilt_x=0.0):
    return fv.FrameCamera(fv.look_at([x, y, z], [x + tilt_x, y, 0], up=(0, 1, 0)),
                          fv.perspective(50, W / H, 0.1, 1000), 50.0, W / H)


def render(cam):
    """Ray-trace the top of the block and the ground (walls are ignored)."""
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    dirs = fv.pixel_rays(cam, W, H, np.column_stack([u.ravel(), v.ravel()]))
    o = cam.position
    t_top = (BLOCK[4] - o[2]) / dirs[:, 2]
    top = o + dirs * t_top[:, None]
    on_top = surface_height(top[:, 0], top[:, 1]) > 0
    t_ground = -o[2] / dirs[:, 2]
    ground = o + dirs * t_ground[:, None]
    hit = np.where(on_top[:, None], top, ground)
    return texture(hit[:, 0], hit[:, 1]).reshape(H, W, 3).astype(np.float32)


@pytest.fixture(scope="module")
def views():
    cams = [camera(x, y) for x, y in ((-12, -6), (-4, 4), (4, -5), (12, 5),
                                      (0, 0), (-8, 8), (8, -8))]
    return [fr.View(i, c, render(c)) for i, c in enumerate(cams)]


def test_projection_inverts_pixel_rays():
    cam = camera(3, -2, tilt_x=10)
    px = np.array([[10.5, 20.25], [160, 128], [300, 240]])
    dirs = fv.pixel_rays(cam, W, H, px)
    pts = cam.position + dirs * 50.0
    u, v, ok = fr.project_points(cam, W, H, pts)
    assert ok.all()
    np.testing.assert_allclose(np.column_stack([u, v]), px, atol=1e-6)


def test_keyframes_by_distance():
    pos = np.array([[i * 2.0, 0, 50] for i in range(20)])
    assert fr.select_keyframes(pos, range(20), 7.0) == [0, 4, 8, 12, 16, 19]
    assert fr.select_keyframes(pos, [], 5.0) == []


def test_grid_fitting_limits_the_cell_count():
    g = fr.GroundGrid.fitting([0, 0], [1000, 500], 0.1, max_cells=20000)
    assert g.width * g.height <= 20500
    X, Y = g.xy()
    assert Y[0, 0] > Y[-1, 0]               # row 0 is north


def test_texture_from_frames_matches_the_ground_pattern(views):
    grid = fr.GroundGrid([-20, -15], [20, 15], 0.5)
    X, Y = grid.xy()
    z = surface_height(X, Y)
    color, seen = fr.texture_from_frames(grid, z, views)
    assert (seen > 0).mean() > 0.9
    # Away from the block's edges, where the synthetic frames have no walls.
    x0, x1, y0, y1, _ = BLOCK
    edge = ((np.abs(np.abs(X) - x1) < 3) & (np.abs(Y) < y1 + 3)) | \
           ((np.abs(np.abs(Y) - y1) < 3) & (np.abs(X) < x1 + 3))
    ok = (seen > 0) & ~edge
    np.testing.assert_allclose(color[ok], texture(X[ok], Y[ok]), atol=12)


def test_texture_skips_cells_hidden_behind_the_surface():
    trimesh = pytest.importorskip("trimesh")
    grid = fr.GroundGrid([-10, -10], [10, 10], 1.0)
    X, Y = grid.xy()
    z = np.where((np.abs(X) < 2) & (np.abs(Y) < 10), 30.0, 0.0)   # a wall
    verts, faces, _ = fr.heightfield_mesh(grid, z)
    caster = trimesh.Trimesh(verts, faces, process=False)
    cam = fv.FrameCamera(fv.look_at([-40, 0, 20], [5, 0, 0], up=(0, 0, 1)),
                         fv.perspective(80, 1, 0.1, 1000), 80.0, 1.0)
    img = np.full((64, 64, 3), 200, np.float32)
    _, seen = fr.texture_from_frames(grid, z, [fr.View(0, cam, img)], caster)
    behind = (X > 4) & (np.abs(Y) < 3)
    assert seen[behind].sum() == 0
    _, seen_free = fr.texture_from_frames(grid, z, [fr.View(0, cam, img)], None)
    assert seen_free[behind].sum() > 0


def test_sparse_points_lie_on_block_and_ground(views):
    pts, cols, err = fr.sparse_points(views, neighbours=2, max_error=1.5)
    assert len(pts) > 200
    truth = surface_height(pts[:, 0], pts[:, 1])
    near = np.abs(pts[:, 2] - truth) < 0.5
    assert near.mean() > 0.85
    assert (pts[near, 2] > 4).sum() > 20 and (pts[near, 2] < 1).sum() > 20
    assert cols.dtype == np.uint8 and err.max() < 1.5


def test_heightfield_mesh_drops_unknown_cells():
    grid = fr.GroundGrid([0, 0], [3, 3], 1.0)
    z = np.zeros((3, 3))
    v, f, n = fr.heightfield_mesh(grid, z)
    assert len(v) == 9 and len(f) == 8
    z[1, 1] = np.nan
    _, f, _ = fr.heightfield_mesh(grid, z)
    assert len(f) == 2


def test_fill_offsets_takes_the_height_around_a_gap():
    off = np.array([[np.nan, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                   np.float32)
    filled = fr.fill_offsets(off, median_window=0)
    assert filled[0, 0] == pytest.approx(1.0) and filled[2, 2] == 1.0
    nothing = fr.fill_offsets(np.full((3, 3), np.nan, np.float32), 0)
    assert (nothing == 0).all()                     # nothing known: the DEM


def test_fill_holes_interpolates_and_respects_the_valid_area():
    z = np.full((5, 5), np.nan, np.float32)
    z[:, 0], z[:, 4] = 0.0, 4.0
    valid = np.ones((5, 5), bool)
    valid[0, 2] = False
    out = fr.fill_holes(z, valid)
    assert np.isnan(out[0, 2])
    assert 0.0 < out[2, 2] < 4.0
    assert np.all(np.diff(out[2]) >= 0)             # monotone across the gap


def test_resampling_heights_between_grids():
    coarse = fr.GroundGrid([0, 0], [10, 10], 1.0)
    X, Y = coarse.xy()
    z = (X + 2 * Y).astype(np.float32)
    fine = fr.GroundGrid([2, 2], [8, 8], 0.25)
    up = fr.upsample_heights(coarse, z, fine)
    FX, FY = fine.xy()
    np.testing.assert_allclose(up, FX + 2 * FY, atol=1e-4)


def test_colmap_model_round_trips_the_pose(tmp_path, views):
    fr.write_colmap_model(str(tmp_path), views[:2], ["a.png", "b.png"])
    lines = [l for l in open(tmp_path / "images.txt") if not l.startswith("#")]
    vals = [float(x) for x in lines[0].split()[1:8]]
    qw, qx, qy, qz, tx, ty, tz = vals
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    centre = -R.T @ np.array([tx, ty, tz])
    np.testing.assert_allclose(centre, views[0].camera.position, atol=1e-6)
    cams = open(tmp_path / "cameras.txt").read()
    assert "PINHOLE 320 256" in cams
    assert os.path.isfile(tmp_path / "points3D.txt")


def test_height_colors_range():
    c = fr.height_colors([0.0, 25.0, np.nan])
    np.testing.assert_allclose(c[0], [0.20, 0.30, 0.75])
    np.testing.assert_allclose(c[1], [0.85, 0.20, 0.15])
    np.testing.assert_allclose(c[2], [0.5, 0.5, 0.5])


def test_sample_image_handles_many_points():
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    u = np.tile([0.0, 1.5, 3.0], 20000)       # more than SHRT_MAX samples
    v = np.tile([0.0, 1.0, 2.0], 20000)
    out = fr.sample_image(img, u, v)
    assert out.shape == (60000,)
    np.testing.assert_allclose(out[:3], [0.0, 5.5, 11.0])


def test_masked_pixels_count_as_unseen(views):
    grid = fr.GroundGrid([-20, -15], [20, 15], 1.0)
    X, Y = grid.xy()
    z = surface_height(X, Y)
    v = views[4]                                    # camera above the origin
    masked = fr.View(v.index, v.camera, v.image,
                     mask=np.zeros(v.image.shape[:2], bool))
    _, seen = fr.texture_from_frames(grid, z, [masked])
    assert seen.sum() == 0
    half = np.zeros(v.image.shape[:2], bool)
    half[:, : v.image.shape[1] // 2] = True        # left half of the image
    _, seen_half = fr.texture_from_frames(
        grid, z, [fr.View(v.index, v.camera, v.image, mask=half)])
    _, seen_all = fr.texture_from_frames(grid, z, [v])
    assert 0.35 < seen_half.sum() / seen_all.sum() < 0.65


def _perturbed(views, fovy=45.0, pos_sigma=0.4, rot_sigma_deg=0.4, seed=3):
    """The same frames with telemetry-like errors and a wrong field of view."""
    rng = np.random.default_rng(seed)
    out = []
    for v in views:
        w, h = v.size
        K, R, t = fr.opencv_projection(v.camera, w, h)
        C = -R.T @ t + rng.normal(0, pos_sigma, 3)
        R = fr._rodrigues(np.radians(rng.normal(0, rot_sigma_deg, 3))) @ R
        f = (h / 2) / math.tan(math.radians(fovy) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        out.append(fr.View(v.index, fr.camera_from_opencv(K, R, -R @ C, w, h),
                           v.image))
    return out


def test_ground_sampler_interpolates_the_dem():
    g = fr.GroundGrid([0, 0], [10, 10], 1.0)
    X, Y = g.xy()
    f = fr.ground_sampler(g, X + 2 * Y)
    np.testing.assert_allclose(f(np.array([3.2, 7.0]), np.array([4.5, 2.0])),
                               [12.2, 11.0], atol=1e-9)
    assert np.isnan(f(np.array([-5.0]), np.array([5.0])))[0]


def test_camera_from_opencv_round_trips():
    cam = camera(3, -2, tilt_x=8)
    K, R, t = fr.opencv_projection(cam, W, H)
    back = fr.camera_from_opencv(K, R, t, W, H)
    np.testing.assert_allclose(back.view, cam.view, atol=1e-9)
    assert back.fovy == pytest.approx(cam.fovy)


def test_refine_poses_makes_heights_consistent_again(views):
    """Telemetry-like pose errors and a wrong field of view wreck the
    stereo heights; refining the poses from the images (held on the DEM)
    fixes them."""
    pytest.importorskip("scipy")
    bad = _perturbed(views, fovy=45.0, seed=4)
    flat = fr.GroundGrid([-60, -60], [60, 60], 2.0)
    ground = fr.ground_sampler(flat, np.zeros((flat.height, flat.width)))
    refined, report = fr.refine_poses(bad, refine_fov=True, ground=ground)
    assert report.points > 100
    assert report.rms_after < 1.0 < report.rms_before
    assert abs(report.fovy_after - 50.0) < abs(report.fovy_before - 50.0)

    grid = fr.GroundGrid([-24, -18], [24, 18], 0.5)
    X, Y = grid.xy()
    inside = (np.abs(X) < 7) & (np.abs(Y) < 5)
    outside = ((np.abs(X) > 13) | (np.abs(Y) > 11)) & (np.abs(X) < 20) \
        & (np.abs(Y) < 14)
    dem = np.zeros((grid.height, grid.width))

    def heights(vs):
        # Poses as given: no orientation from the images, no DEM scaling.
        off = fr.stereo_surface(grid, dem, vs, np.arange(-10.0, 16.01, 0.5),
                                from_images=False, dem_scale=False).offset
        return np.nanmedian(off[inside]), np.nanmedian(off[outside])

    block_bad, ground_bad = heights(bad)
    block, ground_h = heights(refined)
    assert abs(block_bad - 5.0) + abs(ground_bad) > 2.0   # broken before
    assert block == pytest.approx(5.0, abs=0.6)           # fixed after
    assert ground_h == pytest.approx(0.0, abs=0.4)


def test_texture_ortho_is_sharper_than_the_surface_grid(views):
    coarse = fr.GroundGrid([-20, -15], [20, 15], 1.0)
    X, Y = coarse.xy()
    z = surface_height(X, Y).astype(np.float32)
    fine, img = fr.texture_ortho(coarse, z, views, texture_gsd=0.1,
                                 max_edge=1000, rows_per_block=64)
    assert fine.gsd == pytest.approx(0.1) and img.shape == (fine.height, fine.width, 3)
    FX, FY = fine.xy()
    away = (np.abs(FX) > 13) & (np.abs(FX) < 18) & (np.abs(FY) < 10)
    err = np.abs(img[away].astype(float) - texture(FX[away], FY[away])).mean()
    assert err < 12
    vis = fr.visibility_masks(coarse, z, views, None)
    assert len(vis) == len(views) and vis[0].shape == z.shape
    capped, small = fr.texture_ortho(coarse, z, views, 0.01, max_edge=200)
    assert max(small.shape[:2]) <= 201


def test_lower_mode_finds_the_ground_under_vegetation():
    rng = np.random.default_rng(0)
    ground = rng.normal(0.0, 0.2, 600)
    trees = rng.uniform(3.0, 15.0, 900)               # more points on trees
    assert fr.lower_mode(np.concatenate([ground, trees]), 0.25) == \
        pytest.approx(0.0, abs=0.15)
    assert np.median(np.concatenate([ground, trees])) > 3   # median fails


def test_grid_size_survives_float_rounding():
    # 0.1 is not exact in binary: 25.6 / 0.1 = 256.00000000000003
    g = fr.GroundGrid([0.0, 0.0], [97.0, 25.6], 0.1)
    assert (g.width, g.height) == (970, 256)
    g = fr.GroundGrid([0.0, 0.0], [5.0, 5.0], 1.0, width=7, height=3)
    assert (g.width, g.height) == (7, 3)


@pytest.mark.parametrize("seed", range(6))
def test_texture_ortho_blocks_fit_any_extent(views, seed):
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-22, -15, 2)
    size = rng.uniform(5, 30, 2)
    coarse = fr.GroundGrid(lo, lo + size, rng.uniform(0.3, 1.1))
    z = np.zeros((coarse.height, coarse.width), np.float32)
    gsd = float(rng.choice([0.03, 0.05, 0.07, 0.1, 0.13]))
    fine, img = fr.texture_ortho(coarse, z, views, gsd, max_edge=600,
                                 rows_per_block=int(rng.integers(7, 64)))
    assert img.shape == (fine.height, fine.width, 3)


def test_surface_round_trip(tmp_path):
    g = fr.GroundGrid([0, 0], [10, 5], 0.5)
    z = np.random.default_rng(0).normal(size=(g.height, g.width)).astype(np.float32)
    tg = fr.GroundGrid([0, 0], [10, 5], 0.1)
    img = np.full((tg.height, tg.width, 3), 7, np.uint8)
    path = str(tmp_path / fr.SURFACE_FILE)
    fr.save_surface(path, g, z, z - 1, np.ones_like(z), np.full(z.shape, 3),
                    {"rgb": (tg, img)}, origin=(1.0, 2.0, 3.0))
    back = fr.load_surface(path)
    np.testing.assert_allclose(back["surface_z"], z)
    assert (back["grid"].width, back["grid"].height) == (g.width, g.height)
    assert back["tex_grid"].gsd == pytest.approx(0.1)
    assert back["rgb"].shape == img.shape and "thermal" not in back
    np.testing.assert_allclose(back["origin"], [1, 2, 3])


def test_nadir_distances_to_surface_and_dem(tmp_path):
    g = fr.GroundGrid([0, 0], [20, 20], 1.0)
    X, Y = g.xy()
    surface = fr.ground_sampler(g, np.where(X < 10, 15.0, 2.0))
    dem = fr.ground_sampler(g, np.full(X.shape, 2.0))
    rows = fr.nadir_distances([[5, 5, 60], [15, 5, 60], [50, 5, 60]], surface, dem)
    assert rows[0]["above_surface"] == pytest.approx(45) and rows[0]["above_dem"] == pytest.approx(58)
    assert rows[1]["above_surface"] == pytest.approx(58)
    assert rows[2]["surface"] is None and rows[2]["above_dem"] is None
    path = tmp_path / "n.csv"
    fr.write_nadir_csv(str(path), rows, ["a.png", "b.png", "c.png"],
                       origin=(1000, 2000, 300))
    lines = path.read_text().splitlines()
    assert lines[0].startswith("frame,imagefile,east")
    assert lines[1].split(",")[2:5] == ["1005.000", "2005.000", "360.000"]
    assert lines[3].endswith(",,,,")


def test_height_votes_take_the_median_per_cell():
    grid = fr.GroundGrid([0, 0], [4, 4], 1.0)
    votes = fr.HeightVotes(grid, np.zeros((4, 4)), np.arange(0, 10.01, 0.5))
    for h in (3.0, 3.1, 9.0):                       # two agree, one outlier
        assert votes.add([[0.5, 3.5, h], [0.6, 3.4, h]]) == 1
    votes.add([[2.5, 0.5, 50.0]])                   # outside the band: ignored
    res = votes.result()
    assert res.offset[0, 0] == pytest.approx(3.0)
    assert res.views[0, 0] == 3 and res.confidence[0, 0] == pytest.approx(2 / 3)
    assert np.isnan(res.offset[3, 2]) and np.isnan(res.offset[1, 1])
    top = fr.HeightVotes(grid, np.zeros((4, 4)), np.arange(0, 10.01, 0.5))
    top.add([[0.5, 3.5, 1.0], [0.6, 3.4, 6.0]], top=True)
    assert top.result().offset[0, 0] == pytest.approx(6.0)


@pytest.mark.parametrize("from_images", [False, True])
def test_stereo_surface_recovers_block_and_ground(views, from_images):
    grid = fr.GroundGrid([-24, -18], [24, 18], 0.5)
    X, Y = grid.xy()
    res = fr.stereo_surface(grid, np.zeros((grid.height, grid.width)), views,
                            np.arange(-2.0, 9.01, 0.25), neighbours=2,
                            from_images=from_images)
    inside = (np.abs(X) < 7) & (np.abs(Y) < 5)
    outside = ((np.abs(X) > 13) | (np.abs(Y) > 11)) & (np.abs(X) < 20) \
        & (np.abs(Y) < 14)
    assert np.isfinite(res.offset[inside]).mean() > 0.7
    assert np.nanmedian(res.offset[inside]) == pytest.approx(5.0, abs=0.4)
    assert np.nanmedian(res.offset[outside]) == pytest.approx(0.0, abs=0.3)
    total, used, from_img = res.extras["pairs"]
    assert used >= total // 2 and (from_img > 0) == from_images


def test_covered_area_closes_enclosed_gaps_only():
    m = np.zeros((20, 30), bool)
    m[3:17, 3:27] = True
    m[8:12, 10:20] = False                      # a hole inside
    c = fr.covered_area(m, margin=1)
    assert c[9, 15] and c[2, 15] and not c[0, 0] and not c[19, 29]
