# -*- coding: utf-8 -*-
"""A reconstruction does not hold what it is not working on.

The 3D Flight Viewer's reconstruction read every keyframe of the range
before it started and kept them, as float images, until it was done: 11 MB
each at the default size - 14 GB for the 5 km of a survey flight at the
default spacing. Feature descriptors (2 MB per frame) and, with occlusion,
a mask of the whole surface grid per frame came on top.

Images are now read when a step needs them and a limited amount of them
kept; descriptors live while a frame's neighbours are matched; what a camera
sees is stored for the part of the grid it looks at. These tests run the
steps on views that hold nothing and compare them with the same steps on
views that hold everything.
"""
import numpy as np
import pytest

from bambi_wildlife_detection.core import flight_reconstruction as fr

from .test_flight_reconstruction import H, W, camera, render, surface_height

cv2 = pytest.importorskip("cv2")

POSITIONS = ((-12, -6), (-4, 4), (4, -5), (12, 5), (0, 0), (-8, 8), (8, -8))


@pytest.fixture(scope="module")
def scene():
    cameras = [camera(x, y) for x, y in POSITIONS]
    return cameras, [render(c) for c in cameras]


@pytest.fixture
def held(scene):
    """Views that hold their images - how it always was."""
    cameras, pictures = scene
    return [fr.View(i, c, p) for i, (c, p) in enumerate(zip(cameras, pictures))]


@pytest.fixture
def lazy(scene):
    """The same views holding nothing, with room for the fewest images."""
    cameras, pictures = scene
    views = fr.lazy_views(range(len(cameras)), cameras.__getitem__,
                          pictures.__getitem__, budget=1)
    return views, views[0]._images


# ---------------------------------------------------------------------------
# Images that are read when they are needed
# ---------------------------------------------------------------------------

class TestFrameImages:
    @staticmethod
    def _images(budget, **kwargs):
        read = []

        def picture(index):
            read.append(index)
            return np.full((10, 10), index, np.uint8)       # 100 bytes

        return fr.FrameImages(picture, budget, **kwargs), read

    def test_an_image_is_read_once_while_it_is_kept(self):
        images, read = self._images(1000)
        first = images.get(3)
        assert images.get(3) is first
        assert read == [3] and images.reads == 1

    def test_no_more_than_the_budget_is_kept(self):
        images, read = self._images(500, min_kept=1)
        for index in range(40):
            images.get(index)

        assert len(images) == 5
        assert images.peak_bytes <= 600           # the budget and one image
        assert images.peak_held <= 6

    def test_the_one_not_used_for_longest_is_dropped(self):
        images, read = self._images(300, min_kept=1)
        for index in (0, 1, 2):
            images.get(index)
        images.get(0)                   # 0 was used after 1 and 2
        images.get(3)                   # ... so 1 goes

        del read[:]
        images.get(0)
        images.get(2)
        assert read == []
        images.get(1)
        assert read == [1]

    def test_a_dropped_image_is_read_again(self):
        images, read = self._images(100, min_kept=1)
        images.get(0)
        images.get(1)
        assert images.get(0)[0, 0] == 0
        assert read == [0, 1, 0]

    def test_a_few_images_are_kept_whatever_the_budget(self):
        """A stereo pair and its neighbours are worked on together."""
        images, read = self._images(1)
        for index in (0, 1, 2, 3, 0, 1, 2, 3):
            images.get(index)
        assert read == [0, 1, 2, 3]
        assert fr.MIN_IMAGES_KEPT == 4

    def test_a_frame_that_cannot_be_read(self):
        images = fr.FrameImages(lambda index: None)
        assert images.try_get(7) is None
        with pytest.raises(RuntimeError, match="Frame 7 can no longer be read"):
            images.get(7)


class TestViews:
    def test_a_view_given_its_image_holds_it(self, scene):
        cameras, pictures = scene
        view = fr.View(4, cameras[0], pictures[0])
        assert view.image is pictures[0] and view.size == (W, H)

    def test_a_view_needs_an_image_or_somewhere_to_read_it(self, scene):
        with pytest.raises(ValueError):
            fr.View(0, scene[0][0])

    def test_a_lazy_view_knows_its_size_without_its_image(self, lazy):
        views, images = lazy
        before = images.reads
        assert [view.size for view in views] == [(W, H)] * len(views)
        assert images.reads == before

    def test_its_image_is_read_when_asked_for(self, lazy, scene):
        views, images = lazy
        _cameras, pictures = scene
        assert len(images) == fr.MIN_IMAGES_KEPT      # not all seven
        for view, picture in zip(views, pictures):
            np.testing.assert_array_equal(view.image, picture)

    def test_another_camera_does_not_read_the_image(self, lazy, scene):
        views, images = lazy
        before = images.reads
        moved = views[0].with_camera(scene[0][3])

        assert images.reads == before
        assert moved.camera is scene[0][3] and moved.index == views[0].index
        np.testing.assert_array_equal(moved.image, scene[1][0])

    def test_frames_that_cannot_be_read_are_left_out(self, scene):
        cameras, pictures = scene
        views = fr.lazy_views(
            range(len(cameras)), cameras.__getitem__,
            lambda index: None if index in (1, 5) else pictures[index])
        assert [view.index for view in views] == [0, 2, 3, 4, 6]

    def test_the_views_share_one_mask(self, scene):
        """It used to be read, resized and eroded once per keyframe."""
        cameras, pictures = scene
        asked = []

        def mask(width, height):
            asked.append((width, height))
            return np.ones((height, width), bool)

        views = fr.lazy_views(range(len(cameras)), cameras.__getitem__,
                              pictures.__getitem__, mask)
        assert asked == [(W, H)]
        assert all(view.mask is views[0].mask for view in views)

    def test_loading_reports_and_can_be_cancelled(self, scene):
        cameras, pictures = scene
        said = []
        fr.lazy_views(range(3), cameras.__getitem__, pictures.__getitem__,
                      progress=lambda text, share: said.append((text, share)))
        assert said == [("Loading frame 0", 0.0),
                        ("Loading frame 1", pytest.approx(1 / 3)),
                        ("Loading frame 2", pytest.approx(2 / 3))]
        with pytest.raises(fr.Cancelled):
            fr.lazy_views(range(3), cameras.__getitem__, pictures.__getitem__,
                          cancel=lambda: True)

    def test_a_long_range_holds_a_budget_not_the_range(self):
        """600 keyframes of a megabyte: 600 MB before, here 16."""
        megabyte = np.zeros((1024, 1024), np.uint8)
        camera_of = [camera(float(x), 0.0) for x in range(3)]
        views = fr.lazy_views(range(600), lambda i: camera_of[i % 3],
                              lambda i: megabyte.copy(), budget=16 << 20)
        images = views[0]._images

        assert len(views) == 600
        assert images.peak_bytes <= 17 << 20
        assert len(images) == 16


# ---------------------------------------------------------------------------
# The steps give the same result on views that hold nothing
# ---------------------------------------------------------------------------

class TestTheSameResult:
    def test_sparse_points(self, held, lazy):
        views, images = lazy
        expected = fr.sparse_points(held, neighbours=2, max_error=1.5)
        got = fr.sparse_points(views, neighbours=2, max_error=1.5)

        assert len(expected[0]) > 200
        for a, b in zip(expected, got):
            np.testing.assert_array_equal(a, b)
        assert images.peak_held <= fr.MIN_IMAGES_KEPT

    @pytest.mark.parametrize("from_images", [True, False])
    def test_the_stereo_surface(self, held, lazy, from_images):
        views, images = lazy
        grid = fr.GroundGrid([-24, -18], [24, 18], 0.5)
        dem = np.zeros((grid.height, grid.width))
        heights = np.arange(-2.0, 9.01, 0.25)

        expected = fr.stereo_surface(grid, dem, held, heights,
                                     from_images=from_images)
        got = fr.stereo_surface(grid, dem, views, heights,
                                from_images=from_images)

        np.testing.assert_array_equal(expected.offset, got.offset)
        np.testing.assert_array_equal(expected.views, got.views)
        assert images.peak_held <= fr.MIN_IMAGES_KEPT

    def test_the_texture(self, held, lazy):
        views, images = lazy
        coarse = fr.GroundGrid([-20, -15], [20, 15], 1.0)
        X, Y = coarse.xy()
        z = surface_height(X, Y).astype(np.float32)

        def ortho(vs):
            visibility = fr.visibility_masks(coarse, z, vs, None)
            return fr.texture_ortho(coarse, z, vs, texture_gsd=0.25,
                                    max_edge=400, visibility=visibility,
                                    rows_per_block=32)[1]

        np.testing.assert_array_equal(ortho(held), ortho(views))
        assert images.peak_held <= fr.MIN_IMAGES_KEPT

    def test_the_refined_poses(self, held, lazy):
        pytest.importorskip("scipy")
        views, images = lazy
        flat = fr.GroundGrid([-60, -60], [60, 60], 2.0)
        ground = fr.ground_sampler(flat, np.zeros((flat.height, flat.width)))

        expected, report = fr.refine_poses(held, ground=ground)
        got, same = fr.refine_poses(views, ground=ground)

        assert report.points > 100 and vars(same) == vars(report)
        for a, b in zip(expected, got):
            np.testing.assert_allclose(a.camera.position, b.camera.position)
            assert b._image is None                 # still holds nothing
        assert images.peak_held <= fr.MIN_IMAGES_KEPT


# ---------------------------------------------------------------------------
# Feature descriptors
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_descriptors_live_while_a_frames_neighbours_are_matched(self, held):
        features = fr._Features(held, 1500)
        tracks = fr._match_tracks(features, 2, 0.75, None)

        assert len(tracks) > 50
        assert features.peak_held <= 3              # a frame and its two
        assert all(features[k][1] is None for k in range(len(held) - 1))
        assert all(len(features[k][0]) > 100 for k in range(len(held)))

    def test_the_tracks_are_the_ones_of_all_features_at_once(self, held):
        at_once = fr._Features(held, 1500)
        everything = [at_once[k] for k in range(len(held))]

        assert fr._match_tracks(fr._Features(held, 1500), 2, 0.75, None) \
            == fr._match_tracks(everything, 2, 0.75, None)

    def test_sparse_points_do_not_hold_them_either(self, held, monkeypatch):
        made = []

        class Counted(fr._Features):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                made.append(self)

        monkeypatch.setattr(fr, "_Features", Counted)
        fr.sparse_points(held, neighbours=2)

        assert made[0].peak_held <= 3

    def test_features_are_worked_out_when_first_asked_for(self, lazy):
        views, images = lazy
        before = images.reads
        features = fr._Features(views, 500)
        assert images.reads == before
        points, descriptors = features[5]
        assert len(points) == len(descriptors) > 50
        features.done_with(5)
        assert features[5][1] is None and len(features[5][0]) == len(points)


# ---------------------------------------------------------------------------
# What a camera sees of the surface
# ---------------------------------------------------------------------------

class TestVisibility:
    @staticmethod
    def _seen():
        seen = np.zeros((400, 500), bool)
        seen[120:160, 300:340] = True
        seen[130, 310] = False
        return seen

    def test_it_stands_for_the_whole_array(self):
        seen = self._seen()
        visibility = fr.Visibility(seen)

        assert visibility.shape == (400, 500)
        np.testing.assert_array_equal(np.asarray(visibility), seen)

    def test_it_stores_what_the_camera_looks_at(self):
        seen = self._seen()
        assert fr.Visibility(seen).nbytes == 40 * 40
        assert seen.nbytes == 400 * 500

    def test_it_is_indexed_like_the_array(self):
        seen = self._seen()
        visibility = fr.Visibility(seen)
        rows = np.random.default_rng(1).integers(0, 400, (64, 70))
        cols = np.random.default_rng(2).integers(0, 500, (64, 70))

        np.testing.assert_array_equal(visibility[rows, cols], seen[rows, cols])
        assert visibility[np.array([130]), np.array([310])].tolist() == [False]
        assert visibility[np.array([131]), np.array([310])].tolist() == [True]

    def test_a_camera_that_sees_nothing(self):
        visibility = fr.Visibility(np.zeros((30, 40), bool))
        assert visibility.nbytes == 0
        assert not np.asarray(visibility).any()
        assert not visibility[np.arange(30), np.arange(30)].any()

    def test_the_masks_of_a_surface(self, held):
        coarse = fr.GroundGrid([-200, -200], [200, 200], 1.0)
        z = np.zeros((coarse.height, coarse.width), np.float32)
        masks = fr.visibility_masks(coarse, z, held, None)

        assert len(masks) == len(held) and masks[0].shape == z.shape
        assert all(np.asarray(mask).sum() > 1000 for mask in masks)
        # each camera looks at a corner of this grid, and that is stored
        assert sum(mask.nbytes for mask in masks) < 0.05 * len(held) * z.size
