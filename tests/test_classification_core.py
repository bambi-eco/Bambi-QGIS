# -*- coding: utf-8 -*-
"""Crop geometry and backbone plumbing.

The geometry is the part worth pinning: it decides what the classifier
actually sees, and a subtle error here degrades every downstream call without
ever raising.
"""
import numpy as np
import pytest

from bambi_wildlife_detection.core import classification as cl
from bambi_wildlife_detection.core.classification import CropConfig, Window
from bambi_wildlife_detection.core.track_matching import Affine


# ---------------------------------------------------------------------------
# Crop window
# ---------------------------------------------------------------------------

class TestCropWindow:

    def test_an_unpadded_square_box_is_itself(self):
        window = cl.crop_window(
            (10, 20, 30, 40), CropConfig(padding=0.0, letterbox=False))
        assert window == Window(10, 20, 30, 40)

    def test_padding_expands_on_every_side(self):
        window = cl.crop_window(
            (0, 0, 100, 100), CropConfig(padding=0.10, letterbox=False))
        # 10 % of 100 on each side.
        assert window == Window(-10, -10, 110, 110)

    def test_the_centre_never_moves(self):
        box = (10, 40, 50, 60)
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        for config in (CropConfig(), CropConfig(padding=0.5),
                       CropConfig(letterbox=False), CropConfig(padding=0.0)):
            window = cl.crop_window(box, config)
            assert window.centre == pytest.approx((cx, cy))

    def test_letterbox_squares_on_the_longer_side(self):
        """A deer from above is elongated; stretching it would change the
        proportions the sex head is reading."""
        window = cl.crop_window(
            (0, 0, 100, 20), CropConfig(padding=0.0, letterbox=True))
        assert window.width == pytest.approx(100)
        assert window.height == pytest.approx(100)

    def test_without_letterbox_the_aspect_is_kept(self):
        window = cl.crop_window(
            (0, 0, 100, 20), CropConfig(padding=0.0, letterbox=False))
        assert (window.width, window.height) == pytest.approx((100, 20))

    def test_a_window_may_leave_the_frame(self):
        """An animal at the edge still gets a centred crop; clamping here
        would shift it off-centre exactly when it is hardest to classify."""
        window = cl.crop_window((0, 0, 20, 20), CropConfig(padding=0.5))
        assert window.x1 < 0 and window.y1 < 0

    def test_a_degenerate_box_does_not_explode(self):
        window = cl.crop_window((50, 50, 50, 50), CropConfig())
        assert window.width == 0 and window.centre == (50, 50)

    def test_reversed_corners_are_tolerated(self):
        forward = cl.crop_window((10, 10, 50, 30), CropConfig())
        reversed_ = cl.crop_window((50, 30, 10, 10), CropConfig())
        assert forward.width == pytest.approx(reversed_.width)
        assert forward.centre == pytest.approx(reversed_.centre)

    def test_size_override_keeps_the_centre_and_takes_the_size(self):
        window = cl.crop_window(
            (100, 100, 120, 110), CropConfig(padding=0.0, letterbox=False),
            size_override=(50.0, 40.0))
        assert window.centre == pytest.approx((110, 105))
        assert (window.width, window.height) == pytest.approx((50, 40))


# ---------------------------------------------------------------------------
# Thermal-anchored sizing
# ---------------------------------------------------------------------------

class TestAnchoredSize:

    def test_a_thermal_box_maps_into_rgb_pixels(self):
        # Thermal is half the RGB scale, so one thermal pixel is two RGB ones.
        affine = Affine.scaling(0.5, 0.5)
        size = cl.anchored_size((0, 0, 40, 20), affine.inverse_scale())
        assert size == pytest.approx((80.0, 40.0))

    def test_rotation_does_not_shrink_the_scale(self):
        """Taking the scale from `a` and `d` alone would return their cosine."""
        import math

        angle = math.radians(30)
        affine = Affine(math.cos(angle), -math.sin(angle),
                        math.sin(angle), math.cos(angle), 0.0, 0.0)
        sx, sy = affine.inverse_scale()
        assert sx == pytest.approx(1.0)
        assert sy == pytest.approx(1.0)

    def test_the_identity_leaves_a_box_alone(self):
        size = cl.anchored_size((0, 0, 30, 10), Affine.identity().inverse_scale())
        assert size == pytest.approx((30.0, 10.0))

    def test_a_degenerate_affine_does_not_divide_by_zero(self):
        affine = Affine(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert affine.inverse_scale() == (1.0, 1.0)

    def test_the_stored_affine_round_trips(self):
        original = Affine(0.8, 0.05, -0.05, 0.8, 30.0, -12.0)
        assert Affine.from_json(original.as_json()) == original

    def test_an_absent_affine_reads_as_the_identity(self):
        assert Affine.from_json(None) == Affine.identity()


# ---------------------------------------------------------------------------
# Cutting pixels
# ---------------------------------------------------------------------------

def _image(width=100, height=80):
    """A gradient, so a misplaced crop is visible in the values."""
    ramp = np.arange(width, dtype=np.uint8)
    return np.repeat(np.tile(ramp, (height, 1))[:, :, None], 3, axis=2)


class TestExtractCrop:

    def test_a_crop_has_the_requested_size(self):
        crop = cl.extract_crop(_image(), Window(10, 10, 50, 50), 224)
        assert crop.shape == (224, 224, 3)

    def test_an_interior_crop_keeps_the_image_content(self):
        image = _image()
        crop = cl.extract_crop(image, Window(0, 0, 100, 80), 100)
        # Left edge dark, right edge bright — the gradient survived.
        assert crop[50, 0, 0] < crop[50, -1, 0]

    def test_out_of_bounds_regions_are_filled_not_clipped(self):
        crop = cl.extract_crop(_image(), Window(-40, -40, 40, 40), 80)
        # The top-left quadrant lies outside the image entirely.
        assert crop[5, 5, 0] == cl.EDGE_FILL

    def test_the_fill_is_grey_rather_than_black(self):
        """Black would put a hard edge round an animal at the frame border,
        which the backbone would happily encode as a feature."""
        assert 80 < cl.EDGE_FILL < 160

    def test_a_window_entirely_outside_yields_only_fill(self):
        crop = cl.extract_crop(_image(), Window(500, 500, 560, 560), 32)
        assert (crop == cl.EDGE_FILL).all()

    def test_an_animal_at_the_edge_stays_centred(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[0:10, 0:10] = 255                    # animal in the corner
        crop = cl.extract_crop(image, Window(-10, -10, 15, 15), 50)
        # Its centre (5, 5) maps to the middle of a window centred on (2.5, 2.5).
        assert crop[25, 25, 0] == 255

    def test_a_single_channel_frame_becomes_three(self):
        grey = np.zeros((10, 10), dtype=np.uint8)
        assert cl.to_rgb(grey).shape == (10, 10, 3)

    def test_an_alpha_channel_is_dropped(self):
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        assert cl.to_rgb(rgba).shape == (10, 10, 3)

    def test_an_rgb_frame_is_untouched(self):
        rgb = _image()
        assert cl.to_rgb(rgb) is rgb


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class TestBatching:

    def test_splits_into_full_and_partial_chunks(self):
        assert [list(b) for b in cl.batched(list(range(5)), 2)] == \
            [[0, 1], [2, 3], [4]]

    def test_an_empty_sequence_yields_nothing(self):
        assert list(cl.batched([], 4)) == []

    @pytest.mark.parametrize("size", [0, -1])
    def test_a_nonsense_batch_size_still_makes_progress(self, size):
        # Zero would loop forever; one item at a time is slow but finishes.
        assert [list(b) for b in cl.batched([1, 2], size)] == [[1], [2]]

    def test_grouping_by_frame_keeps_every_detection(self):
        rows = [{"frame": 1, "detection_id": 1}, {"frame": 2, "detection_id": 2},
                {"frame": 1, "detection_id": 3}]
        grouped = cl.group_by_frame(rows)
        assert set(grouped) == {1, 2}
        assert len(grouped[1]) == 2


# ---------------------------------------------------------------------------
# Device and error messages
# ---------------------------------------------------------------------------

class TestDevice:

    def test_cpu_is_honoured_without_torch(self):
        # Selecting CPU must not require importing torch at all.
        assert cl.resolve_device("cpu") == "cpu"

    def test_missing_torch_is_reported_usefully(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(cl.BackboneError) as caught:
            cl.resolve_device("auto")
        assert "Dependency Manager" in str(caught.value)

    def test_half_precision_is_off_on_cpu(self, monkeypatch):
        monkeypatch.setattr(cl, "resolve_device", lambda pref: "cpu")
        backbone = cl.Backbone(device="cpu", fp16=True)
        # fp16 on CPU is slower, not faster.
        assert backbone.fp16 is False

    def test_a_gated_failure_points_at_the_access_check(self):
        message = cl._load_failure_message(
            "facebook/x", RuntimeError("401 Client Error: gated repo"))
        assert "Request access" in message
        assert "Check access" in message
        assert "huggingface.co/facebook/x" in message

    def test_a_network_failure_says_so(self):
        message = cl._load_failure_message(
            "facebook/x", OSError("Failed to resolve huggingface.co"))
        assert "network" in message.lower()

    def test_an_unknown_failure_still_names_the_model(self):
        message = cl._load_failure_message("facebook/x", ValueError("odd"))
        assert "facebook/x" in message and "odd" in message

    def test_the_dim_defaults_to_the_published_width(self):
        assert cl.Backbone(device="cpu").dim == 1280
