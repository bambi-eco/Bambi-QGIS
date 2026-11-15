# -*- coding: utf-8 -*-
"""Animals only one camera saw, when the classifier reads both cameras.

A "matched" classifier takes the RGB and the thermal vector side by side -
twice the width of one. "Fall back to RGB / thermal" handed it the single
vector there was; the classifier refuses a vector of the wrong width, and
that refusal ended the whole classification run. Now such an animal goes to
the task's single-camera classifier, and the record of each frame names
which of the two called it.

(Stub heads, so no torch is needed.)
"""
import numpy as np
import pytest

from bambi_wildlife_detection import bambi_processing
from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import classification as cl

DIM = 4


class StubHead:
    """A classifier that refuses a vector of the wrong width, as the real
    ones do, and calls class 1 where the first value is above 2."""

    def __init__(self, dim, classes=("roe deer", "chamois"), path="m.pt"):
        self.feature_dim = dim
        self.classes = list(classes)
        self.path = path
        self.seen = 0
        self.closed = False

    def predict(self, features):
        array = np.asarray(features, dtype=np.float32)
        if array.shape[1] != self.feature_dim:
            raise cl.HeadError(
                f"expects {self.feature_dim}-d features but was given "
                f"{array.shape[1]}-d")
        self.seen += len(array)
        return [(int(row[0] > 2), 0.9) for row in array]

    def close(self):
        self.closed = True


@pytest.fixture
def resolver():
    """Thermal is the camera being classified: 10 has an RGB partner, 11
    and 12 were seen by the thermal camera alone."""
    return cl.FeatureResolver(
        primary="t",
        vectors_t={10: np.full(DIM, 1.0), 11: np.full(DIM, 5.0),
                   12: np.full(DIM, 1.0)},
        vectors_w={20: np.full(DIM, 3.0)},
        partner_of_primary={10: 20})


def _fallback(head, modality="thermal", log=None):
    return bambi_processing._SingleCameraHead(
        lambda: head, modality, "species", log)


def test_the_matched_classifier_is_only_given_pairs(resolver):
    """It was given the one vector there was - and refused it, which ended
    the run."""
    head = StubHead(2 * DIM)
    usable, calls = BambiProcessor()._predict_in_batches(
        head, resolver, [10, 11, 12], "matched", cl.UNMATCHED_THERMAL)
    assert usable == [10]
    assert head.seen == 1


def test_an_animal_one_camera_saw_goes_to_that_cameras_classifier(resolver):
    matched = StubHead(2 * DIM, path="species_matched.pt")
    thermal = StubHead(DIM, path="species_thermal.pt")
    via = {}
    fallback = _fallback(thermal)

    usable, calls = BambiProcessor()._predict_in_batches(
        matched, resolver, [10, 11, 12], "matched", cl.UNMATCHED_THERMAL,
        fallback=fallback, via=via)

    assert usable == [10, 11, 12]
    assert [index for index, _p in calls] == [1, 1, 0]   # 10: RGB 3.0 first
    assert (matched.seen, thermal.seen) == (1, 2)
    assert via == {11: ("thermal", "species_thermal.pt"),
                   12: ("thermal", "species_thermal.pt")}
    assert fallback.frames == 2


def test_the_two_classifiers_need_not_order_their_classes_alike(resolver):
    matched = StubHead(2 * DIM, classes=("roe deer", "chamois"))
    thermal = StubHead(DIM, classes=("chamois", "roe deer", "ibex"))
    usable, calls = BambiProcessor()._predict_in_batches(
        matched, resolver, [11, 12], "matched", cl.UNMATCHED_THERMAL,
        fallback=_fallback(thermal))
    # the thermal head's class 1 is "roe deer" - the matched head's class 0
    assert dict(zip(usable, (index for index, _p in calls))) == {11: 0, 12: 1}


def test_a_class_the_matched_classifier_does_not_know_is_no_call(resolver):
    matched = StubHead(2 * DIM, classes=("roe deer",))
    thermal = StubHead(DIM, classes=("roe deer", "ibex"))
    usable, _calls = BambiProcessor()._predict_in_batches(
        matched, resolver, [11, 12], "matched", cl.UNMATCHED_THERMAL,
        fallback=_fallback(thermal))
    assert usable == [12]


def test_falling_back_to_a_camera_that_did_not_see_it_skips_it(resolver):
    rgb = StubHead(DIM)
    usable, _calls = BambiProcessor()._predict_in_batches(
        StubHead(2 * DIM), resolver, [11, 12], "matched", cl.UNMATCHED_RGB,
        fallback=_fallback(rgb, "rgb"))
    assert usable == [] and rgb.seen == 0


def test_the_fallback_classifier_is_only_opened_when_needed(resolver):
    opened = []

    def open_head():
        opened.append(1)
        return StubHead(DIM)

    fallback = bambi_processing._SingleCameraHead(
        open_head, "thermal", "species")
    processor = BambiProcessor()
    processor._predict_in_batches(
        StubHead(2 * DIM), resolver, [10], "matched", cl.UNMATCHED_THERMAL,
        fallback=fallback)
    assert opened == []
    for _ in range(2):
        processor._predict_in_batches(
            StubHead(2 * DIM), resolver, [11], "matched",
            cl.UNMATCHED_THERMAL, fallback=fallback)
    assert opened == [1]


def test_no_classifier_to_fall_back_on_skips_and_says_so(resolver):
    log = []

    def open_head():
        raise FileNotFoundError("species_thermal.pt")

    fallback = bambi_processing._SingleCameraHead(
        open_head, "thermal", "species", log.append)
    usable, _calls = BambiProcessor()._predict_in_batches(
        StubHead(2 * DIM), resolver, [10, 11], "matched",
        cl.UNMATCHED_THERMAL, fallback=fallback)
    assert usable == [10]
    assert len(log) == 1 and "they are skipped" in log[0]
    fallback.get()
    assert len(log) == 1                    # said once


def test_closing_reports_what_the_fallback_classified(resolver):
    log = []
    thermal = StubHead(DIM)
    fallback = _fallback(thermal, log=log.append)
    BambiProcessor()._predict_in_batches(
        StubHead(2 * DIM), resolver, [11, 12], "matched",
        cl.UNMATCHED_THERMAL, fallback=fallback)
    fallback.close()
    assert thermal.closed
    assert any("2 frame(s) of animals only one camera saw" in line
               for line in log)


def test_single_camera_classifiers_are_untouched(resolver):
    thermal = StubHead(DIM)
    usable, _calls = BambiProcessor()._predict_in_batches(
        thermal, resolver, [10, 11, 12], "thermal", cl.UNMATCHED_SKIP)
    assert usable == [10, 11, 12]


class TestWhenThereIsAFallback:
    def _fallback(self, spec, modality_in="matched", unmatched="thermal"):
        log = []
        found = BambiProcessor()._single_camera_fallback(
            {}, "species", spec, modality_in, unmatched, "non_geo",
            log.append)
        return found, log

    def test_the_published_classifiers_have_one(self):
        found, _log = self._fallback({"model": "default"})
        assert found is not None and found.modality == "thermal"

    @pytest.mark.parametrize("modality_in, unmatched", [
        ("thermal", "rgb"), ("matched", "skip")])
    def test_none_is_needed(self, modality_in, unmatched):
        found, log = self._fallback({"model": "default"}, modality_in,
                                    unmatched)
        assert found is None and log == []

    def test_a_model_file_of_ones_own_has_none_and_says_so(self):
        found, log = self._fallback({"model": "custom", "path": "x.pt"})
        assert found is None
        assert "skipped" in log[0]

    def test_a_custom_implementation_is_handed_what_there_is(self):
        from bambi_wildlife_detection.core import custom_models

        found, log = self._fallback({"model": custom_models.SOURCE})
        assert found is None and log == []
