# -*- coding: utf-8 -*-
"""Classification heads, frame selection and quorum voting.

The head tests build a real TorchScript module when torch is available, so the
``torch.jit.load`` contract — the ``(embedding, probs)`` tuple and the
``classes`` attribute — is exercised against the real thing rather than a
stand-in, without a 3 GB download. They skip where torch is absent; the voting
and frame-selection tests are pure and always run.
"""
from typing import List

import numpy as np
import pytest

from bambi_wildlife_detection.core import classification as cl
from bambi_wildlife_detection.core.classification import Vote

torch = pytest.importorskip("torch", reason="TorchScript head contract")


DIM = 8


class _StubHead(torch.nn.Module):
    """The published heads' shape: an embedding and class probabilities."""

    def __init__(self, dim=DIM, classes=("clear", "occluded"),
                 name_them=True):
        super().__init__()
        self.project = torch.nn.Linear(dim, 4)
        self.classify = torch.nn.Linear(4, len(classes))
        if name_them:
            # A plain assignment would not survive scripting — TorchScript
            # only serialises attributes it knows the type of, and an
            # attribute set on an already-scripted module is lost on save.
            # This is how the published heads carry their class list.
            self.classes = torch.jit.Attribute(list(classes), List[str])

    def forward(self, features):
        embedding = self.project(features)
        probabilities = torch.softmax(self.classify(embedding), dim=1)
        return embedding, probabilities


def _write_head(path, classes=("clear", "occluded"), dim=DIM,
                name_them=True):
    module = torch.jit.script(
        _StubHead(dim=dim, classes=classes, name_them=name_them))
    module.save(path)
    return path


@pytest.fixture
def head_path(tmp_path):
    return _write_head(str(tmp_path / "occlusion_rgb.pt"))


# ---------------------------------------------------------------------------
# Loading and class discovery
# ---------------------------------------------------------------------------

class TestHead:

    def test_classes_come_from_the_model(self, head_path):
        head = cl.Head(head_path, feature_dim=DIM)
        assert head.classes == ["clear", "occluded"]
        assert head.class_source == "classes"

    def test_a_head_without_names_is_probed_for_its_count(self, tmp_path):
        """Enough structure that the user only has to supply labels."""
        path = _write_head(str(tmp_path / "custom.pt"),
                           classes=("a", "b", "c"), name_them=False)
        head = cl.Head(path, feature_dim=DIM)
        assert head.class_source == "probe"
        assert head.classes == ["class 0", "class 1", "class 2"]

    def test_without_a_feature_dim_nothing_can_be_probed(self, tmp_path):
        path = _write_head(str(tmp_path / "custom.pt"), name_them=False)
        head = cl.Head(path, feature_dim=0)
        assert head.classes == []
        assert head.class_source == "unknown"

    def test_a_missing_file_is_reported_clearly(self, tmp_path):
        head = cl.Head(str(tmp_path / "nope.pt"))
        with pytest.raises(cl.HeadError, match="not found"):
            head.load()

    def test_a_file_that_is_not_a_model_is_reported(self, tmp_path):
        path = str(tmp_path / "broken.pt")
        with open(path, "wb") as handle:
            handle.write(b"not a torchscript archive")
        with pytest.raises(cl.HeadError, match="Could not load"):
            cl.Head(path).load()

    def test_loading_twice_loads_once(self, head_path, monkeypatch):
        head = cl.Head(head_path, feature_dim=DIM)
        head.load()
        loaded = []
        monkeypatch.setattr(torch.jit, "load",
                            lambda *a, **k: loaded.append(1))
        head.load()
        assert loaded == []


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPredict:

    def test_one_call_per_row(self, head_path):
        head = cl.Head(head_path, feature_dim=DIM)
        calls = head.predict(np.zeros((3, DIM), dtype=np.float32))
        assert len(calls) == 3
        for index, probability in calls:
            assert index in (0, 1)
            assert 0.0 <= probability <= 1.0

    def test_a_single_vector_is_accepted(self, head_path):
        head = cl.Head(head_path, feature_dim=DIM)
        assert len(head.predict(np.zeros(DIM, dtype=np.float32))) == 1

    def test_an_empty_batch_returns_nothing(self, head_path):
        head = cl.Head(head_path, feature_dim=DIM)
        assert head.predict(np.zeros((0, DIM), dtype=np.float32)) == []

    def test_the_wrong_feature_width_is_caught_with_advice(self, head_path):
        """The usual cause is a matched head fed one modality's features."""
        head = cl.Head(head_path, feature_dim=DIM)
        with pytest.raises(cl.HeadError, match="matched"):
            head.predict(np.zeros((2, DIM * 2), dtype=np.float32))

    def test_prediction_is_deterministic(self, head_path):
        head = cl.Head(head_path, feature_dim=DIM)
        features = np.arange(2 * DIM, dtype=np.float32).reshape(2, DIM)
        assert head.predict(features) == head.predict(features)


# ---------------------------------------------------------------------------
# Frame selection (§5.2a)
# ---------------------------------------------------------------------------

class TestVisibleDetections:

    def test_the_head_is_preferred_when_it_ran(self):
        chosen, source = cl.visible_detections(
            [1, 2, 3],
            head_labels={1: "clear", 2: "occluded", 3: "clear"},
            annotated={1: 1, 2: 0, 3: 0},
            clear_annotations=(0,))
        assert chosen == [1, 3]
        assert source == cl.FRAMES_FROM_HEAD

    def test_annotations_are_used_when_the_head_did_not_run(self):
        """A hand annotation is better evidence than a 78 %-accurate head."""
        chosen, source = cl.visible_detections(
            [1, 2, 3], head_labels={}, annotated={1: 0, 2: 1, 3: 0},
            clear_annotations=(0,))
        assert chosen == [1, 3]
        assert source == cl.FRAMES_FROM_ANNOTATIONS

    def test_with_no_labels_at_all_every_frame_votes(self):
        chosen, source = cl.visible_detections([1, 2, 3])
        assert chosen == [1, 2, 3]
        # Reported, not silent: voting over occluded frames while the UI says
        # "visible only" would be the wrong kind of quiet.
        assert source == cl.FRAMES_FROM_NOTHING

    def test_all_frames_ignores_occlusion_entirely(self):
        chosen, source = cl.visible_detections(
            [1, 2, 3], head_labels={1: "clear", 2: "occluded", 3: "occluded"},
            use_all=True)
        assert chosen == [1, 2, 3]
        assert source == cl.FRAMES_FROM_ALL

    def test_a_track_can_end_up_with_no_votable_frame(self):
        chosen, _source = cl.visible_detections(
            [1, 2], head_labels={1: "occluded", 2: "occluded"})
        assert chosen == []

    def test_a_frame_the_head_did_not_reach_does_not_vote(self):
        chosen, _source = cl.visible_detections(
            [1, 2, 3], head_labels={1: "clear"})
        assert chosen == [1]

    def test_custom_clear_labels_are_honoured(self):
        """A third-party head may name its classes anything."""
        chosen, _source = cl.visible_detections(
            [1, 2], head_labels={1: "visible", 2: "hidden"},
            clear_head_labels=("visible",))
        assert chosen == [1]

    def test_the_order_of_the_track_is_preserved(self):
        chosen, _source = cl.visible_detections(
            [5, 3, 9], head_labels={5: "clear", 3: "clear", 9: "clear"})
        assert chosen == [5, 3, 9]


# ---------------------------------------------------------------------------
# Quorum
# ---------------------------------------------------------------------------

class TestFeatureResolver:
    """Turning 'detection 4711 of the primary modality' into a head's input."""

    def _resolver(self, primary="t"):
        return cl.FeatureResolver(
            primary=primary,
            vectors_t={10: np.full(4, 1.0), 11: np.full(4, 2.0)},
            vectors_w={20: np.full(4, 3.0), 21: np.full(4, 4.0)},
            partner_of_primary=({10: 20} if primary == "t" else {20: 10}))

    def test_single_modality_from_the_primary_side(self):
        resolver = self._resolver("t")
        assert resolver.resolve(10, "thermal")[0] == 1.0

    def test_single_modality_from_the_partner_side(self):
        resolver = self._resolver("t")
        assert resolver.resolve(10, "rgb")[0] == 3.0

    def test_matched_concatenates_rgb_then_thermal(self):
        """The published heads were trained on [RGB, thermal], in that order,
        whichever modality happens to be primary here."""
        for primary in ("t", "w"):
            resolver = self._resolver(primary)
            anchor = 10 if primary == "t" else 20
            combined = resolver.resolve(anchor, "matched")
            assert len(combined) == 8
            assert combined[0] == 3.0        # RGB first
            assert combined[4] == 1.0        # then thermal

    def test_an_unmatched_detection_yields_nothing_by_default(self):
        """Zeros are not an option: the matched heads were trained on real
        pairs and cannot take a stand-in for a missing modality."""
        resolver = self._resolver("t")
        assert resolver.resolve(11, "matched") is None

    def test_an_unmatched_detection_can_fall_back(self):
        resolver = self._resolver("t")
        assert resolver.resolve(
            11, "matched", unmatched=cl.UNMATCHED_THERMAL)[0] == 2.0
        # There is no RGB side at all for this one, so that fallback declines.
        assert resolver.resolve(
            11, "matched", unmatched=cl.UNMATCHED_RGB) is None

    def test_a_detection_with_no_vector_yields_nothing(self):
        resolver = self._resolver("t")
        assert resolver.resolve(99, "thermal") is None

    def test_matched_doubles_the_expected_width(self):
        resolver = self._resolver("t")
        assert resolver.resolved_dim("matched", 1280) == 2560
        assert resolver.resolved_dim("rgb", 1280) == 1280

    def test_an_unknown_input_configuration_is_rejected(self):
        resolver = self._resolver("t")
        with pytest.raises(ValueError):
            resolver.resolve(10, "infrared")

    def test_an_unknown_primary_modality_is_rejected(self):
        with pytest.raises(ValueError):
            cl.FeatureResolver("rgb", {}, {})


class TestQuorumVote:

    def test_a_clear_majority_wins(self):
        calls = [(1, "male")] * 7 + [(0, "female_juvenile")] * 3
        result = cl.quorum_vote(calls)
        assert result == Vote("male", 1, 7, 10, 0.7)

    def test_the_paper_s_borderline_male_still_carries(self):
        """A2 D11: 109 of 150 clear frames. Many frames of a true male look
        female because the antler resolves only from some angles."""
        calls = [(1, "male")] * 109 + [(0, "female_juvenile")] * 41
        result = cl.quorum_vote(calls)
        assert result.label == "male"
        assert (result.votes, result.n) == (109, 150)

    def test_a_female_is_not_over_called(self):
        """A2 D6: 24 male votes of 93 — comfortably female."""
        calls = [(1, "male")] * 24 + [(0, "female_juvenile")] * 69
        assert cl.quorum_vote(calls).label == "female_juvenile"

    def test_exactly_half_does_not_clear_a_majority_quorum(self):
        calls = [(1, "male")] * 5 + [(0, "female_juvenile")] * 5
        assert cl.quorum_vote(calls, quorum=0.5) is None

    def test_a_stricter_quorum_abstains(self):
        calls = [(1, "male")] * 6 + [(0, "female_juvenile")] * 4
        assert cl.quorum_vote(calls, quorum=0.5).label == "male"
        assert cl.quorum_vote(calls, quorum=0.8) is None

    def test_too_few_frames_abstains(self):
        calls = [(1, "male")] * 2
        assert cl.quorum_vote(calls, min_frames=3) is None
        assert cl.quorum_vote(calls, min_frames=2).label == "male"

    def test_no_frames_at_all_abstains(self):
        assert cl.quorum_vote([]) is None

    def test_a_unanimous_track_is_reported_as_such(self):
        result = cl.quorum_vote([(0, "clear")] * 4)
        assert result.fraction == 1.0 and result.votes == result.n

    def test_ties_do_not_depend_on_processing_order(self):
        forward = cl.quorum_vote(
            [(0, "a"), (1, "b"), (2, "c")], quorum=0.0)
        backward = cl.quorum_vote(
            [(2, "c"), (1, "b"), (0, "a")], quorum=0.0)
        assert forward == backward

    def test_three_classes_are_handled(self):
        calls = ([(0, "roe deer")] * 2 + [(1, "red deer")] * 5 +
                 [(2, "wild boar")] * 1)
        result = cl.quorum_vote(calls)
        assert result.label == "red deer" and result.class_index == 1

    def test_the_margin_is_what_makes_a_call_reviewable(self):
        result = cl.quorum_vote([(1, "male")] * 106 +
                                [(0, "female_juvenile")] * 9)
        # "male 106/115" is the number an ecologist needs to judge it.
        assert (result.votes, result.n) == (106, 115)
        assert result.fraction == pytest.approx(106 / 115)
