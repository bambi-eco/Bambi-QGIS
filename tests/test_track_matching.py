# -*- coding: utf-8 -*-
"""Cross-modal track matching (paper §3.2).

The affine is bootstrapped from data rather than given, so the tests are built
around a known ground-truth transform: synthetic RGB detections are pushed
through it to make the thermal side, and the matcher has to recover both the
transform and which track is which.
"""
import math

import pytest

from bambi_wildlife_detection.core import track_matching as tm
from bambi_wildlife_detection.core.track_matching import (
    Affine, Candidate, MatchConfig,
)

# The transform every fixture is built from: a mild zoom, a small rotation and
# an offset — the kind of relationship two lenses on one airframe have.
TRUTH = Affine(a=0.80, b=0.05, c=-0.05, d=0.80, tx=30.0, ty=-12.0)


def _rows(track_id, frames, x0, y0, dx=4.0, dy=2.0, confidence=0.8,
          first_id=0, curve=6.0):
    """A track of boxes, as store rows.

    The path is deliberately *not* a straight line: an affine is unconstrained
    perpendicular to a line, so a perfectly straight track carries no
    information about the transform across it. Real animals wander, and the
    matcher rightly refuses to fit collinear points.
    """
    rows = []
    for i, frame in enumerate(frames):
        cx = x0 + dx * i + curve * math.sin(i * 0.7)
        cy = y0 + dy * i + curve * math.cos(i * 0.9)
        rows.append({
            "detection_id": first_id + i,
            "track_id": track_id,
            "frame": frame,
            "x1": cx - 5, "y1": cy - 5, "x2": cx + 5, "y2": cy + 5,
            "confidence": confidence,
        })
    return rows


def _through(rows, affine=TRUTH, first_id=1000, jitter=0.0):
    """Map RGB rows onto the thermal side using *affine*."""
    out = []
    for i, row in enumerate(rows):
        cx = (row["x1"] + row["x2"]) / 2
        cy = (row["y1"] + row["y2"]) / 2
        mx, my = affine.apply(cx, cy)
        mx += jitter * ((i % 3) - 1)
        my += jitter * ((i % 2) - 0.5)
        out.append({
            "detection_id": first_id + i,
            "track_id": row["track_id"],
            "frame": row["frame"],
            "x1": mx - 4, "y1": my - 4, "x2": mx + 4, "y2": my + 4,
            "confidence": row["confidence"],
        })
    return out


def _identity_map(frames):
    return {f: f for f in frames}


# ---------------------------------------------------------------------------
# Fitting the affine
# ---------------------------------------------------------------------------

class TestFitAffine:

    def test_recovers_a_known_transform_exactly(self):
        points = [(10.0, 20.0), (300.0, 40.0), (150.0, 260.0), (500.0, 500.0)]
        pairs = [(p, TRUTH.apply(*p)) for p in points]
        fitted = tm.fit_affine(pairs)

        for got, want in zip(fitted, TRUTH):
            assert got == pytest.approx(want, abs=1e-9)

    def test_is_robust_to_small_noise(self):
        points = [(x * 37.0 % 900, x * 53.0 % 700) for x in range(1, 40)]
        pairs = []
        for i, p in enumerate(points):
            qx, qy = TRUTH.apply(*p)
            pairs.append((p, (qx + (i % 3 - 1) * 0.5, qy + (i % 2 - 0.5))))

        fitted = tm.fit_affine(pairs)
        assert tm.affine_rmse(fitted, pairs) < 1.0
        assert fitted.a == pytest.approx(TRUTH.a, abs=0.02)
        assert fitted.tx == pytest.approx(TRUTH.tx, abs=2.0)

    def test_too_few_points_is_unfittable(self):
        assert tm.fit_affine([]) is None
        assert tm.fit_affine([((0, 0), (1, 1)), ((1, 1), (2, 2))]) is None

    def test_collinear_points_are_rejected(self):
        """An affine through one line is unconstrained across it, and would
        send every off-line detection somewhere arbitrary."""
        points = [(float(t), 2.0 * float(t)) for t in range(10)]
        pairs = [(p, TRUTH.apply(*p)) for p in points]
        assert tm.fit_affine(pairs) is None

    def test_coincident_points_are_rejected(self):
        pairs = [((5.0, 5.0), (1.0, 1.0))] * 8
        assert tm.fit_affine(pairs) is None

    def test_rmse_of_a_perfect_fit_is_zero(self):
        points = [(1.0, 2.0), (300.0, 4.0), (5.0, 600.0)]
        pairs = [(p, TRUTH.apply(*p)) for p in points]
        assert tm.affine_rmse(TRUTH, pairs) == pytest.approx(0.0, abs=1e-9)

    def test_rmse_without_pairs_is_infinite(self):
        assert tm.affine_rmse(TRUTH, []) == float("inf")

    def test_json_form_round_trips(self):
        linear, translation = TRUTH.as_json()
        assert linear == [[TRUTH.a, TRUTH.b], [TRUTH.c, TRUTH.d]]
        assert translation == [TRUTH.tx, TRUTH.ty]


# ---------------------------------------------------------------------------
# Seeding and estimation
# ---------------------------------------------------------------------------

class TestEstimateAffine:

    def test_seeds_only_from_unambiguous_frames(self):
        # Frames 0-2 hold one animal each side; frame 3 holds two, so pairing
        # it would require the answer we are trying to find.
        rgb = _rows(1, [0, 1, 2, 3], 100.0, 100.0)
        rgb += _rows(2, [3], 400.0, 400.0, first_id=50)
        thermal = _through(rgb)

        seeds = tm.seed_pairs(tm.to_detections(thermal),
                              tm.to_detections(rgb), _identity_map(range(4)))
        assert len(seeds) == 3

    def test_recovers_the_transform_from_a_flight(self):
        rgb = _rows(1, range(12), 100.0, 120.0)
        rgb += _rows(2, range(12), 500.0, 300.0, dx=-3.0, dy=5.0, first_id=100)
        thermal = _through(rgb)

        affine, rmse, count = tm.estimate_affine(
            tm.to_detections(thermal), tm.to_detections(rgb),
            _identity_map(range(12)))

        assert rmse < 0.01
        assert count >= 12
        assert affine.a == pytest.approx(TRUTH.a, abs=0.01)
        assert affine.ty == pytest.approx(TRUTH.ty, abs=0.5)

    def test_a_herd_with_no_unambiguous_frame_still_registers(self):
        """Several animals in *every* frame leaves nothing to seed from.

        This is the normal case on the paper's flights, not an edge case, so
        the estimator has to start from the frame-size scale and iterate rather
        than give up.
        """
        rgb = _rows(1, range(10), 100.0, 120.0)
        rgb += _rows(2, range(10), 600.0, 400.0, dx=-2.0, dy=3.0, first_id=100)
        rgb += _rows(3, range(10), 300.0, 700.0, dx=1.0, dy=-4.0, first_id=200)
        thermal = _through(rgb)
        detections_t = tm.to_detections(thermal)
        detections_w = tm.to_detections(rgb)
        frame_map = _identity_map(range(10))

        assert tm.seed_pairs(detections_t, detections_w, frame_map) == []

        affine, rmse, count = tm.estimate_affine(
            detections_t, detections_w, frame_map,
            frame_size_t=(1024, 1024), frame_size_w=(1024, 1024))

        assert rmse < 0.01
        assert count == 30
        assert affine.a == pytest.approx(TRUTH.a, abs=0.01)

    def test_the_starting_point_is_reported(self):
        rgb = _rows(1, range(10), 100.0, 120.0)
        thermal = _through(rgb)
        logged = []
        tm.estimate_affine(tm.to_detections(thermal), tm.to_detections(rgb),
                           _identity_map(range(10)), log_fn=logged.append)
        assert any("unambiguous frames" in line for line in logged)

    def test_one_perfectly_straight_track_cannot_register(self):
        """An affine is unconstrained perpendicular to a line.

        A lone animal walking dead straight genuinely carries no information
        about the transform across its path, so the honest answer is an
        infinite residual and a log line — not a confident wrong transform
        that would then mismatch everything.
        """
        rgb = _rows(1, range(20), 100.0, 120.0, curve=0.0)
        thermal = _through(rgb)
        logged = []

        affine, rmse, _ = tm.estimate_affine(
            tm.to_detections(thermal), tm.to_detections(rgb),
            _identity_map(range(20)), log_fn=logged.append)

        assert rmse == float("inf")
        assert affine == Affine.identity()
        assert any("nothing could be fitted" in line for line in logged)

    def test_a_second_track_is_enough_to_register_straight_paths(self):
        """Two straight tracks on different headings span the plane again."""
        rgb = _rows(1, range(20), 100.0, 120.0, dx=5.0, dy=1.0, curve=0.0)
        rgb += _rows(2, range(20), 600.0, 500.0, dx=-1.0, dy=6.0, curve=0.0,
                     first_id=100)
        thermal = _through(rgb)

        _affine, rmse, _ = tm.estimate_affine(
            tm.to_detections(thermal), tm.to_detections(rgb),
            _identity_map(range(20)))
        assert rmse < 0.01

    def test_falls_back_to_frame_size_and_says_so(self):
        logged = []
        affine, rmse, _ = tm.estimate_affine(
            [], [], {}, frame_size_t=(640, 512), frame_size_w=(1280, 1024),
            log_fn=logged.append)

        assert affine.a == pytest.approx(0.5)
        assert affine.d == pytest.approx(0.5)
        assert rmse == float("inf")
        # A silently bad affine yields zero matches, which looks exactly like
        # "there were no animals" — so it has to be said out loud.
        assert any("nothing could be fitted" in line for line in logged)

    def test_fallback_without_frame_sizes_is_the_identity(self):
        affine, _, _ = tm.estimate_affine([], [], {})
        assert affine == Affine.identity()


# ---------------------------------------------------------------------------
# Track costs
# ---------------------------------------------------------------------------

class TestCandidates:

    def test_median_distance_of_a_true_pair_is_near_zero(self):
        rgb = _rows(1, range(10), 100.0, 120.0)
        thermal = _through(rgb)
        found = tm.candidates(tm.to_detections(thermal),
                              tm.to_detections(rgb),
                              _identity_map(range(10)), TRUTH)

        assert len(found) == 1
        assert found[0].shared == 10
        assert found[0].median_dist == pytest.approx(0.0, abs=1e-6)

    def test_a_single_bad_frame_does_not_break_a_pair(self):
        """The median is chosen over the mean precisely for this."""
        rgb = _rows(1, range(11), 100.0, 120.0)
        thermal = _through(rgb)
        thermal[5]["x1"] += 900        # one badly placed box
        thermal[5]["x2"] += 900

        found = tm.candidates(tm.to_detections(thermal),
                              tm.to_detections(rgb),
                              _identity_map(range(11)), TRUTH)
        assert found[0].median_dist < 1.0

    def test_every_co_occurring_pair_is_considered(self):
        rgb = _rows(1, range(10), 100.0, 120.0)
        rgb += _rows(2, range(10), 300.0, 200.0, first_id=100)
        thermal = _through(rgb)

        found = tm.candidates(tm.to_detections(thermal),
                              tm.to_detections(rgb),
                              _identity_map(range(10)), TRUTH)
        assert len(found) == 4      # 2 x 2, the wrong pairings included

    def test_tracks_that_never_co_occur_are_not_candidates(self):
        rgb = _rows(1, range(0, 5), 100.0, 120.0)
        thermal = _through(_rows(2, range(20, 25), 100.0, 120.0),
                           first_id=2000)

        found = tm.candidates(tm.to_detections(thermal),
                              tm.to_detections(rgb),
                              _identity_map(range(30)), TRUTH)
        assert found == []

    def test_mean_confidence_is_per_track(self):
        rgb = _rows(1, range(4), 100.0, 120.0, confidence=0.9)
        thermal = _through(rgb)
        for row in thermal:
            row["confidence"] = 0.3

        found = tm.candidates(tm.to_detections(thermal),
                              tm.to_detections(rgb),
                              _identity_map(range(4)), TRUTH)
        assert found[0].conf_w == pytest.approx(0.9)
        assert found[0].conf_t == pytest.approx(0.3)

    def test_frame_correspondence_is_carried_through(self):
        rgb = _rows(1, [0, 1, 2], 100.0, 120.0)
        thermal = _through(rgb, first_id=7000)
        # Thermal frame 0 was taken at the same moment as RGB frame 10.
        shifted = [dict(row, frame=row["frame"] + 10) for row in rgb]

        found = tm.candidates(tm.to_detections(thermal),
                              tm.to_detections(shifted),
                              {0: 10, 1: 11, 2: 12}, TRUTH)
        pairs = found[0].pairs
        assert [(p["frame_t"], p["frame_w"]) for p in pairs] == \
            [(0, 10), (1, 11), (2, 12)]
        assert pairs[0]["detection_id_t"] == 7000


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _candidate(**overrides):
    values = dict(track_id_t=1, track_id_w=11, shared=10, median_dist=5.0,
                  conf_t=0.5, conf_w=0.5, pairs=[])
    values.update(overrides)
    return Candidate(**values)


class TestGates:

    def test_a_good_candidate_passes(self):
        assert tm.passes_gates(_candidate(), MatchConfig())

    def test_too_few_shared_frames_is_rejected(self):
        assert not tm.passes_gates(_candidate(shared=7), MatchConfig())
        assert tm.passes_gates(_candidate(shared=8), MatchConfig())

    def test_distance_gate_is_exclusive(self):
        config = MatchConfig(gate_px=28.0)
        assert tm.passes_gates(_candidate(median_dist=27.9), config)
        assert not tm.passes_gates(_candidate(median_dist=28.0), config)

    def test_the_gate_is_configurable(self):
        wide = MatchConfig(gate_px=60.0)
        assert tm.passes_gates(_candidate(median_dist=50.0), wide)
        assert not tm.passes_gates(_candidate(median_dist=50.0), MatchConfig())

    def test_low_confidence_on_either_side_is_rejected(self):
        assert not tm.passes_gates(_candidate(conf_t=0.1), MatchConfig())
        assert not tm.passes_gates(_candidate(conf_w=0.1), MatchConfig())

    def test_defaults_match_the_paper(self):
        config = MatchConfig()
        assert config.min_shared == 8
        assert config.gate_px == 28.0
        assert config.min_confidence == 0.20

    def test_rejection_reasons_are_counted_for_the_log(self):
        counts = tm.rejection_reasons([
            _candidate(shared=2),
            _candidate(median_dist=99.0),
            _candidate(conf_t=0.05),
            _candidate(),
        ], MatchConfig())
        assert counts == {"shared_frames": 1, "distance": 1, "confidence": 1}


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

class TestAssign:

    def test_assignment_is_one_to_one(self):
        found = tm.assign([
            _candidate(track_id_t=1, track_id_w=11, median_dist=1.0),
            _candidate(track_id_t=1, track_id_w=12, median_dist=2.0),
            _candidate(track_id_t=2, track_id_w=11, median_dist=3.0),
            _candidate(track_id_t=2, track_id_w=12, median_dist=4.0),
        ])

        assert len(found) == 2
        assert {c.track_id_t for c in found} == {1, 2}
        assert {c.track_id_w for c in found} == {11, 12}

    def test_the_global_optimum_beats_the_greedy_choice(self):
        """Greedy takes 1-11 at 1.0 and is then forced into 2-12 at 50.0
        (total 51); the optimum pairs 1-12 and 2-11 for 22."""
        found = tm.assign([
            _candidate(track_id_t=1, track_id_w=11, median_dist=1.0),
            _candidate(track_id_t=1, track_id_w=12, median_dist=2.0),
            _candidate(track_id_t=2, track_id_w=11, median_dist=20.0),
            _candidate(track_id_t=2, track_id_w=12, median_dist=50.0),
        ], MatchConfig(gate_px=100.0))

        assert {(c.track_id_t, c.track_id_w) for c in found} == {(1, 12), (2, 11)}

    def test_gated_pairs_are_never_assigned(self):
        """A rectangular problem forces every row to take something, so the
        gate has to be re-applied after the assignment."""
        found = tm.assign([
            _candidate(track_id_t=1, track_id_w=11, median_dist=1.0),
            _candidate(track_id_t=2, track_id_w=12, median_dist=999.0),
        ])
        assert [(c.track_id_t, c.track_id_w) for c in found] == [(1, 11)]

    def test_an_unmatched_track_is_simply_left_out(self):
        found = tm.assign([
            _candidate(track_id_t=1, track_id_w=11, median_dist=1.0),
            _candidate(track_id_t=1, track_id_w=12, median_dist=2.0),
        ])
        assert len(found) == 1

    def test_nothing_admissible_yields_nothing(self):
        assert tm.assign([_candidate(shared=1)]) == []
        assert tm.assign([]) == []

    def test_results_come_back_closest_first(self):
        found = tm.assign([
            _candidate(track_id_t=1, track_id_w=11, median_dist=9.0),
            _candidate(track_id_t=2, track_id_w=12, median_dist=1.0),
        ])
        assert [c.median_dist for c in found] == [1.0, 9.0]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class TestMatchTracks:

    def test_a_two_animal_flight_matches_correctly(self):
        rgb = _rows(1, range(15), 100.0, 120.0)
        rgb += _rows(2, range(15), 600.0, 400.0, dx=-3.0, dy=2.0, first_id=100)
        thermal = _through(rgb, jitter=0.4)

        result = tm.match_tracks(thermal, rgb, _identity_map(range(15)))

        assert result["affine_rmse"] < 2.0
        assert {(m["track_id_t"], m["track_id_w"])
                for m in result["matches"]} == {(1, 1), (2, 2)}
        assert all(m["shared"] == 15 for m in result["matches"])

    def test_output_is_ready_for_the_store(self):
        rgb = _rows(1, range(12), 100.0, 120.0)
        thermal = _through(rgb)
        match = tm.match_tracks(thermal, rgb, _identity_map(range(12)))

        row = match["matches"][0]
        assert set(row) >= {"track_id_t", "track_id_w", "shared",
                            "median_dist", "conf_t", "conf_w", "pairs"}
        assert set(row["pairs"][0]) == {"frame_t", "frame_w",
                                        "detection_id_t", "detection_id_w",
                                        "dist"}

    def test_a_short_track_is_not_confirmed(self):
        """Confirmation is what keeps phantoms out of the census."""
        rgb = _rows(1, range(4), 100.0, 120.0)
        thermal = _through(rgb)
        result = tm.match_tracks(thermal, rgb, _identity_map(range(4)))
        assert result["matches"] == []
        assert result["rejected"] == 1

    def test_a_low_confidence_track_is_not_confirmed(self):
        rgb = _rows(1, range(15), 100.0, 120.0, confidence=0.05)
        thermal = _through(rgb)
        result = tm.match_tracks(thermal, rgb, _identity_map(range(15)))
        assert result["matches"] == []

    def test_a_single_modality_track_has_no_partner(self):
        rgb = _rows(1, range(15), 100.0, 120.0)
        rgb_only = _rows(2, range(15), 800.0, 500.0, first_id=100)
        thermal = _through(rgb)      # track 2 was never seen in thermal

        result = tm.match_tracks(thermal, rgb + rgb_only,
                                 _identity_map(range(15)))
        assert [m["track_id_w"] for m in result["matches"]] == [1]

    def test_frames_without_a_counterpart_are_skipped(self):
        """At the ends of a flight one camera is often not yet recording."""
        rgb = _rows(1, range(15), 100.0, 120.0)
        thermal = _through(rgb)
        partial = {f: f for f in range(10)}      # last five have no partner

        result = tm.match_tracks(thermal, rgb, partial)
        assert result["matches"][0]["shared"] == 10

    def test_an_empty_flight_matches_nothing(self):
        result = tm.match_tracks([], [], {})
        assert result["matches"] == []
        assert result["candidates"] == 0

    def test_the_log_reports_what_was_confirmed(self):
        logged = []
        rgb = _rows(1, range(15), 100.0, 120.0)
        thermal = _through(rgb)
        tm.match_tracks(thermal, rgb, _identity_map(range(15)),
                        log_fn=logged.append)
        assert any("confirmed" in line for line in logged)

    def test_the_log_says_which_gate_rejected_everything(self):
        """'No matches' and 'the gate is wrong' look identical otherwise."""
        logged = []
        rgb = _rows(1, range(15), 100.0, 120.0)
        thermal = _through(rgb, jitter=6.0)
        tm.match_tracks(thermal, rgb, _identity_map(range(15)),
                        config=MatchConfig(gate_px=1.0),
                        log_fn=logged.append)

        line = next(x for x in logged if "no pair confirmed" in x)
        assert "1 were too far apart" in line
        assert "1 px gate" in line
        assert "closest candidate sat at" in line

    def test_the_log_names_the_shared_frame_gate(self):
        logged = []
        rgb = _rows(1, range(3), 100.0, 120.0)
        thermal = _through(rgb)
        tm.match_tracks(thermal, rgb, _identity_map(range(3)),
                        log_fn=logged.append)
        line = next(x for x in logged if "no pair confirmed" in x)
        assert "1 shared too few frames" in line
