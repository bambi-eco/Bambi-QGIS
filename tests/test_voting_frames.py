# -*- coding: utf-8 -*-
"""Which frames of a track may vote on its species, sex and age.

Occlusion is a quality filter, not a prerequisite. It used to be decided
once for the whole flight: as soon as any detection had an occlusion label,
every frame without one counted as occluded. One hand-labelled track with an
occlusion value then left every tracker track without a frame to vote with,
so none of them got a species - and so did calls of the occlusion classifier
left over from before the animals were detected again.

(No torch needed: the classifier heads are tested in
``test_classification_heads.py``, which a machine without torch skips.)
"""
from bambi_wildlife_detection import bambi_processing
from bambi_wildlife_detection.core import classification as cl

CLEAR, OCCLUDED = 0, 1


def _vote(ids, **known):
    return cl.visible_detections(ids, clear_annotations=(CLEAR,), **known)


def test_one_labelled_track_does_not_silence_the_others():
    """Track A was labelled by hand, track B never was."""
    annotated = {1: CLEAR, 2: OCCLUDED, 3: CLEAR}

    assert _vote([1, 2, 3], annotated=annotated) == (
        [1, 3], cl.FRAMES_FROM_ANNOTATIONS)
    # B's frames have no label at all: nobody called them occluded
    assert _vote([10, 11, 12], annotated=annotated) == (
        [10, 11, 12], cl.FRAMES_FROM_NOTHING)


def test_calls_for_boxes_that_are_gone_do_not_silence_the_new_ones():
    """The classifier ran, then the animals were detected again."""
    stale = {1: "clear", 2: "occluded"}
    assert cl.visible_detections([50, 51], head_labels=stale) == (
        [50, 51], cl.FRAMES_FROM_NOTHING)


def test_a_frame_is_decided_by_the_best_that_is_known_about_it():
    chosen, source = _vote(
        [1, 2, 3, 4, 5, 6],
        head_labels={1: "clear", 2: "occluded"},
        annotated={2: CLEAR, 3: CLEAR, 4: OCCLUDED})
    # 1, 2: the classifier's call (it outranks what is stored for 2);
    # 3, 4: the stored value; 5, 6: nothing known
    assert chosen == [1, 3, 5, 6]
    assert source == cl.FRAMES_FROM_HEAD


def test_a_track_all_called_occluded_has_no_vote():
    assert cl.visible_detections(
        [1, 2], head_labels={1: "occluded", 2: "occluded"}) == (
        [], cl.FRAMES_FROM_HEAD)
    assert _vote([1, 2], annotated={1: OCCLUDED, 2: OCCLUDED}) == (
        [], cl.FRAMES_FROM_ANNOTATIONS)


def test_without_any_label_every_frame_votes_and_says_so():
    assert cl.visible_detections([3, 1, 2]) == ([3, 1, 2],
                                               cl.FRAMES_FROM_NOTHING)
    assert cl.visible_detections([3, 1, 2], head_labels={}, annotated={}) \
        == ([3, 1, 2], cl.FRAMES_FROM_NOTHING)


def test_all_frames_ignores_occlusion():
    assert cl.visible_detections(
        [1, 2], head_labels={1: "occluded", 2: "occluded"}, use_all=True) == (
        [1, 2], cl.FRAMES_FROM_ALL)


def test_the_classifiers_own_words_for_clear_are_honoured():
    chosen, _source = cl.visible_detections(
        [1, 2, 3], head_labels={1: "visible", 2: "hidden"},
        clear_head_labels=("visible",))
    assert chosen == [1, 3]


def test_unfiltered_votes_are_counted():
    voting = {1: [1, 3], 2: [10, 11, 12], 3: []}
    assert cl.unlabelled_voters(voting, {1: "clear"}, {3: CLEAR}) == 3
    assert cl.unlabelled_voters(voting) == 5
    assert cl.unlabelled_voters({}, {1: "clear"}) == 0


def test_the_log_names_each_source_with_its_tracks():
    describe = bambi_processing._describe_frame_sources
    one = describe({1: cl.FRAMES_FROM_HEAD, 2: cl.FRAMES_FROM_HEAD})
    assert one == "clear frames per the occlusion classifier"
    mixed = describe({1: cl.FRAMES_FROM_ANNOTATIONS,
                      2: cl.FRAMES_FROM_NOTHING, 3: cl.FRAMES_FROM_NOTHING})
    assert mixed.startswith("all frames - no occlusion labels")
    assert "2 track(s)" in mixed and "annotations: 1 track(s)" in mixed
    assert describe({}) == "no tracks"
