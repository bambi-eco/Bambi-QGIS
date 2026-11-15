# -*- coding: utf-8 -*-
"""A heading from true north is not a bearing on the map.

The drone reports where its camera looks from true north. The map's north -
the project's CRS - is true north on the projection's central meridian only;
away from it the two differ by the grid convergence: 1.4 degrees at Salzburg
and 3.9 degrees in Vorarlberg in UTM zone 33. The heading was used as a map
bearing, so every frame was turned by that angle about the point below the
drone - unless a hand-calibrated rotation correction had taken it up.
"""
import json
import math

import numpy as np
import pytest

pytest.importorskip("pyproj")

from bambi_wildlife_detection.core import grid_north  # noqa: E402

UTM33 = 32633
SALZBURG = (47.80, 13.05)
BREGENZ = (47.50, 9.75)
VIENNA = (48.21, 16.37)


def by_formula(latitude, longitude, central_meridian):
    """What a transverse Mercator's convergence is to first order."""
    return (longitude - central_meridian) * math.sin(math.radians(latitude))


# ---------------------------------------------------------------------------
# The angle
# ---------------------------------------------------------------------------

class TestConvergence:
    @pytest.mark.parametrize("place", [SALZBURG, BREGENZ, VIENNA])
    def test_in_a_utm_zone(self, place):
        assert grid_north.convergence(*place, UTM33) == pytest.approx(
            by_formula(*place, 15.0), abs=0.01)

    def test_how_large_it_is_in_austria(self):
        assert grid_north.convergence(*SALZBURG, UTM33) == pytest.approx(
            -1.445, abs=0.005)
        assert grid_north.convergence(*BREGENZ, UTM33) == pytest.approx(
            -3.876, abs=0.005)
        # ... and in the zone the place belongs to
        assert grid_north.convergence(*BREGENZ, 32632) == pytest.approx(
            0.553, abs=0.005)

    def test_east_of_the_central_meridian_the_maps_north_is_east(self):
        assert grid_north.convergence(*VIENNA, UTM33) > 1.0
        assert grid_north.convergence(*SALZBURG, UTM33) < -1.0
        assert grid_north.convergence(47.5, 15.0, UTM33) == pytest.approx(
            0.0, abs=1e-6)

    def test_south_of_the_equator_it_is_the_other_way_round(self):
        # Cape Town, west of zone 34's central meridian at 21 degrees east
        assert grid_north.convergence(-33.9, 18.4, 32734) == pytest.approx(
            by_formula(-33.9, 18.4, 21.0), abs=0.01)
        assert grid_north.convergence(-33.9, 18.4, 32734) > 1.0

    def test_other_projections(self):
        # Austria Lambert: a cone about 13 degrees 20 minutes east
        assert grid_north.convergence(*SALZBURG, 31287) == pytest.approx(
            -0.21, abs=0.02)
        # Mercator: meridians are the map's verticals
        assert grid_north.convergence(*SALZBURG, 3857) == pytest.approx(
            0.0, abs=1e-6)

    def test_it_is_the_projections_own_figure(self):
        from pyproj import Proj

        factors = Proj(f"EPSG:{UTM33}").get_factors(SALZBURG[1], SALZBURG[0])
        assert grid_north.convergence(*SALZBURG, UTM33) == pytest.approx(
            factors.meridian_convergence, abs=1e-4)

    def test_many_places_at_once(self):
        angles = grid_north.convergence(
            [SALZBURG[0], BREGENZ[0]], [SALZBURG[1], BREGENZ[1]], UTM33)
        assert angles == pytest.approx([-1.445, -3.876], abs=0.005)


class TestPoses:
    def test_a_pose_without_the_angle_is_used_as_it_was(self):
        pose = {"rotation": [0.0, 0.0, 350.0]}
        assert grid_north.of_pose(pose) == 0.0
        assert grid_north.map_heading(pose) == 350.0

    def test_a_pose_with_it_has_it_taken_from_its_heading(self):
        pose = {"rotation": [0.0, 0.0, 1.0], grid_north.KEY: 3.0}
        assert grid_north.map_heading(pose) == pytest.approx(358.0)
        west = {"rotation": [0.0, 0.0, 359.0], grid_north.KEY: -3.0}
        assert grid_north.map_heading(west) == pytest.approx(2.0)

    @pytest.mark.parametrize("value", [None, "", "north", float("nan")])
    def test_an_angle_that_is_none(self, value):
        assert grid_north.of_pose({grid_north.KEY: value}) == 0.0
        assert grid_north.of_pose(None) == 0.0


# ---------------------------------------------------------------------------
# In the poses file
# ---------------------------------------------------------------------------

def _poses_file(tmp_path, images, **top):
    path = tmp_path / "poses_t.json"
    path.write_text(json.dumps(dict({"images": images}, **top)))
    return str(path)


def _image(latitude, longitude, heading=0.0):
    return {"imagefile": "f.jpg", "location": [0.0, 0.0, 80.0],
            "rotation": [0.0, 0.0, heading], "lat": latitude, "lng": longitude}


class TestRecord:
    def test_every_pose_gets_the_angle_where_it_was_taken(self, tmp_path):
        path = _poses_file(tmp_path, [_image(*SALZBURG), _image(*BREGENZ)],
                           mask="mask_T.png")

        angle = grid_north.record(path, UTM33)

        poses = json.loads(open(path).read())
        assert [image[grid_north.KEY] for image in poses["images"]] \
            == pytest.approx([-1.445, -3.876], abs=0.005)
        assert angle == pytest.approx((-1.445 - 3.876) / 2, abs=0.005)
        assert poses[grid_north.HEADINGS] == {
            "north": "map", "epsg": UTM33,
            grid_north.KEY: pytest.approx(angle, abs=1e-4)}
        assert poses["mask"] == "mask_T.png"            # the rest is kept
        assert poses["images"][0]["rotation"] == [0.0, 0.0, 0.0]

    def test_a_frame_without_a_position_gets_the_flights_angle(self, tmp_path):
        lost = dict(_image(0.0, 0.0), lat=None, lng=None)
        nowhere = _image(0.0, 0.0)                      # "no position": 0, 0
        path = _poses_file(tmp_path, [_image(*SALZBURG), lost, nowhere])

        angle = grid_north.record(path, UTM33)

        images = json.loads(open(path).read())["images"]
        assert [image[grid_north.KEY] for image in images] \
            == pytest.approx([angle] * 3, abs=1e-4)

    def test_frames_that_say_nothing_about_where_they_were_taken(self, tmp_path):
        bare = {"imagefile": "f.jpg", "location": [0, 0, 80],
                "rotation": [0, 0, 0]}
        path = _poses_file(tmp_path, [bare, dict(bare)])
        before = open(path).read()

        assert grid_north.record(path, UTM33) is None
        assert open(path).read() == before

    def test_recording_twice_changes_nothing(self, tmp_path):
        path = _poses_file(tmp_path, [_image(*SALZBURG)])
        grid_north.record(path, UTM33)
        once = open(path).read()
        grid_north.record(path, UTM33)
        assert open(path).read() == once

    def test_what_a_poses_file_says_about_its_headings(self, tmp_path):
        path = _poses_file(tmp_path, [_image(*SALZBURG)])
        assert grid_north.recorded(json.loads(open(path).read())) is None
        grid_north.record(path, UTM33)
        assert grid_north.recorded(json.loads(open(path).read()))["epsg"] \
            == UTM33
        assert grid_north.recorded({grid_north.HEADINGS: "true"}) is None
        assert grid_north.recorded(None) is None


class TestWhatIsSaid:
    def test_the_angle_in_words(self):
        assert grid_north.describe(-1.445, UTM33) == (
            "The map's north (EPSG:32633) lies 1.45 degrees west of true "
            "north here")
        assert "1.02 degrees east" in grid_north.describe(1.022, UTM33)

    def test_older_poses_get_a_note(self):
        poses = {"images": [_image(*SALZBURG)]}
        note = grid_north.note_for_older_poses(poses, UTM33)
        assert note.startswith("Note: The map's north (EPSG:32633) lies 1.44 "
                               "degrees west of true north here.")
        assert "Extracting the frames again" in note

    def test_poses_that_have_the_angle_get_none(self, tmp_path):
        path = _poses_file(tmp_path, [_image(*SALZBURG)])
        grid_north.record(path, UTM33)
        assert grid_north.note_for_older_poses(
            json.loads(open(path).read()), UTM33) is None

    def test_where_it_does_not_matter_nothing_is_said(self):
        on_the_meridian = {"images": [_image(47.5, 15.02)]}
        assert grid_north.note_for_older_poses(on_the_meridian, UTM33) is None
        assert grid_north.note_for_older_poses({"images": []}, UTM33) is None
        assert grid_north.note_for_older_poses(
            {"images": [{"rotation": [0, 0, 0]}]}, UTM33) is None

    def test_a_rotation_correction_that_may_contain_the_angle(self):
        hint = grid_north.correction_hint(math.radians(-1.2), -1.445)
        assert "z = -1.20 degrees" in hint
        assert "contains the 1.45 degrees" in hint
        assert "should then be about 0.25 degrees" in hint     # z - angle

    def test_no_hint_without_a_correction_or_an_angle(self):
        assert grid_north.correction_hint(0.0, -1.445) is None
        assert grid_north.correction_hint(math.radians(2.0), 0.01) is None
        assert grid_north.correction_hint(None, -1.445) is None


# ---------------------------------------------------------------------------
# At extraction
# ---------------------------------------------------------------------------

class TestAtExtraction:
    @pytest.fixture
    def processor(self):
        from bambi_wildlife_detection.bambi_processing import BambiProcessor

        return BambiProcessor

    def test_the_poses_get_their_angles(self, processor, tmp_path):
        path = _poses_file(tmp_path, [_image(*SALZBURG)])
        said = []
        processor._headings_to_the_map(
            path, {"target_epsg": UTM33}, said.append)

        assert json.loads(open(path).read())["images"][0][grid_north.KEY] \
            == pytest.approx(-1.445, abs=0.005)
        assert said == [
            "The map's north (EPSG:32633) lies 1.44 degrees west of true "
            "north here; the headings are turned accordingly."]

    def test_a_calibrated_rotation_is_pointed_out(self, processor, tmp_path):
        path = _poses_file(tmp_path, [_image(*SALZBURG)])
        said = []
        processor._headings_to_the_map(
            path, {"target_epsg": UTM33,
                   "rotation": {"x": 0.0, "y": 0.0, "z": math.radians(-1.5)}},
            said.append)

        assert len(said) == 2 and "z = -1.50 degrees" in said[1]

    def test_unticked_the_headings_stay_bearings_on_the_map(
            self, processor, tmp_path):
        path = _poses_file(tmp_path, [_image(*SALZBURG)])
        before = open(path).read()
        said = []
        processor._headings_to_the_map(
            path, {"target_epsg": UTM33, "correct_grid_convergence": False},
            said.append)

        assert open(path).read() == before
        assert said == ["Headings are used as bearings on the map (grid "
                        "convergence not taken into account)."]

    def test_frames_without_positions_are_said_to_be_so(self, processor, tmp_path):
        path = _poses_file(tmp_path, [{"rotation": [0, 0, 0]}])
        said = []
        processor._headings_to_the_map(path, {"target_epsg": UTM33},
                                       said.append)
        assert said == ["The frames carry no geographic position: headings "
                        "are used as bearings on the map."]

    def test_a_crs_that_cannot_be_used_leaves_the_poses_alone(
            self, processor, tmp_path):
        path = _poses_file(tmp_path, [_image(*SALZBURG)])
        before = open(path).read()
        said = []
        processor._headings_to_the_map(path, {"target_epsg": 999999},
                                       said.append)
        assert open(path).read() == before
        assert said[0].startswith("Warning: the grid convergence could not "
                                  "be worked out")

    def test_the_geo_referencing_says_when_poses_are_older(self, processor):
        said = []
        processor._say_if_headings_are_true_north(
            {"images": [_image(*BREGENZ)]}, {"target_epsg": UTM33},
            said.append)
        assert len(said) == 1 and "3.88 degrees west" in said[0]

    def test_the_setting_is_part_of_a_flights_settings(self):
        from bambi_wildlife_detection.core import config_schema

        entry = next(entry for entry in config_schema.CONFIG_ENTRIES
                     if entry.key == "Extraction/GridConvergence")
        assert entry.default is True
        assert config_schema.WIDGET_BINDINGS["Extraction/GridConvergence"] \
            == ("grid_convergence_check", "check")


# ---------------------------------------------------------------------------
# Where the camera looks
# ---------------------------------------------------------------------------

class TestWhereTheCameraLooks:
    """The plugin's camera model, with the real ray caster: a camera 100 m
    up, tilted 40 degrees forward, looks at a place 84 m ahead. Where that
    place is on the map is known from the globe."""

    HEIGHT, TILT = 100.0, 40.0

    @pytest.fixture
    def looking(self):
        pytest.importorskip("alfspy")
        trimesh = pytest.importorskip("trimesh")
        from alfspy.core.convert.convert import pixel_to_world_coord
        from pyproj import CRS, Geod, Transformer

        from bambi_wildlife_detection.core import camera_pose

        ground = trimesh.Trimesh(
            vertices=[[-5000, -5000, 0], [5000, -5000, 0], [5000, 5000, 0],
                      [-5000, 5000, 0]],
            faces=[[0, 1, 2], [0, 2, 3]], process=False)
        to_map = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_epsg(UTM33), always_xy=True)
        distance = self.HEIGHT * math.tan(math.radians(self.TILT))

        def look(place, heading, with_angle):
            latitude, longitude = place
            pose = {"location": [0.0, 0.0, self.HEIGHT],
                    "rotation": [self.TILT, 0.0, heading], "fovy": [50.0]}
            if with_angle:
                pose[grid_north.KEY] = grid_north.convergence(
                    latitude, longitude, UTM33)
            camera = camera_pose.build_camera(pose, {}, {}, aspect_ratio=1.0)
            hit = np.reshape(np.asarray(pixel_to_world_coord(
                [256], [256], 512, 512, ground, camera,
                include_misses=False), dtype=float), (-1, 3))[0]

            east, north = to_map.transform(longitude, latitude)
            there = Geod(ellps="WGS84").fwd(
                longitude, latitude, heading, distance)
            east_there, north_there = to_map.transform(there[0], there[1])
            truly = np.array([east_there - east, north_there - north])
            return float(np.linalg.norm(hit[:2] - truly))

        return look

    @pytest.mark.parametrize("heading", [0.0, 90.0, 215.0])
    @pytest.mark.parametrize("place", [BREGENZ, SALZBURG, VIENNA])
    def test_with_the_angle_it_looks_where_it_looks(self, looking, place,
                                                    heading):
        # what is left is the map's scale: a thousandth of 84 m
        assert looking(place, heading, with_angle=True) < 0.2

    @pytest.mark.parametrize("place, off", [
        (BREGENZ, 5.68), (SALZBURG, 2.12), (VIENNA, 1.50)])
    def test_without_it_everything_is_turned_about_the_drone(
            self, looking, place, off):
        """84 m ahead, by the angle: metres."""
        assert looking(place, 0.0, with_angle=False) == pytest.approx(
            off, abs=0.1)

    def test_a_rotation_correction_still_acts_on_top(self):
        pytest.importorskip("alfspy")
        from bambi_wildlife_detection.core import camera_pose

        pose = {"location": [0.0, 0.0, 100.0], "rotation": [40.0, 0.0, 10.0]}
        turned = dict(pose, **{grid_north.KEY: -3.0})
        # 3 degrees taken by the pose are 3 degrees less for the correction
        by_pose = camera_pose.corrected_pose(turned, {}, {})[1]
        by_hand = camera_pose.corrected_pose(
            pose, {}, {"x": 0.0, "y": 0.0, "z": math.radians(-3.0)})[1]
        assert np.allclose(np.asarray(by_pose), np.asarray(by_hand), atol=1e-6)

    def test_a_ready_made_rotation_is_left_alone(self):
        pytest.importorskip("alfspy")
        from bambi_wildlife_detection.core import camera_pose

        quaternion = [0.0, 0.0, 0.0, 1.0]
        pose = {"location": [0, 0, 100], "rotation": quaternion}
        plain = camera_pose.corrected_pose(pose, {}, {})[1]
        turned = camera_pose.corrected_pose(
            dict(pose, **{grid_north.KEY: -3.0}), {}, {})[1]
        assert np.allclose(np.asarray(plain), np.asarray(turned))
