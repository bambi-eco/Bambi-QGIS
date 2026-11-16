# -*- coding: utf-8 -*-
"""How far off the place of a detection may be
(core/position_uncertainty.py): the budget of one detection, a flight's
detections, and the animals."""
import math

import pytest

from bambi_wildlife_detection.core import position_uncertainty as pu

ONE_DEGREE = math.radians(1.0)


def only(**kept):
    """An accuracy in which only what is named is unknown."""
    values = dict(gps_horizontal_m=0.0, gps_vertical_m=0.0,
                  camera_angle_deg=0.0, terrain_m=0.0, pixel_px=0.0,
                  sync_s=0.0)
    values.update(kept)
    return pu.Accuracy(**values)


# ---------------------------------------------------------------------------
# What is taken as known
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_the_defaults_are_a_drone_without_a_base_station(self):
        accuracy = pu.Accuracy()

        assert (accuracy.gps_horizontal_m, accuracy.gps_vertical_m,
                accuracy.camera_angle_deg, accuracy.terrain_m,
                accuracy.pixel_px, accuracy.sync_s) == (
            1.5, 1.0, 1.0, 1.0, 2.0, 0.1)
        assert pu.PRESETS["Standard GNSS"] == accuracy
        assert pu.PRESETS["RTK"].gps_horizontal_m == 0.05
        assert pu.PRESETS["RTK"].camera_angle_deg == 1.0   # RTK is no gimbal

    def test_from_the_settings(self):
        accuracy = pu.Accuracy.from_config({
            "position_gps_horizontal_m": "0.05", "position_sync_s": 0,
            "something_else": 3})

        assert (accuracy.gps_horizontal_m, accuracy.sync_s) == (0.05, 0.0)
        assert accuracy.terrain_m == 1.0

    @pytest.mark.parametrize("config", [
        {"position_camera_angle_deg": -1},
        {"position_terrain_m": "about one"},
        {"position_pixel_px": float("nan")},
    ])
    def test_an_accuracy_that_is_none_is_refused(self, config):
        with pytest.raises(pu.UncertaintyError, match="not an accuracy"):
            pu.Accuracy.from_config(config)

    def test_a_truth_value_is_no_number(self):
        with pytest.raises(pu.UncertaintyError):
            pu.Accuracy(sync_s=True).validate()

    def test_the_angle_of_a_pixel(self):
        assert pu.pixel_angle(50.0, 512) == pytest.approx(
            2 * math.tan(math.radians(25)) / 512)
        assert pu.pixel_angle(None, 512) is None
        assert pu.pixel_angle(50.0, 0) is None


# ---------------------------------------------------------------------------
# The budget of one detection
# ---------------------------------------------------------------------------

class TestBudget:
    def test_straight_below_a_degree_is_the_height_times_it(self):
        found = pu.budget(80.0, 0.0, only(camera_angle_deg=1.0))

        assert found["radial_m"] == pytest.approx(80 * ONE_DEGREE)
        assert found["cross_m"] == pytest.approx(80 * ONE_DEGREE)
        assert found["error_m"] == pytest.approx(80 * ONE_DEGREE * 2 ** 0.5)
        assert found["error95_m"] == pytest.approx(2 * found["error_m"])
        assert (found["off_nadir_deg"], found["range_m"]) == (0.0, 80.0)
        assert found["main_source"] == pu.ANGLE

    def test_to_the_side_the_same_degree_reaches_further(self):
        found = pu.budget(80.0, 80.0, only(camera_angle_deg=1.0))

        reach = 80 * 2 ** 0.5
        assert found["off_nadir_deg"] == pytest.approx(45.0)
        assert found["range_m"] == pytest.approx(reach)
        # along the line of sight: R^2 / H; across it: R
        assert found["radial_m"] == pytest.approx(reach ** 2 / 80 * ONE_DEGREE)
        assert found["cross_m"] == pytest.approx(reach * ONE_DEGREE)
        assert found["radial_m"] == pytest.approx(
            2 * pu.budget(80.0, 0.0, only(camera_angle_deg=1.0))["radial_m"])

    def test_the_cameras_place_is_the_detections(self):
        found = pu.budget(80.0, 60.0, only(gps_horizontal_m=1.5))

        assert (found["radial_m"], found["cross_m"]) == (1.5, 1.5)
        assert found["main_source"] == pu.GPS

    def test_height_only_matters_to_the_side(self):
        below = pu.budget(80.0, 0.0, only(gps_vertical_m=1.0, terrain_m=1.0))
        beside = pu.budget(80.0, 40.0, only(gps_vertical_m=1.0,
                                            terrain_m=1.0))

        assert below["error_m"] == 0.0 and below["main_source"] is None
        # a metre of height is half a metre on the ground at tan = 0.5
        assert beside["parts"][pu.HEIGHT] == pytest.approx((0.5, 0.0))
        assert beside["parts"][pu.TERRAIN] == pytest.approx((0.5, 0.0))
        assert beside["radial_m"] == pytest.approx(0.5 * 2 ** 0.5)
        assert beside["cross_m"] == 0.0

    def test_a_pixel_is_an_angle(self):
        per_pixel = pu.pixel_angle(50.0, 512)

        found = pu.budget(80.0, 0.0, only(pixel_px=2.0), pixel_rad=per_pixel)

        assert found["radial_m"] == pytest.approx(80 * 2 * per_pixel)
        assert found["main_source"] == pu.PIXEL
        assert pu.budget(80.0, 0.0, only(pixel_px=2.0))["error_m"] == 0.0

    def test_a_frame_of_another_moment_is_the_way_the_drone_flew(self):
        timing = only(sync_s=0.1)

        ahead = pu.budget(80.0, 30.0, timing, speed_ms=5.0,
                          flight_angle_rad=0.0)
        abeam = pu.budget(80.0, 30.0, timing, speed_ms=5.0,
                          flight_angle_rad=math.pi / 2)

        assert (ahead["radial_m"], ahead["cross_m"]) == pytest.approx(
            (0.5, 0.0))
        assert (abeam["radial_m"], abeam["cross_m"]) == pytest.approx(
            (0.0, 0.5), abs=1e-12)
        assert ahead["main_source"] == pu.TIMING
        assert pu.budget(80.0, 30.0, timing)["error_m"] == 0.0   # hovering

    def test_ground_that_falls_away_makes_everything_worse(self):
        level = pu.budget(80.0, 80.0, pu.Accuracy())
        falling = pu.budget(80.0, 80.0, pu.Accuracy(), slope=-0.5)
        rising = pu.budget(80.0, 80.0, pu.Accuracy(), slope=0.5)

        assert (level["terrain_factor"], falling["terrain_factor"]) == (
            1.0, 2.0)
        assert rising["terrain_factor"] == pytest.approx(2 / 3)
        assert falling["radial_m"] > 3 * level["radial_m"]
        assert rising["radial_m"] < level["radial_m"]
        # across the line of sight the slope changes nothing
        assert falling["cross_m"] == level["cross_m"] == rising["cross_m"]
        # the angle goes by the square, the place by the factor itself
        assert falling["parts"][pu.ANGLE][0] == pytest.approx(
            4 * level["parts"][pu.ANGLE][0])
        assert falling["parts"][pu.GPS][0] == pytest.approx(
            2 * level["parts"][pu.GPS][0])

    def test_ground_along_the_line_of_sight_is_held_and_marked(self):
        found = pu.budget(80.0, 80.0, pu.Accuracy(), slope=-0.99)

        assert found["grazing"] is True
        assert found["terrain_factor"] == pu.MAX_TERRAIN_FACTOR
        assert math.isfinite(found["error_m"]) and found["error_m"] > 50
        assert pu.budget(80.0, 80.0, pu.Accuracy(),
                         slope=-0.5)["grazing"] is False

    def test_the_parts_add_by_their_squares(self):
        found = pu.budget(80.0, 50.0, pu.Accuracy(),
                          pixel_rad=pu.pixel_angle(50.0, 512), speed_ms=5.0,
                          flight_angle_rad=0.6)

        assert list(found["parts"]) == list(pu.SOURCES)
        assert found["radial_m"] == pytest.approx(math.sqrt(sum(
            radial ** 2 for radial, _cross in found["parts"].values())))
        assert found["cross_m"] == pytest.approx(math.sqrt(sum(
            cross ** 2 for _radial, cross in found["parts"].values())))
        assert found["error_m"] == pytest.approx(
            math.hypot(found["radial_m"], found["cross_m"]))

    def test_everything_known_is_no_error(self):
        found = pu.budget(80.0, 50.0, only())

        assert (found["error_m"], found["main_source"]) == (0.0, None)

    def test_rtk_leaves_the_angle(self):
        """From 50 m up, nearly below: the drone's own place is what
        matters most - until a base station fixes it."""
        gnss = pu.budget(50.0, 10.0, pu.PRESETS["Standard GNSS"])
        rtk = pu.budget(50.0, 10.0, pu.PRESETS["RTK"])

        assert gnss["main_source"] == pu.GPS
        assert rtk["main_source"] == pu.ANGLE
        assert rtk["error_m"] < 0.6 * gnss["error_m"]
        assert rtk["error_m"] > 1.0          # a degree is still a degree

    def test_from_higher_up_the_angle_takes_over(self):
        assert pu.budget(80.0, 30.0, pu.Accuracy())["main_source"] == pu.ANGLE

    @pytest.mark.parametrize("height, offset", [
        (0.0, 10.0), (-5.0, 10.0), (float("nan"), 10.0), (1.0, 50.0)])
    def test_no_place_can_be_read(self, height, offset):
        with pytest.raises(pu.UncertaintyError):
            pu.budget(height, offset, pu.Accuracy())


# ---------------------------------------------------------------------------
# From where the camera and the animal were
# ---------------------------------------------------------------------------

class TestBox:
    def test_the_geometry_is_read_from_the_two_places(self):
        found = pu.describe_box((100.0, 200.0, 380.0), (160.0, 280.0, 300.0),
                                pu.Accuracy())

        assert found["height_m"] == 80.0
        assert found["range_m"] == pytest.approx(math.hypot(100.0, 80.0))
        assert found["off_nadir_deg"] == pytest.approx(
            math.degrees(math.atan2(100.0, 80.0)))
        # 60 m east and 80 m north of the camera
        assert found["azimuth_deg"] == pytest.approx(
            math.degrees(math.atan2(60.0, 80.0)))
        assert found["slope_deg"] == 0.0

    def test_the_slope_is_the_one_along_the_line_of_sight(self):
        camera, accuracy = (0.0, 0.0, 80.0), pu.Accuracy()
        rising_east = (0.2, 0.0)

        east = pu.describe_box(camera, (50.0, 0.0, 0.0), accuracy,
                               rising_east)
        west = pu.describe_box(camera, (-50.0, 0.0, 0.0), accuracy,
                               rising_east)
        north = pu.describe_box(camera, (0.0, 50.0, 0.0), accuracy,
                                rising_east)

        assert east["slope_deg"] == pytest.approx(math.degrees(math.atan(0.2)))
        assert west["slope_deg"] == pytest.approx(-east["slope_deg"])
        assert north["slope_deg"] == pytest.approx(0.0)
        assert west["error_m"] > north["error_m"] > east["error_m"]

    def test_the_drones_speed_counts_along_its_way(self):
        timing = only(sync_s=0.2)

        ahead = pu.describe_box((0, 0, 80.0), (30.0, 0.0, 0.0), timing,
                                velocity=(5.0, 0.0))
        abeam = pu.describe_box((0, 0, 80.0), (0.0, 30.0, 0.0), timing,
                                velocity=(5.0, 0.0))

        assert (ahead["radial_m"], ahead["cross_m"]) == pytest.approx(
            (1.0, 0.0))
        assert (abeam["radial_m"], abeam["cross_m"]) == pytest.approx(
            (0.0, 1.0), abs=1e-12)

    def test_straight_below(self):
        found = pu.describe_box((0, 0, 80.0), (0.0, 0.0, 0.0), pu.Accuracy(),
                                gradient=(0.3, 0.3))

        assert found["off_nadir_deg"] == 0.0
        assert found["terrain_factor"] == 1.0     # no lean, no slope effect

    def test_a_camera_below_the_animal_gives_no_place(self):
        assert pu.describe_box((0, 0, 290.0), (10.0, 0.0, 300.0),
                               pu.Accuracy()) is None
        assert pu.describe_box((0, 0, 301.0), (500.0, 0.0, 300.0),
                               pu.Accuracy()) is None


# ---------------------------------------------------------------------------
# A flight
# ---------------------------------------------------------------------------

def _boxes():
    return [
        {"detection_id": 1, "track_id": 1, "frame": 0, "x": 0.0, "y": 0.0,
         "z": 300.0},
        {"detection_id": 2, "track_id": 1, "frame": 1, "x": 0.0, "y": 40.0,
         "z": 300.0},
        {"detection_id": 3, "track_id": 2, "frame": 2, "x": 100.0, "y": 80.0,
         "z": 300.0},
        {"detection_id": 4, "track_id": None, "frame": 2, "x": 12.0,
         "y": 0.0, "z": 300.0},
    ]


CAMERAS = {frame: (5.0 * frame, 0.0, 380.0) for frame in range(3)}


class TestFlight:
    def test_one_row_per_detection_with_a_place(self):
        rows, left_out = pu.box_rows("Meadow", _boxes(), CAMERAS,
                                     pu.Accuracy())

        assert left_out == 0
        assert [list(row) for row in rows] == [pu.BOX_COLUMNS] * 4
        first = rows[0]
        assert (first["flight"], first["detection_id"], first["track_id"],
                first["frame"]) == ("Meadow", 1, 1, 0)
        assert (first["height_m"], first["off_nadir_deg"]) == (80.0, 0.0)
        assert rows[2]["range_m"] == pytest.approx(
            math.sqrt(90.0 ** 2 + 80.0 ** 2 + 80.0 ** 2))
        assert rows[2]["error_m"] > rows[1]["error_m"] > first["error_m"]
        assert rows[3]["track_id"] is None
        assert first["grazing"] is False

    def test_the_speed_is_read_from_the_frames_either_side(self):
        seconds = {0: 0.0, 1: 1.0, 2: 2.0}
        timing = only(sync_s=0.2)

        rows, _left_out = pu.box_rows("Meadow", _boxes(), CAMERAS, timing,
                                      seconds=seconds)
        without, _left_out = pu.box_rows("Meadow", _boxes(), CAMERAS, timing)

        # 5 m per second, due east: a metre in a fifth of a second. Box 4
        # lies ahead; box 2 is 40 m to the side and 5 m behind
        assert rows[3]["radial_m"] == pytest.approx(1.0)
        assert rows[3]["cross_m"] == pytest.approx(0.0, abs=1e-12)
        assert rows[1]["cross_m"] == pytest.approx(40 / math.hypot(5, 40))
        assert rows[1]["radial_m"] == pytest.approx(5 / math.hypot(5, 40))
        assert all(row["error_m"] == 0.0 for row in without)

    def test_the_terrain_is_asked_at_the_detection(self):
        asked = []

        def gradient_of(x, y):
            asked.append((x, y))
            return (0.0, -0.4) if y > 50 else None

        rows, _left_out = pu.box_rows("Meadow", _boxes(), CAMERAS,
                                      pu.Accuracy(), gradient_of=gradient_of)

        assert asked == [(0.0, 0.0), (0.0, 40.0), (100.0, 80.0), (12.0, 0.0)]
        assert rows[1]["slope_deg"] == 0.0            # not known: level
        assert rows[2]["slope_deg"] < 0

    def test_what_gives_no_place_is_left_out_and_counted(self):
        boxes = _boxes() + [
            {"detection_id": 5, "track_id": 3, "frame": 9, "x": 0.0,
             "y": 0.0, "z": 300.0},                    # no such frame
            {"detection_id": 6, "track_id": 3, "frame": 0, "x": None,
             "y": None, "z": None},                    # never placed
            {"detection_id": 7, "track_id": 3, "frame": 0, "x": 0.0,
             "y": 0.0, "z": 400.0},                    # above the camera
        ]

        rows, left_out = pu.box_rows("Meadow", boxes, CAMERAS, pu.Accuracy())

        assert (len(rows), left_out) == (4, 3)


# ---------------------------------------------------------------------------
# The animals, and all together
# ---------------------------------------------------------------------------

class TestAnimals:
    def test_an_animal_is_as_far_off_as_its_boxes_typically_are(self):
        rows, _left_out = pu.box_rows("Meadow", _boxes(), CAMERAS,
                                      pu.Accuracy())
        animals = [
            {"flight": "Meadow", "track_id": 1, "species": "roe deer",
             "x": 0.0, "y": 20.0},
            {"flight": "Meadow", "track_id": 2, "species": "red deer",
             "x": 100.0, "y": 80.0},
            {"flight": "Meadow", "track_id": 9, "species": "roe deer",
             "x": None, "y": None},
        ]

        found = pu.track_rows(rows, animals)

        assert [list(row) for row in found] == [pu.TRACK_COLUMNS] * 3
        first = found[0]
        assert first["boxes"] == 2
        assert first["error_m"] == pytest.approx(
            (rows[0]["error_m"] + rows[1]["error_m"]) / 2)
        assert first["error95_m"] == pytest.approx(2 * first["error_m"])
        assert first["best_error_m"] == rows[0]["error_m"]
        assert first["main_source"] == pu.GPS
        assert found[1]["main_source"] == pu.ANGLE
        # an animal none of whose boxes has a place
        assert (found[2]["boxes"], found[2]["error_m"],
                found[2]["main_source"]) == (0, None, None)

    def test_the_same_track_number_in_two_flights_is_two_animals(self):
        rows, _ = pu.box_rows("Meadow", _boxes()[:1], CAMERAS, pu.Accuracy())
        other, _ = pu.box_rows("Forest", _boxes()[2:3], CAMERAS,
                               pu.Accuracy())
        other[0]["track_id"] = 1

        found = pu.track_rows(rows + other, [
            {"flight": "Meadow", "track_id": 1}, {"flight": "Forest",
                                                  "track_id": 1}])

        assert found[0]["error_m"] < found[1]["error_m"]


class TestSummary:
    @pytest.fixture
    def rows(self):
        return [{"error_m": float(e), "main_source": source,
                 "grazing": e > 50, "height_m": 80.0,
                 "off_nadir_deg": float(e)}
                for e, source in zip(
                    (1, 2, 2, 3, 3, 3, 4, 6, 12, 60),
                    [pu.GPS] * 6 + [pu.ANGLE] * 4)]

    def test_together(self, rows):
        summary = pu.summarise(rows)

        assert summary["boxes"] == 10
        assert (summary["median_m"], summary["max_m"]) == (3.0, 60.0)
        assert summary["median95_m"] == 6.0
        assert summary["p90_m"] == pytest.approx(12 + 0.1 * 48)
        assert summary["within"] == {"1": 0.1, "2": 0.3, "5": 0.7,
                                     "10": 0.8}
        assert summary["main_sources"] == {pu.GPS: 0.6, pu.ANGLE: 0.4}
        assert summary["grazing"] == 1
        assert summary["median_height_m"] == 80.0

    def test_in_a_sentence(self, rows):
        assert pu.sentence(pu.summarise(rows)) == (
            "10 detections: typically 3.0 m off (median; nine in ten "
            "within 16.8 m, the worst 60.0 m). What matters most is the "
            "camera position - for 60 % of them. 1 lie where the terrain "
            "all but runs along the line of sight: their place is hardly "
            "known.")

    def test_nothing_to_judge(self):
        summary = pu.summarise([])

        assert (summary["boxes"], summary["median_m"]) == (0, None)
        assert summary["within"]["5"] is None
        assert pu.sentence(summary) == "No detection has a place to judge."

    def test_in_a_few_lines(self, rows):
        document = {"summary": pu.summarise(rows), "by_flight": {
            "Meadow": pu.summarise(rows[:5]), "Empty": pu.summarise([])}}

        lines = pu.describe(document)

        assert lines[0].startswith("10 detections: typically 3.0 m off")
        assert lines[1] == ("  Meadow: median 2.0 m, nine in ten within "
                            "3.0 m")
        assert len(lines) == 2


class TestWritten:
    def test_the_tables(self, tmp_path):
        import csv

        rows, _ = pu.box_rows("Meadow", _boxes(), CAMERAS, pu.Accuracy(),
                              gradient_of=lambda x, y: (-0.99, 0.0))
        boxes = pu.write_table(rows, str(tmp_path / "boxes.csv"))
        tracks = pu.write_table(
            pu.track_rows(rows, [{"flight": "Meadow", "track_id": 1,
                                  "species": "roe deer", "x": 0.0,
                                  "y": 20.0}]),
            str(tmp_path / "tracks.csv"), pu.TRACK_COLUMNS)

        with open(boxes, encoding="utf-8", newline="") as handle:
            found = list(csv.DictReader(handle))
        assert list(found[0]) == pu.BOX_COLUMNS
        assert (found[0]["detection_id"], found[3]["track_id"]) == ("1", "")
        assert found[0]["grazing"] == "" and found[2]["grazing"] == "yes"
        with open(tracks, encoding="utf-8", newline="") as handle:
            found = list(csv.DictReader(handle))
        assert list(found[0]) == pu.TRACK_COLUMNS
        assert found[0]["boxes"] == "2"

    def test_the_document_can_be_read_again(self, tmp_path):
        rows, _ = pu.box_rows("Meadow", _boxes(), CAMERAS, pu.Accuracy())
        document = {"summary": pu.summarise(rows), "boxes": rows}

        again = pu.read_json(pu.write_json(document,
                                           str(tmp_path / "u.json")))

        assert again["summary"]["boxes"] == 4
        assert again["boxes"][0]["main_source"] == pu.GPS
