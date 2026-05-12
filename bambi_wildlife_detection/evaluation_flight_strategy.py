import abc
import copy
import math
import os.path
import shutil
import sys
from typing import List, Optional
# pyproj must be imported before geopandas: geopandas checks pyproj
# availability at import time and caches the result.  Importing pyproj first
# ensures the correct (QGIS-bundled) version is in sys.modules when geopandas
# runs its compatibility check.
from pyproj import Transformer, CRS
import geopandas as gpd
from fiona.drvsupport import supported_drivers
import shapely
import json
import random
from collections import namedtuple
import simplekml

GridPosition = namedtuple("GridPosition", ["latitude", "longitude", "x", "y"])
TransectStatus = namedtuple("TransectStatus", ["is_valid", "already_visited"])


class EvaluationFlightStrategy(abc.ABC):
    """
    Abstract class representing a strategy to create a plan of flight routes for evaluation
    """

    @abc.abstractmethod
    def create_routes(
        self,
        area_path: str,
        start_points_path: str,
        target_path: str,
        invalid_areas_path: Optional[str] = None,
    ) -> List:
        """
        :param area_path: Path to the shape file (.shp, .kml, ...) representing the flight area
        :param start_points_path: Path to the shape file (.shp, .kml, ...) representing the start points
        :param target_path: Path to the target folder, were output files should be created
        :param invalid_areas_path: Path to the shape file (.shp, .kml, ...) representing the no-flight area
        :return: Valid routes
        """
        pass

    def read_data(
        self,
        area_path: str,
        start_points_path: str,
        target_crs,
        invalid_areas_path: Optional[str] = None,
    ):
        # read in all the data
        flight_areas = gpd.read_file(area_path)
        if flight_areas.crs is None:
            # GeoJSON (RFC 7946) and many exported files carry no CRS member;
            # WGS-84 is the correct default for those formats.
            flight_areas = flight_areas.set_crs("epsg:4326")
        # convert the file for later calculations like geod.fwd()
        if flight_areas.crs.srs != target_crs.srs:
            flight_areas = flight_areas.to_crs(target_crs)
        invalid_areas = None
        if invalid_areas_path is not None:
            invalid_areas = gpd.read_file(invalid_areas_path)
            if invalid_areas.crs is None:
                invalid_areas = invalid_areas.set_crs("epsg:4326")
            # convert the file for later calculations like geod.fwd()
            if invalid_areas.crs.srs != target_crs.srs:
                invalid_areas = invalid_areas.to_crs(target_crs)

        start_points_df = gpd.read_file(start_points_path)
        # TODO remove - in future all files should have a defined crs!...
        if start_points_df.crs is None:
            start_points_df = start_points_df.set_crs("epsg:31258")
        # convert the file for later calculations like geod.fwd()
        if start_points_df.crs.srs != target_crs.srs:
            start_points_df = start_points_df.to_crs(target_crs)

        return flight_areas, start_points_df, invalid_areas

    def get_crs(self, epsg_code):
        target_crs = CRS.from_epsg(epsg_code)
        if not all([x.unit_name == "metre" for x in target_crs.axis_info]):
            raise Exception("Target CRS has to use metres on both axis")
        return target_crs

    def get_route_geojson(self, route, visited_transect_names, transformer):
        coordinates = []
        features = []
        # create a geojson for the route
        # region
        pre_lng = None
        pre_lat = None
        for position_idx, position in enumerate(route):
            lng, lat = transformer.transform(position[0], position[1])
            coordinates.append((lat, lng))
            features.append(
                {
                    "type": "Feature",
                    "properties": {"name": f"{position_idx}"},
                    "geometry": {"coordinates": [lat, lng], "type": "Point"},
                }
            )
            if position_idx > 0:
                transect_id = visited_transect_names[position_idx - 1]
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"name": f"{transect_id}"},
                        "geometry": {
                            "coordinates": [(pre_lat, pre_lng), (lat, lng)],
                            "type": "LineString",
                        },
                    }
                )
            pre_lat = lat
            pre_lng = lng
        features.append(
            {
                "type": "Feature",
                "properties": {"name": "total-route"},
                "geometry": {"coordinates": coordinates, "type": "LineString"},
            }
        )
        route_geojson = {"type": "FeatureCollection", "features": features}
        return route_geojson

    def save_total_route_kml(
        self,
        route,
        visited_transect_names,
        transformer,
        target_file,
        add_start_end=True,
    ):

        geojson = self.get_route_geojson(route, visited_transect_names, transformer)

        total_routes = [
            x for x in geojson["features"] if x["properties"]["name"] == "total-route"
        ]
        if len(total_routes) == 0:
            raise Exception("No total route found in geojson. Cannot save kml.")
            return

        transect_names_str = "transects-" + "-".join(visited_transect_names)

        # see https://stackoverflow.com/questions/71113897/converting-geojson-to-kml-using-python
        kml = simplekml.Kml()

        if add_start_end:
            kml.newpoint(
                name="Start",
                coords=[
                    (
                        total_routes[0]["geometry"]["coordinates"][0][0],
                        total_routes[0]["geometry"]["coordinates"][0][1],
                    )
                ],
                description="Start of the route",
            )

        kml.newlinestring(
            name="Total Route",
            coords=total_routes[0]["geometry"]["coordinates"],
            description=transect_names_str,
        )

        if add_start_end:
            kml.newpoint(
                name="End",
                coords=[
                    (
                        total_routes[0]["geometry"]["coordinates"][-1][0],
                        total_routes[0]["geometry"]["coordinates"][-1][1],
                    )
                ],
                description="End of the route",
            )

        kml.save(target_file)

    def calculate_grid(
        self,
        flight_area,
        grid_size,
        invalid_areas,
        transformer,
        min_transect_overlap,
        target_path: Optional[str] = None,
        x_offset: int = 0,
        y_offset: int = 0,
        padding_north: int = 0,
        padding_east: int = 0,
        padding_south: int = 0,
        padding_west: int = 0,
    ):
        flight_area_boundary = flight_area.bounds
        bottom_left_latitude = (
            flight_area_boundary[0] + x_offset - abs(padding_west) * grid_size
        )
        bottom_left_longitude = (
            flight_area_boundary[1] + y_offset - abs(padding_south) * grid_size
        )
        top_right_latitude = (
            flight_area_boundary[2] + x_offset + abs(padding_east) * grid_size
        )
        top_right_longitude = (
            flight_area_boundary[3] + y_offset + abs(padding_north) * grid_size
        )

        # calculate the latitude positions of the grid based on the grid size
        current_latitude = bottom_left_latitude
        intermediate_latitudes = [current_latitude]
        while current_latitude <= top_right_latitude:
            current_latitude += grid_size
            intermediate_latitudes.append(current_latitude)

        # calculate the longitude positions of the grid based on the grid size
        current_longitude = bottom_left_longitude
        intermediate_longitudes = [current_longitude]
        while current_longitude <= top_right_longitude:
            current_longitude += grid_size
            intermediate_longitudes.append(current_longitude)

        # create geojsons for the grid points (all and the filtere dones)
        # region
        grid_point_geojson = {"type": "FeatureCollection", "features": []}
        all_grid_point_geojson = {"type": "FeatureCollection", "features": []}
        # endregion

        # now lets build the actual grid and also filter valid and invalid nodes
        valid_nodes = []
        grid = []
        width = len(intermediate_latitudes)
        height = len(intermediate_longitudes)
        for x, latitude in enumerate(intermediate_latitudes):
            row = []
            for y, longitude in enumerate(intermediate_longitudes):
                # check if latitude, longitude is within area
                p = shapely.geometry.Point(latitude, longitude)
                is_valid = p.within(flight_area)
                if is_valid:
                    valid_nodes.append(GridPosition(latitude, longitude, x, y))
                row.append([latitude, longitude, is_valid, False])
                # add the point to the grid geojsons
                new_lng, new_lat = transformer.transform(latitude, longitude)

                # region
                feature = {
                    "type": "Feature",
                    "properties": {"name": f"{x}-{y}"},
                    "geometry": {"coordinates": [new_lat, new_lng], "type": "Point"},
                }
                all_grid_point_geojson["features"].append(feature)
                if is_valid:
                    grid_point_geojson["features"].append(feature)
                # endregion
            grid.append(row)

        # create the grid files
        if target_path is not None:
            with open(os.path.join(target_path, "grid.geojson"), "w") as f:
                json.dump(all_grid_point_geojson, f)
            with open(os.path.join(target_path, "grid_filtered.geojson"), "w") as f:
                json.dump(grid_point_geojson, f)

        # based on the grid lets build up the flight transects
        transects = {}
        for x in range(0, width):
            for y in range(0, height):
                # create horizontal transects
                if y + 1 < height:
                    # create the transect
                    start = grid[x][y]
                    end = grid[x][y + 1]

                    line = shapely.geometry.LineString([start[0:2], end[0:2]])
                    # check the intersection ratio of the transect with the actual flight area
                    # if ratio > min_transect_overlap it is intended as a valid transect
                    intersection = shapely.intersection(line, flight_area)
                    ratio = intersection.length / line.length
                    transects[f"{x}_{y}_{x}_{y + 1}"] = TransectStatus(
                        ratio > min_transect_overlap, False
                    )

                    # if we have got no-go areas, we have to check them too
                    if invalid_areas is not None:
                        # check if there is any intersection between our transect and one of the no-go areas
                        for geometry in invalid_areas.geometry:
                            intersection = shapely.intersection(line, geometry)
                            if intersection.length > 0:
                                transects[f"{x}_{y}_{x}_{y + 1}"] = TransectStatus(
                                    False, False
                                )
                                break
                # create vertical transects
                if x + 1 < width:
                    # create the transect
                    start = grid[x][y]
                    end = grid[x + 1][y]
                    line = shapely.geometry.LineString([start[0:2], end[0:2]])
                    # check the intersection ratio of the transect with the actual flight area
                    # if ratio > min_transect_overlap it is intended as a valid transect
                    intersection = shapely.intersection(line, flight_area)
                    ratio = intersection.length / line.length
                    transects[f"{x}_{y}_{x + 1}_{y}"] = TransectStatus(
                        ratio > min_transect_overlap, False
                    )

                    # if we have got no-go areas, we have to check them too
                    if invalid_areas is not None:
                        # check if there is any intersection between our transect and one of the no-go areas
                        for geometry in invalid_areas.geometry:
                            intersection = shapely.intersection(line, geometry)
                            if intersection.length > 0:
                                transects[f"{x}_{y}_{x + 1}_{y}"] = TransectStatus(
                                    False, False
                                )
                                break

        # create geo jsons for the transect lines (all and valid)
        # region
        transect_features = []
        transect_features_valid = []

        for key, value in transects.items():
            start_x, start_y, end_x, end_y = key.split("_")
            start_point = grid[int(start_x)][int(start_y)]
            start_lng, start_lat = transformer.transform(start_point[0], start_point[1])
            end_point = grid[int(end_x)][int(end_y)]
            end_lng, end_lat = transformer.transform(end_point[0], end_point[1])
            transect_line = {
                "type": "Feature",
                "properties": {"name": key},
                "geometry": {
                    "coordinates": [[start_lat, start_lng], [end_lat, end_lng]],
                    "type": "LineString",
                },
            }
            transect_features.append(transect_line)
            if value[0]:
                transect_features_valid.append(transect_line)

        if target_path is not None:
            with open(os.path.join(target_path, "transects.geojson"), "w") as f:
                json.dump(
                    {"type": "FeatureCollection", "features": transect_features}, f
                )
            with open(os.path.join(target_path, "transects_valids.geojson"), "w") as f:
                json.dump(
                    {"type": "FeatureCollection", "features": transect_features_valid},
                    f,
                )
        # endregion

        return grid, valid_nodes, transects, width, height

    def create_startpoints_geojson(
        self, start_points, transformer, target_path: Optional[str]
    ):
        # region
        start_geojson = {"type": "FeatureCollection", "features": []}
        for start_point_idx, start_point in enumerate(start_points):
            lng, lat = transformer.transform(start_point[1], start_point[2])
            feature = {
                "type": "Feature",
                "properties": {"name": f"start-{start_point_idx}"},
                "geometry": {"coordinates": [lat, lng], "type": "Point"},
            }
            start_geojson["features"].append(feature)

        if target_path is not None:
            shutil.rmtree(target_path, ignore_errors=True)
            os.makedirs(target_path, exist_ok=True)
            with open(os.path.join(target_path, "startpoints.geojson"), "w") as f:
                json.dump(start_geojson, f)
        # endregion


class RandomStrategy(EvaluationFlightStrategy):
    """
    Implementation of the "Random" strategy creating a set of completely random flights
    as basis for the actual planning.
    """

    def __init__(
        self,
        grid_size: float = 400.0,
        max_start_and_stop_distance: float = 3000.0,
        min_transects: int = 40,
        max_transects: Optional[int] = None,
        max_number_of_overlapping_transects: int = 0,
        max_distance: float = 2000.0,
        min_transect_overlap: float = 0.75,
        number_of_retries: int = 100,
        max_number_of_flights: int = 100,
        target_crs_epsg: int = 32633,
        min_transects_per_route: int = 3,
        x_offset: float = 0,
        y_offset: float = 0,
        padding_north: int = 0,
        padding_east: int = 0,
        padding_south: int = 0,
        padding_west: int = 0,
        random_search: bool = True,
        seed: Optional[int] = None,
    ):
        """
        :param grid_size: Horizontal/Vertical distance between individual grid points
        :param max_start_and_stop_distance: Max. distance allowed between selected start point and first route point
            + selected start point and last route point
        :param min_transects: Min. number of expected transects within all routes of a plan
        :param max_transects: Optional upper bound on transects. When set, plan construction stops
            as soon as the transect count is in [min_transects, max_transects]. If None the
            behaviour is unchanged (all valid flights are accumulated until min_transects is met).
        :param max_number_of_overlapping_transects: Ratio (0 to 1; based on min_transects) defining the maximum
            number of transects that may overlap (re-used) within a flight.
        :param max_distance: Max. distance of a route (without start and stop distance)
        :param min_transect_overlap: Min. overlap ratio of a transect with the area to be assumed as valid
        :param number_of_retries: Max. number of retries to create a plan
        :param max_number_of_flights: Number of flights that should be generated for the base collection
            from which the final plan will be selected
        :param target_crs_epsg: EPSG code of the UTM tile of the area of interest needed for metric based
            calculations
        :param min_transects_per_route: Min. number of transects per route
        :param x_offset: Offset of the grid along the West/East Axis. Positive value moves the grid to East.
            Negative value to west.
        :param y_offset: Offset of the grid along the North/South Axis. Positive value moves the grid to North.
            Negative value to South.
        :param padding_north: Add x additional lines of grid points to the North of the flight area
            (useful, y_offset is used)
        :param padding_east: Add x additional lines of grid points to the East of the flight area
            (useful, x_offset is used)
        :param padding_south: Add x additional lines of grid points to the South of the flight area
            (useful, y_offset is used)
        :param padding_west: Add x additional lines of grid points to the West of the flight area
            (useful, x_offset is used)
        :param random_search: Flag if random search should be used or sorted search based on the length of the
            generated sorts for selecting the final flights for the flight plan
        :param seed: Seed information used for random process; If none random seed is used
        """

        self.__padding_west = padding_west
        self.__padding_south = padding_south
        self.__padding_east = padding_east
        self.__padding_north = padding_north
        self.__y_offset = y_offset
        self.__x_offset = x_offset
        self.__max_number_of_flights = max_number_of_flights
        self.__min_transects = min_transects
        self.__max_transects = max_transects
        self.__max_number_of_overlapping_transects = max_number_of_overlapping_transects
        self.__min_transects_per_route = min_transects_per_route
        self.__target_crs_epsg = target_crs_epsg
        self.__min_transect_overlap = min_transect_overlap
        self.__number_of_retries = number_of_retries
        self.__max_distance = max_distance
        self.__max_start_and_stop_distance = max_start_and_stop_distance
        self.__grid_size = grid_size
        self.__random_search = random_search
        if seed is None:
            self.__seed = random.randrange(sys.maxsize)
            self.__random = random.Random(self.__seed)
        else:
            self.__seed = seed
            self.__random = random.Random(seed)

    def create_routes(
        self,
        area_path: str,
        start_points_path: str,
        target_path: str,
        invalid_areas_path: Optional[str] = None,
    ) -> List:
        target_crs = self.get_crs(self.__target_crs_epsg)
        transformer = Transformer.from_crs(target_crs, CRS.from_epsg(4326))

        flight_areas, start_points_df, invalid_areas = self.read_data(
            area_path, start_points_path, target_crs, invalid_areas_path
        )

        start_points = start_points_df.geometry.get_coordinates().values.tolist()
        start_points = [[idx, x[0], x[1]] for idx, x in enumerate(start_points)]

        # create the startpoints.geojson
        self.create_startpoints_geojson(start_points, transformer, target_path)
        route_base_path = os.path.join(target_path, "routes")
        os.makedirs(os.path.join(route_base_path, "valid"), exist_ok=True)
        os.makedirs(os.path.join(route_base_path, "valid_kml"), exist_ok=True)
        os.makedirs(os.path.join(route_base_path, "invalid"), exist_ok=True)
        os.makedirs(os.path.join(route_base_path, "all"), exist_ok=True)

        # based on the flight area build up the evaluation grid
        grid, valid_nodes, transects, width, height = self.calculate_grid(
            flight_areas.geometry[0],
            self.__grid_size,
            invalid_areas,
            transformer,
            self.__min_transect_overlap,
            target_path,
            self.__x_offset,
            self.__y_offset,
            self.__padding_north,
            self.__padding_east,
            self.__padding_south,
            self.__padding_west,
        )

        # definition of the flight pattern:
        flight_pattern = {
            "north": (0, 1),  # go upwards
            "east": (1, 0),  # go right
            "south": (0, -1),  # downwards
            "west": (-1, 0),  # go left
        }
        inverse_direction = {
            "north": "south",
            "south": "north",
            "west": "east",
            "east": "west",
        }

        with open(os.path.join(target_path, "log.txt"), "w") as logfile:
            logfile.write("Parameters:\n")

            parameters_to_print = copy.deepcopy(self.__dict__)
            del parameters_to_print["_RandomStrategy__random"]
            for key, value in parameters_to_print.items():
                logfile.write(f"{key[len('_RandomStrategy__'):]}={value},\n")

            logfile.write("\n-----------------------------------\n")
            number_of_valid_transects = len([x for x in transects.values() if x.is_valid is True])

            logfile.write(f"Found {number_of_valid_transects} valid transects for the area.\n")
            logfile.write("-----------------------------------\n")

            flights = []
            route_idx = 0
            while len(flights) < self.__max_number_of_flights:
                grid_copy = copy.deepcopy(grid)
                transects_copy = copy.deepcopy(transects)
                possible_starts = []
                for node in valid_nodes:
                    # check if position has already been used within the route planning
                    grid_pos = grid_copy[node[2]][node[3]]
                    if grid_pos[3]:
                        continue
                    possible_starts.append(node)

                if len(possible_starts) == 0:
                    logfile.write(
                        "\nNo possible start points available, can't create new routes. Stopping...\n"
                    )
                    break
                # randomly select a grid position as first route position
                first_idx = self.__random.randint(0, len(possible_starts) - 1)
                first_point = possible_starts[first_idx]
                first_point_x = first_point[2]
                first_point_y = first_point[3]
                logfile.write(
                    f"\nCreating route {route_idx} starting at gridpoint {first_point_x}/{first_point_y}\n"
                )
                grid_copy[first_point_x][first_point_y][3] = True
                current_distance = 0
                route = [first_point]
                new_visited_points = [[first_point_x, first_point_y]]
                visited_transects = []
                dead_end = False

                length_stop = False
                last_direction = None
                while (
                    current_distance <= self.__max_distance
                    and not dead_end
                    and not length_stop
                ):
                    # get the last route position ...
                    last_point = route[-1]
                    # ... and its position on the grid
                    last_point_x = last_point[2]
                    last_point_y = last_point[3]

                    # create a copy of the flight pattern and add the first step as additional fifth step
                    temp_flight_pattern = flight_pattern.copy()

                    # now try the flight pattern and check if there is any dead end (no direction applicable)

                    possible_directions = []
                    for direction_idx, direction in temp_flight_pattern.items():
                        new_x = last_point_x + direction[0]
                        new_y = last_point_y + direction[1]
                        min_x = min(last_point_x, new_x)
                        min_y = min(last_point_y, new_y)
                        max_x = max(last_point_x, new_x)
                        max_y = max(last_point_y, new_y)
                        # check if the new position is valid at all for our grid
                        if (
                            new_x >= 0
                            and new_y >= 0
                            and new_x < width
                            and new_y < height
                        ):
                            transect_key = f"{min_x}_{min_y}_{max_x}_{max_y}"
                            transect = transects_copy[transect_key]
                            if transect[0] and not transect[1]:
                                possible_directions.append(direction_idx)

                    if len(possible_directions) == 0:
                        logfile.write("-- Found dead end\n")
                        dead_end = True
                        break

                    next_direction_idx = None
                    while next_direction_idx is None:
                        random_index = self.__random.randint(
                            0, len(possible_directions) - 1
                        )
                        tmp_direction_idx = possible_directions[random_index]
                        if (
                            last_direction is not None
                            and tmp_direction_idx == inverse_direction[last_direction]
                        ):
                            continue
                        next_direction_idx = tmp_direction_idx

                    direction = flight_pattern[next_direction_idx]
                    last_direction = next_direction_idx
                    # calculate the new position based on the last grid position and the current flight direction
                    new_x = last_point_x + direction[0]
                    new_y = last_point_y + direction[1]

                    # check if the transect defined by the last position and the new position is:
                    # - valid at all
                    # - has not been visited before!
                    # to do so, check the transects dictionary
                    min_x = min(last_point_x, new_x)
                    min_y = min(last_point_y, new_y)
                    max_x = max(last_point_x, new_x)
                    max_y = max(last_point_y, new_y)

                    transect_key = f"{min_x}_{min_y}_{max_x}_{max_y}"
                    transect = transects_copy[transect_key]

                    logfile.write(
                        f"-- Going to next transect ({next_direction_idx}) between grid positions "
                        f"{min_x}/{min_y} and {max_x}/{max_y}\n"
                    )
                    # if the transect is valid proceed and get the next global position from the grid
                    next_point = grid_copy[new_x][new_y]

                    # now update our distances
                    # update the current flight length based on the distance from the last to the new point
                    distance = math.sqrt(
                        (last_point[1] - next_point[1]) ** 2
                        + (last_point[0] - next_point[0]) ** 2
                    )

                    # if we are exceeding the max distance lets check the next direction
                    if current_distance + distance > self.__max_distance:
                        logfile.write(
                            f"-- Exceeding max. route distance with {current_distance + distance}m \n"
                        )
                        length_stop = True
                        break
                    current_distance += distance

                    transects_copy[transect_key] = transects_copy[
                        transect_key
                    ]._replace(already_visited=True)
                    visited_transects.append(transect_key)
                    # otherwise we have found a new valid route position
                    last_point = GridPosition(
                        next_point[0], next_point[1], new_x, new_y
                    )
                    # update the grid meta information, showing if the position has been visited once
                    if not grid_copy[new_x][new_y][3]:
                        new_visited_points.append([new_x, new_y])
                    grid_copy[new_x][new_y][3] = True
                    route.append(last_point)
                num_transects = len(visited_transects)
                if num_transects > self.__min_transects_per_route:
                    logfile.write("-- Created valid route \n")
                    flights.append(
                        {
                            "route": route,
                            "transects": visited_transects,
                            "idx": route_idx,
                            "start_point": None,
                        }
                    )
                    route_idx += 1
                else:
                    logfile.write("-- Created invalid route. Too few transects. \n")

            logfile.write("\n-----------------------------------\n")
            logfile.write("Trying to assign start points to created routes \n")
            logfile.write("-----------------------------------\n")
            flights_for_start_points = []
            filtered_flights = []
            for route_idx, flight in enumerate(flights):
                route = flight["route"]
                first_point = route[0]
                last_point = route[-1]

                # find the closest start point
                smallest_distance = None
                smallest_start_idx = None
                for start_point_idx, start_point in enumerate(start_points):
                    distance = math.sqrt(
                        (start_point[2] - first_point[1]) ** 2
                        + (start_point[1] - first_point[0]) ** 2
                    )
                    back_distance = math.sqrt(
                        (start_point[2] - last_point[1]) ** 2
                        + (start_point[1] - last_point[0]) ** 2
                    )
                    if back_distance + distance > self.__max_start_and_stop_distance:
                        logfile.write(
                            f"-- Start point {start_point[0]} is invalid because of to long distance "
                            f"{back_distance + distance} to route's first and last points "
                            f"(route {route_idx})\n"
                        )
                        continue  # go to next start point

                    # if we have got no-go areas, we have to check them too
                    if invalid_areas is not None:
                        touching_invalid_area = False
                        line = shapely.geometry.LineString(
                            [start_point[1:3], first_point[0:2]]
                        )
                        # check if there is any intersection between the start of the route and the
                        # selected start point and one of the no-go areas
                        for geometry in invalid_areas.geometry:
                            intersection = shapely.intersection(line, geometry)
                            if intersection.length > 0:
                                logfile.write(
                                    f"-- Start point {start_point[0]} is invalid because of "
                                    f"intersection with no-fly area on route to start of "
                                    f"route {route_idx}\n"
                                )
                                touching_invalid_area = True
                                break
                        if touching_invalid_area:
                            continue  # go to next start point

                        line = shapely.geometry.LineString(
                            [start_point[1:3], last_point[0:2]]
                        )
                        # check if there is any intersection between the end of the route and the
                        # selected start point and one of the no-go areas
                        for geometry in invalid_areas.geometry:
                            intersection = shapely.intersection(line, geometry)
                            if intersection.length > 0:
                                logfile.write(
                                    f"-- Start point {start_point[0]} is invalid because of "
                                    f"intersection with no-fly area on route to end of "
                                    f"route {route_idx}\n"
                                )
                                touching_invalid_area = True
                                break
                        if touching_invalid_area:
                            continue

                    if (
                        smallest_distance is None
                        or distance + back_distance < smallest_distance
                    ):
                        # we have a valid start point
                        logfile.write(
                            f"-- Found suitable new minimal start point {start_point_idx} for route {route_idx}\n"
                        )
                        smallest_start_idx = start_point_idx
                        smallest_distance = distance + back_distance
                        continue  # go back to start point loop

                    # we have a valid start point
                    logfile.write(
                        f"-- Found suitable start point {start_point_idx} for route {route_idx}\n"
                    )

                flight["start_point"] = smallest_start_idx
                # always write routes. Note: it might get overwritten later!
                route_geojson = self.get_route_geojson(
                    flight["route"], flight["transects"], transformer
                )
                flight_idx = flight["idx"]
                with open(
                    os.path.join(route_base_path, "all", f"route_{flight_idx}.geojson"),
                    "w",
                ) as f:
                    json.dump(route_geojson, f)

                # if the route is invalid write it to the invalid folder
                if smallest_start_idx is None:
                    logfile.write(
                        f"-- Could not find a valid start point within the min. start/end distance "
                        f"for route {route_idx}\n"
                    )
                    with open(
                        os.path.join(
                            route_base_path, "invalid", f"route_{flight_idx}.geojson"
                        ),
                        "w",
                    ) as f:
                        json.dump(route_geojson, f)
                else:
                    logfile.write(
                        f"-- Selected start point {smallest_start_idx} for route {route_idx}\n"
                    )
                    flights_for_start_points.append(smallest_start_idx)
                    filtered_flights.append(flight)

            filtered_flights.sort(key=lambda x: len(x["transects"]))

            current_selected_flights = []
            current_selected_transects = set()
            current_try = 0
            filtered_flights_copy = copy.deepcopy(filtered_flights)

            logfile.write("-----------------------------------\n")
            for flight in filtered_flights_copy:
                logfile.write(
                    f"Valid flight {flight['idx']} with transects: {','.join(flight['transects'])} \n"
                )
            logfile.write("-----------------------------------\n")

            while (
                len(current_selected_transects) < self.__min_transects
                and current_try < self.__number_of_retries
            ):
                logfile.write("-----------------------------------\n")
                logfile.write(
                    f"Trying to create flight plan out of plans ouf of "
                    f"{len(filtered_flights_copy)} suitable flights (#{current_try + 1})! \n"
                )
                logfile.write("-----------------------------------\n")
                current_selected_flights = []
                current_selected_transects = set()
                if self.__random_search:
                    self.__random.shuffle(filtered_flights_copy)
                for flight in filtered_flights_copy:
                    num_of_overlaps = 0
                    for transect in flight["transects"]:
                        if transect in current_selected_transects:
                            num_of_overlaps += 1
                    if num_of_overlaps > self.__max_number_of_overlapping_transects:
                        logfile.write(
                            f"Flight {flight['idx']} can't be added because of {num_of_overlaps} "
                            f"overlapping transects (> {self.__max_number_of_overlapping_transects})\n"
                        )
                        continue
                    current_selected_flights.append(flight)
                    for transect in flight["transects"]:
                        current_selected_transects.add(transect)
                    logfile.write(f"Flight {flight['idx']} added\n")

                    # early stopping if min_transects criteria is reached
                    # if len(current_selected_transects) >= self.__min_transects:
                    #     logfile.write(f"Early stopping because min_transects criteria reached: "
                    #                   f"{len(current_selected_transects)} >= {self.__min_transects}\n")
                    #     break  # out of for loop ("for flight_idx, flight in ...")

                    # early stopping when max_transects is set and count is already in [min, max]
                    if (
                        self.__max_transects is not None
                        and len(current_selected_transects) >= self.__min_transects
                    ):
                        logfile.write(
                            f"Early stopping: transect count {len(current_selected_transects)} "
                            f"reached [{self.__min_transects}, {self.__max_transects}] window\n"
                        )
                        break

                if not self.__random_search:
                    filtered_flights_copy = filtered_flights_copy[1:]

                if len(current_selected_transects) < self.__min_transects:
                    logfile.write(
                        f"Min_transects criteria not reached. Currently selected "
                        f"{len(current_selected_transects)} unique of a minimum "
                        f"{self.__min_transects} transects.\n"
                    )

                current_try += 1

            res = []
            exceeds_max = (
                self.__max_transects is not None
                and len(current_selected_transects) > self.__max_transects
            )
            if (
                len(current_selected_transects) < self.__min_transects
                or exceeds_max
                or current_try >= self.__number_of_retries
            ):
                logfile.write("-----------------------------------\n")
                logfile.write("Could not create a valid plan! \n")
                logfile.write("-----------------------------------\n")
            else:
                logfile.write("-----------------------------------\n")
                logfile.write("Created flight plan! \n")
                logfile.write(
                    f"Number of unique transects: {len(current_selected_transects)} \n"
                )
                logfile.write(
                    f"Valid routes: {[x['idx'] for x in current_selected_flights]} \n"
                )
                for flight in current_selected_flights:
                    res.append(flight["route"])
                    flight_idx = flight["idx"]
                    start_point_idx = flight["start_point"]
                    os.makedirs(
                        os.path.join(route_base_path, f"{start_point_idx}"),
                        exist_ok=True,
                    )
                    start_point = start_points[start_point_idx]
                    new_point = [start_point[1], start_point[2], -1, -1]
                    route = flight["route"]
                    flight["route"] = [new_point] + route + [new_point]
                    transects = ["arrival"] + flight["transects"] + ["departure"]
                    route_geojson = self.get_route_geojson(
                        flight["route"], transects, transformer
                    )
                    with open(
                        os.path.join(
                            route_base_path,
                            f"{start_point_idx}",
                            f"route_{flight_idx}.geojson",
                        ),
                        "w",
                    ) as f:
                        json.dump(route_geojson, f)
                    with open(
                        os.path.join(
                            route_base_path, "all", f"route_{flight_idx}.geojson"
                        ),
                        "w",
                    ) as f:
                        json.dump(route_geojson, f)
                    with open(
                        os.path.join(
                            route_base_path, "valid", f"route_{flight_idx}.geojson"
                        ),
                        "w",
                    ) as f:
                        json.dump(route_geojson, f)

                    # also save KML
                    self.save_total_route_kml(
                        flight["route"],
                        transects,
                        transformer,
                        os.path.join(
                            route_base_path, "valid_kml", f"route_{flight_idx}.kml"
                        ),
                    )

                logfile.write("-----------------------------------\n")
            return res


class RandomLoopStrategy(EvaluationFlightStrategy):
    """
    Implementation of the "Random Loop" strategy using a round robin based approach for the
    creation of flight routes by trying to go the north, then to east, then to south and then to west. In the next try,
    circle this pattern starting with east, south, west, north and so on etc.

    Select random start positions within the flight area grid and create routes based on this approach until you reach
    the max distance or your ending up in a dead end.

    Take all the routes created and find the closest start position. If the start position is too far away from the
    route's first or last point mark it as invalid, otherwise you have found a valid route.
    """

    def __init__(
        self,
        grid_size: float = 400.0,
        max_start_and_stop_distance: float = 2000.0,
        min_transects: int = 40,
        max_transects: Optional[int] = None,
        max_distance: float = 3000.0,
        min_transect_overlap: float = 0.75,
        number_of_retries: int = 50,
        number_of_retries_per_route: int = 50,
        target_crs_epsg: int = 32633,
        min_transects_per_route: int = 3,
        x_offset: int = 0,
        y_offset: int = 0,
        padding_north: int = 0,
        padding_east: int = 0,
        padding_south: int = 0,
        padding_west: int = 0,
        seed: Optional[int] = None,
    ):
        """
        :param grid_size: Horizontal/Vertical distance between individual grid points
        :param max_start_and_stop_distance: Max. distance allowed between selected start point and first route point
            + selected start point and last route point
        :param min_transects: Min. number of expected transects within all routes of a plan
        :param max_transects: Optional upper bound on transects. When set, route accumulation stops
            as soon as the transect count is in [min_transects, max_transects]. If None the
            behaviour is unchanged (routes are accumulated until min_transects is met).
        :param max_distance: Max. distance of a route (without start and stop distance)
        :param min_transect_overlap: Min. overlap ratio of a transect with the area to be assumed as valid
        :param number_of_retries: Max. number of retries to create a plan
        :param number_of_retries_per_route: Max. number of retries to create an individual route
        :param target_crs_epsg: EPSG code of the UTM tile of the area of interest needed for metric based
            calculations
        :param min_transects_per_route: Min. number of transects per route
        :param x_offset: Offset of the grid along the West/East Axis. Positive value moves the grid to East.
            Negative value to west.
        :param y_offset: Offset of the grid along the North/South Axis. Positive value moves the grid to North.
            Negative value to South.
        :param padding_north: Add x additional lines of grid points to the North of the flight area
            (useful, y_offset is used)
        :param padding_east: Add x additional lines of grid points to the East of the flight area
            (useful, x_offset is used)
        :param padding_south: Add x additional lines of grid points to the South of the flight area
            (useful, y_offset is used)
        :param padding_west: Add x additional lines of grid points to the West of the flight area
            (useful, x_offset is used)
        :param seed: Seed information used for random process; If none random seed is used
        """
        self.__min_transects = min_transects
        self.__max_transects = max_transects
        self.__min_transects_per_route = min_transects_per_route
        self.__target_crs_epsg = target_crs_epsg
        self.__number_of_retries_per_route = number_of_retries_per_route
        self.__min_transect_overlap = min_transect_overlap
        self.__number_of_retries = number_of_retries
        self.__max_distance = max_distance
        self.__max_start_and_stop_distance = max_start_and_stop_distance
        self.__grid_size = grid_size
        self.__padding_west = padding_west
        self.__padding_south = padding_south
        self.__padding_east = padding_east
        self.__padding_north = padding_north
        self.__y_offset = y_offset
        self.__x_offset = x_offset
        if seed is None:
            self.__seed = random.randrange(sys.maxsize)
            self.__random = random.Random(self.__seed)
        else:
            self.__seed = seed
            self.__random = random.Random(seed)

    def create_routes(
        self,
        area_path,
        start_points_path,
        target_path,
        invalid_areas_path: Optional[str] = None,
    ) -> List:
        """
        :param area_path: Path to the shape file (.shp, .kml, ...) representing the flight area
        :param start_points_path: Path to the shape file (.shp, .kml, ...) representing the start points
        :param target_path: Path to the target folder, were output files should be created
        :param invalid_areas_path: Path to the shape file (.shp, .kml, ...) representing the no-flight area
        :return: Valid routes__
        """
        target_crs = self.get_crs(self.__target_crs_epsg)
        transformer = Transformer.from_crs(target_crs, CRS.from_epsg(4326))

        flight_areas, start_points_df, invalid_areas = self.read_data(
            area_path, start_points_path, target_crs, invalid_areas_path
        )

        start_points = start_points_df.geometry.get_coordinates().values.tolist()
        start_points = [[idx, x[0], x[1]] for idx, x in enumerate(start_points)]

        # create the startpoints.geojson
        self.create_startpoints_geojson(start_points, transformer, target_path)

        # based on the flight area build up the evaluation grid
        grid, valid_nodes, transects, width, height = self.calculate_grid(
            flight_areas.geometry[0],
            self.__grid_size,
            invalid_areas,
            transformer,
            self.__min_transect_overlap,
            target_path,
            self.__x_offset,
            self.__y_offset,
            self.__padding_north,
            self.__padding_east,
            self.__padding_south,
            self.__padding_west,
        )

        # definition of the flight pattern:
        flight_pattern = [
            (0, 1),  # go upwards
            (1, 0),  # go right
            (0, -1),  # downwards
            (-1, 0),  # go left
        ]

        with open(os.path.join(target_path, "log.txt"), "w") as logfile:
            logfile.write("Parameters:\n")
            logfile.write(f"-- seed: {self.__seed}\n")
            logfile.write(f"-- flight area: {os.path.basename(area_path)}\n")
            logfile.write(
                f"-- start positions: {os.path.basename(start_points_path)}\n"
            )
            invalid_areas_name = (
                os.path.basename(invalid_areas_path)
                if invalid_areas_path is not None
                else invalid_areas_path
            )
            logfile.write(f"-- invalid areas: {invalid_areas_name}\n")
            logfile.write(f"-- grid_size: {self.__grid_size}\n")
            logfile.write(
                f"-- max_start_and_stop_distance : {self.__max_start_and_stop_distance }\n"
            )
            logfile.write(f"-- max_distance: {self.__max_distance}\n")
            logfile.write(
                f"-- min_transects per route: {self.__min_transects_per_route}\n"
            )
            logfile.write(f"-- min_transects per plan: {self.__min_transects}\n")
            logfile.write(f"-- min_transect_overlap: {self.__min_transect_overlap}\n")
            logfile.write(f"-- number_of_retries: {self.__number_of_retries}\n")
            logfile.write(f"-- x_offset: {self.__x_offset}\n")
            logfile.write(f"-- y_offset: {self.__y_offset}\n")
            logfile.write(f"-- padding_north: {self.__padding_north}\n")
            logfile.write(f"-- padding_east: {self.__padding_east}\n")
            logfile.write(f"-- padding_south: {self.__padding_south}\n")
            logfile.write(f"-- padding_west: {self.__padding_west}\n")

            # prepare some variables for the result and checking its validity
            res = []
            transect_count = 0
            planning_retries = 0
            valid_routes = 0
            # try to create a plan with routes until you reach the moment of a valid plan
            # (min. number of transects reached) or you touched the max. number of retries
            while (
                transect_count < self.__min_transects
                and planning_retries < self.__number_of_retries
            ):
                logfile.write("--------------------------\n")
                logfile.write(f"Plan creation try {planning_retries} \n")
                logfile.write("--------------------------\n")
                grid_copy = copy.deepcopy(grid)
                transects_copy = copy.deepcopy(transects)
                res = []
                res_transects = []
                shutil.rmtree(os.path.join(target_path, "routes"), ignore_errors=True)
                transect_count = 0
                # prepare dead_end_counter
                dead_end_counter = 0
                route_idx = 0

                # while there are grid positions, find a random route
                # stop this process when you can't find any new route in "number_of_retries" subsequent tries
                while dead_end_counter < self.__number_of_retries_per_route:
                    # find possible first position on grid
                    possible_starts = []
                    for node in valid_nodes:
                        # check if position has already been used within the route planning
                        grid_pos = grid_copy[node[2]][node[3]]
                        if grid_pos[3]:
                            continue
                        possible_starts.append(node)

                    if len(possible_starts) == 0:
                        logfile.write(
                            "\nNo possible start points available, can't create new routes. "
                            "Stopping...\n"
                        )
                        break
                    # randomly select a grid position as first route position
                    first_idx = self.__random.randint(0, len(possible_starts) - 1)
                    first_point = possible_starts[first_idx]
                    first_point_x = first_point[2]
                    first_point_y = first_point[3]
                    logfile.write(
                        f"\nTrying to create route {route_idx} starting at gridpoint {first_point_x}/{first_point_y}\n"
                    )
                    grid_copy[first_point_x][first_point_y][3] = True
                    current_distance = 0
                    route = [first_point]
                    new_visited_points = [[first_point_x, first_point_y]]
                    visited_transects = []
                    # while we have not found too many dead ends or reached the max. distance
                    # continue finding positions for the route
                    dead_end_stop = False
                    length_stop = False
                    while (
                        current_distance <= self.__max_distance
                        and not dead_end_stop
                        and not length_stop
                    ):
                        # get the last route position ...
                        last_point = route[-1]
                        # ... and its position on the grid
                        last_point_x = last_point[2]
                        last_point_y = last_point[3]

                        # create a copy of the flight pattern and add the first step as additional fifth step
                        temp_flight_pattern = flight_pattern.copy()
                        temp_flight_pattern.append(flight_pattern[0])

                        # now try the flight pattern and check if there is any dead end (no direction applicable)
                        dead_end = True
                        for direction_idx, direction in enumerate(temp_flight_pattern):
                            direction_str = ",".join([str(x) for x in direction])
                            if direction_str == "0,1":
                                direction_name = "North"
                            elif direction_str == "1,0":
                                direction_name = "East"
                            elif direction_str == "-1,0":
                                direction_name = "West"
                            elif direction_str == "0,-1":
                                direction_name = "South"
                            else:
                                # Should actually never happen
                                direction_name = "Error"
                            # calculate the new position based on last grid position and flight direction
                            new_x = last_point_x + direction[0]
                            new_y = last_point_y + direction[1]

                            # check if the transect defined by the last position and the new position is:
                            # - valid at all
                            # - has not been visited before!
                            # to do so, check the transects dictionary
                            min_x = min(last_point_x, new_x)
                            min_y = min(last_point_y, new_y)
                            max_x = max(last_point_x, new_x)
                            max_y = max(last_point_y, new_y)

                            transect_key = f"{min_x}_{min_y}_{max_x}_{max_y}"

                            # check if the new position is valid at all for our grid
                            if (
                                new_x < 0
                                or new_y < 0
                                or new_x >= width
                                or new_y >= height
                            ):
                                logfile.write(
                                    f"-- Next transect ({direction_name}) between grid positions "
                                    f"{min_x}/{min_y} and {max_x}/{max_y} is invalid\n"
                                )
                                continue

                            transect = transects_copy[transect_key]

                            # transect[0] defines if the transect is valid at all
                            if not transect[0]:
                                logfile.write(
                                    f"-- Next transect ({direction_name}) between grid positions "
                                    f"{min_x}/{min_y} and {max_x}/{max_y} is invalid\n"
                                )
                                continue
                            # transect[1] defines if the transect has been visited before
                            if transect[1]:
                                logfile.write(
                                    f"-- Next transect ({direction_name}) between grid positions "
                                    f"{min_x}/{min_y} and {max_x}/{max_y} has already been visited\n"
                                )
                                continue
                            logfile.write(
                                f"-- Going to next transect ({direction_name}) between grid positions "
                                f"{min_x}/{min_y} and {max_x}/{max_y}\n"
                            )
                            # if the transect is valid proceed and get the next global position from the grid
                            next_point = grid_copy[new_x][new_y]

                            # now update our distances
                            # update the current flight length based on the distance from the last to the new point
                            distance = math.sqrt(
                                (last_point[1] - next_point[1]) ** 2
                                + (last_point[0] - next_point[0]) ** 2
                            )

                            # if we are exceeding the max distance lets check the next direction
                            if current_distance + distance > self.__max_distance:
                                logfile.write(
                                    f"-- Exceeding max. route distance with {current_distance + distance}m \n"
                                )
                                length_stop = True
                                break
                            current_distance += distance

                            dead_end = False
                            transects_copy[transect_key] = transects_copy[
                                transect_key
                            ]._replace(already_visited=True)
                            visited_transects.append(transect_key)
                            # otherwise we have found a new valid route position
                            last_point = [next_point[0], next_point[1], new_x, new_y]
                            # update the grid meta information, showing if the position has been visited once
                            if not grid_copy[new_x][new_y][3]:
                                new_visited_points.append([new_x, new_y])
                            grid_copy[new_x][new_y][3] = True
                            last_point_x = new_x
                            last_point_y = new_y
                            route.append(last_point)
                        if dead_end:
                            logfile.write("-- Found dead end\n")
                            dead_end_stop = True
                        # move on in the flight pattern and start with the next direction for the next iteration
                        flight_pattern.append(flight_pattern.pop(0))
                    if dead_end_stop and not length_stop:
                        dead_end_counter += 1
                    else:
                        dead_end_counter = 0

                    num_transects = len(visited_transects)
                    if len(visited_transects) >= self.__min_transects_per_route:
                        logfile.write(
                            f"-- Created route {route_idx} with {num_transects} transects\n"
                        )
                        route_idx += 1
                        res.append(route)
                        res_transects.append(visited_transects)
                    else:
                        logfile.write(
                            f"-- Created route {route_idx} is too short with "
                            f"{num_transects} transects - retrying...\n"
                        )
                        for new_visited_point in new_visited_points:
                            grid_copy[new_visited_point[0]][new_visited_point[1]][
                                3
                            ] = False
                        for visited_transect in visited_transects:
                            transects[visited_transect] = transects[
                                visited_transect
                            ]._replace(already_visited=False)

                logfile.write("\nAssigning routes to closest start positions...\n")

                # now create the final routes and check them
                valid_routes = 0
                for route_idx, route in enumerate(res):
                    first_point = route[0]
                    last_point = route[-1]
                    invalid_route = False

                    # find the closest start point
                    smallest_distance = None
                    smallest_start_idx = None
                    for start_point in start_points:
                        distance = math.sqrt(
                            (start_point[2] - first_point[1]) ** 2
                            + (start_point[1] - first_point[0]) ** 2
                        )
                        back_distance = math.sqrt(
                            (start_point[2] - last_point[1]) ** 2
                            + (start_point[1] - last_point[0]) ** 2
                        )
                        if (
                            back_distance + distance
                            > self.__max_start_and_stop_distance
                        ):
                            logfile.write(
                                f"-- Start point {start_point[0]} too far away for start/end of "
                                f"route {route_idx} with distance {back_distance + distance}m "
                                f"(start distance: {distance}; end distance: {back_distance})\n"
                            )
                            continue

                        # if we have got no-go areas, we have to check them too
                        if invalid_areas is not None:
                            touching_invalid_area = False
                            line = shapely.geometry.LineString(
                                [start_point[0:2], first_point[0:2]]
                            )
                            # check if there is any intersection between the start of the route and
                            # the selected start point and one of the no-go areas
                            for geometry in invalid_areas.geometry:
                                intersection = shapely.intersection(line, geometry)
                                if intersection.length > 0:
                                    logfile.write(
                                        f"-- Start point {start_point[0]} is invalid because of "
                                        f"intersection with no-fly area on route to start of "
                                        f"route {route_idx}\n"
                                    )
                                    touching_invalid_area = True
                                    break
                            if touching_invalid_area:
                                continue

                            line = shapely.geometry.LineString(
                                [start_point[0:2], last_point[0:2]]
                            )
                            # check if there is any intersection between the end of the route
                            # and the selected start point and one of the no-go areas
                            for geometry in invalid_areas.geometry:
                                intersection = shapely.intersection(line, geometry)
                                if intersection.length > 0:
                                    logfile.write(
                                        f"-- Start point {start_point[0]} is invalid because of "
                                        f"intersection with no-fly area on route to end of "
                                        f"route {route_idx}\n"
                                    )
                                    touching_invalid_area = True
                                    break
                            if touching_invalid_area:
                                continue

                        average_distance = (distance + back_distance) / 2
                        if (
                            smallest_distance is None
                            or average_distance < smallest_distance
                        ):
                            smallest_start_idx = start_point[0]
                            smallest_distance = average_distance

                    if smallest_start_idx is None:
                        logfile.write(
                            f"-- Could not find a valid start point within the min. start/end "
                            f"distance for route {route_idx}\n"
                        )
                        invalid_route = True

                    visited_transect_names = res_transects[route_idx]
                    _early_stop = False
                    # if we have found a valid route add the start point at the beginning and end
                    if not invalid_route:
                        closest_start_point = start_points[smallest_start_idx]
                        transect_count += len(route) - 1
                        # early stopping when max_transects is set and count is in [min, max]
                        if (
                            self.__max_transects is not None
                            and transect_count >= self.__min_transects
                        ):
                            logfile.write(
                                f"Early stopping: transect count {transect_count} "
                                f"reached [{self.__min_transects}, {self.__max_transects}] window\n"
                            )
                            _early_stop = True
                        new_point = [
                            closest_start_point[1],
                            closest_start_point[2],
                            -1,
                            -1,
                        ]
                        route = [new_point] + route + [new_point]
                        visited_transect_names = (
                            ["arrival"] + visited_transect_names + ["departure"]
                        )
                        logfile.write(
                            f"-- Valid route {route_idx} with closest start position {smallest_start_idx}\n"
                        )

                    route_geojson = self.get_route_geojson(
                        route, visited_transect_names, transformer
                    )

                    if not invalid_route:
                        os.makedirs(
                            os.path.join(
                                target_path, "routes", f"{smallest_start_idx}"
                            ),
                            exist_ok=True,
                        )
                        with open(
                            os.path.join(
                                target_path,
                                "routes",
                                f"{smallest_start_idx}",
                                f"route_{route_idx}.geojson",
                            ),
                            "w",
                        ) as f:
                            json.dump(route_geojson, f)
                        os.makedirs(
                            os.path.join(target_path, "routes", "valid"), exist_ok=True
                        )
                        with open(
                            os.path.join(
                                target_path,
                                "routes",
                                "valid",
                                f"route_{route_idx}.geojson",
                            ),
                            "w",
                        ) as f:
                            json.dump(route_geojson, f)
                        valid_routes += 1
                    else:
                        os.makedirs(
                            os.path.join(target_path, "routes", "invalid"),
                            exist_ok=True,
                        )
                        with open(
                            os.path.join(
                                target_path,
                                "routes",
                                "invalid",
                                f"route_{route_idx}.geojson",
                            ),
                            "w",
                        ) as f:
                            json.dump(route_geojson, f)
                    os.makedirs(
                        os.path.join(target_path, "routes", "all"), exist_ok=True
                    )
                    with open(
                        os.path.join(
                            target_path, "routes", "all", f"route_{route_idx}.geojson"
                        ),
                        "w",
                    ) as f:
                        json.dump(route_geojson, f)
                    # endregion
                    if _early_stop:
                        break
                planning_retries += 1
            logfile.write("--------------------------------\n")
            logfile.write("--------------------------------\n")
            logfile.write("--------------------------------\n")
            logfile.write(
                f"Created {valid_routes} valid routes covering {transect_count} transects\n"
            )
            return res


if __name__ == "__main__":
    # seed = random.randrange(sys.maxsize)
    # seed = 4640650864630126335
    # print("Seed was:", seed)

    supported_drivers["kml"] = "rw"
    supported_drivers["KML"] = "rw"
    supported_drivers["LIBKML"] = "rw"
    # strategy = RandomStrategy(seed=seed)
    # strategy.create_routes(r"C:\Users\P41743\Desktop\evaluation_plans2\gastein2\UG_Gastein.kml",
    #                      r"C:\Users\P41743\Desktop\evaluation_plans2\gastein2\Startpunkte_Gastein_V0.3.kml",
    #                       r"C:\Users\P41743\Desktop\evaluation_plans2\gastein2\res")#,
    #                        #r"C:\Users\P41743\Desktop\evaluation_plans2\gastein2\UG_Gastein_NOfly.kml")

    # strategy.create_routes(r"C:\Users\P41743\Desktop\evaluation_plans2\dellach\UG_Dellach.kml",
    #                        r"C:\Users\P41743\Desktop\evaluation_plans2\dellach\Startpunkte_Dellach.kml",
    #                        r"C:\Users\P41743\Desktop\evaluation_plans2\dellach\res")

    # strategy.create_routes(r"C:\Users\P41743\Desktop\evaluation_plans\Untersuchungsgebiet_Gastein.shp",
    #                        r"C:\Users\P41743\Desktop\evaluation_plans\Startpunkte_Gastein_V0.3.shp",
    #                        r"C:\Users\P41743\Desktop\evaluation_plans\gastein")

    # strategy.create_routes(r"C:\Users\P41743\Desktop\evaluation_plans2\stjakob\UG_STJakob.kml",
    #                        r"C:\Users\P41743\Desktop\evaluation_plans2\stjakob\Startpunkte_STJakob.kml")

    # strategy = RandomStrategy(
    #     seed=seed,
    #     grid_size=400.0,
    #     max_start_and_stop_distance=3000.0,
    #     min_transects=40,  # ~4 transects per route for 15 routes
    #     max_number_of_overlapping_transects=0, #3/40,
    #     max_distance=1900.0,
    #     min_transect_overlap=1 / 3,  # the lenght of a transect in valid terraiin
    #     number_of_retries=100,
    #     max_number_of_flights=300,
    #     # target_crs_epsg=32633,
    #     min_transects_per_route=3,
    #     x_offset=0,
    #     y_offset=0,
    #     # padding_north: int = 0,
    #     # padding_east: int = 0,
    #     # padding_south: int = 0,
    #     # padding_west: int = 0,
    #     # random_search: bool = True,
    # )

    strategy = RandomStrategy(
        padding_west=0,
        padding_south=0,
        padding_east=0,
        padding_north=0,
        y_offset=90,
        x_offset=40,
        max_number_of_flights=100,
        min_transects=34,
        max_number_of_overlapping_transects=0,
        min_transects_per_route=3,
        target_crs_epsg=32633,
        min_transect_overlap=0.75,
        number_of_retries=300,
        max_distance=1500.0,
        max_start_and_stop_distance=2000.0,
        grid_size=350.0,
        random_search=True,
        seed=8607105702491822030,
    )

    target_folder = r"C:\Users\P41743\Desktop\ST_Feldbach_data\target5"
    # append "route_<date_time>" to the target folder
    # target_folder = os.path.join(
    #     target_folder, f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # )
    os.makedirs(target_folder, exist_ok=True)
    print(f"Target folder: {target_folder}")

    _base = r"C:\Users\P41743\Desktop\ST_Feldbach_data\Feldbach-Straden"
    strategy.create_routes(
        area_path=_base + r"\UG_Feldbach-Straden.kml",
        start_points_path=_base + r"\starting_points_Feldbach-Straden.kml",
        invalid_areas_path=_base + r"\no_fly_Feldbach-Straden.kml",
        target_path=target_folder,
    )

    # print the last lines of the log.txt in the target_folder
    with open(os.path.join(target_folder, "log.txt"), "r") as logfile:
        lines = logfile.readlines()
        for line in lines[-10:]:
            print(line)
