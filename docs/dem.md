# DEM Import & Conversion

The processing pipeline needs a Digital Elevation Model as a GLB/GLTF mesh with a companion metadata JSON (see [DEM metadata JSON](pipeline.md#dem-metadata-json)). The plugin supports five ways to provide one, available as tabs in the **Input** section of the dock widget.

| Where the flight is | What to use |
|---------------------|-------------|
| Austria | **Download DEM (Austria)** - the national 1 m terrain model |
| Anywhere a terrain model can be had as a file (national LiDAR, a survey of your own) | **Convert a GeoTIFF DEM** |
| Anywhere else on land | **Download DEM (worldwide)** - the Copernicus DEM at 30 m |
| Over water | **Flat surface** |

## 1. Automatic download (Austria only)

Use the **Download DEM (Austria)** option to download and tile Austrian elevation data from the BEV ATOM service automatically. The tiles are large (50 km a side) and are kept in a cache, so a second flight in the same region downloads nothing; from them only the flight's own area (plus the padding) is read, also where a flight crosses the edge between two tiles (up to 9.0 that case merged both tiles whole, which took some 20 GB of memory and as much disk space).

**Interrupted downloads.** A tile is several hundred megabytes to gigabytes. It comes in under `<tile>.tif.part` in the cache (`~/.cache/austria_dem`) and gets its name only when it is as long as the server announced and a GeoTIFF whose contents all lie within the file. A connection that breaks is tried again from where it broke; a download that was cancelled, or cut off by closing QGIS, is **continued** by the next one instead of starting at zero. When a tile cannot be fetched the download stops and says so - run it again to continue. A tile already in the cache is checked the same way before it is used. (Up to 9.0 a tile was written straight to its place: a download cut short, or an error page sent in its place, stayed in the cache as the tile and was used by every later run - "Using cached" - until the file was deleted by hand; a tile that failed to download was left out and the elevation model built from the others, with a hole where it belongs.)

## 2. Automatic download (worldwide)

**Download DEM (worldwide)**, on the same tab, makes the elevation model from the **Copernicus DEM** - a model of the whole land surface of the earth, published as open data that needs no account. It asks nothing but where the flight was, which it reads from the AirData CSV or, without one, from the SRT files.

- Only the part under the flight is transferred - a few hundred kilobytes, in a second or two - together with **at least 150 m of ground around it** (more with a larger *Padding*), so that what the camera sees beyond the route still meets the model.
- The mesh is made in the **UTM zone the flight lies in**. Where the *Target CRS* is another one - a new project still has the default - you are asked whether to set it.
- The result is written next to the AirData CSV (into the target folder where there is none) as `copernicus_dem.glb`, `.json` and `.tif`, and set as the flight's DEM. The `.json` says where the heights came from (`source`).

**What you get, and what you do not.**

| | Copernicus DEM | A LiDAR terrain model |
|---|---|---|
| Cell size | 30 m (*GLO-30*); 90 m (*GLO-90*) for the few countries the 30 m release leaves out - used automatically, and said in the log | 1 m or better |
| What it is a model of | the **surface**: over forest somewhere in the canopy, over a village the roofs | the ground |
| How good its heights are | a few metres | decimetres |
| Heights are above | the EGM2008 geoid - sea level, as a drone's log gives its own | the national height system |

For open ground - meadows, fields, steppe, mountains above the trees - that is enough for the animals to be where they were to within the accuracy of the drone's own position. **In forest it is not a terrain model**: an animal on the forest floor is placed on a surface up to a tree's height above it, and what the camera sees at an angle is shifted sideways by that height times the tangent of the angle - 10 m for 20 m of trees at 27 degrees off straight down. Straight below the drone the place is right all the same. Fly with the camera pointing down, set the *Terrain model* accuracy under [Position uncertainty](pipeline.md#position-uncertainty) to what the cover is (4 m in the open, the height of the trees in forest), and where a terrain model of the area exists, convert that instead.

Where the flight reaches the sea, the sea is ground at height nought; where there is nothing but sea, there is nothing to build - use the flat surface.

## 3. Convert a GeoTIFF DEM

Any GeoTIFF DEM can be converted to the required GLB + metadata JSON format using the **GeoTIFF Input** panel:

- **Input file**: Path to the GeoTIFF (`.tif` / `.tiff`)
- **Output folder**: Where the `.glb` and `.json` files will be saved
- **Output CRS**: Target UTM CRS for the mesh (e.g. `EPSG:32633`)
- **Simplification**: Mesh vertex reduction factor (1 = full resolution)
- **Source CRS** *(optional)*: Override the CRS embedded in the file. Use this when the GeoTIFF has incorrect CRS metadata, for example a file containing SWEREF99TM (EPSG:3006) data but tagged as EPSG:32634. Leave empty to auto-detect.

![GeoTiff Conversion](../images/geotiff_conversion.png)

## 4. Flat surface mesh (aquatic/marine surveys)

For surveys over water, where no terrain model exists or is needed (e.g. sharks or manta rays near the water surface), the **Flat Surface** tab generates a flat GLB mesh and its companion JSON:

- **Elevation (MSL)**: Elevation of the flat projection surface in metres above mean sea level. Use `0.0` for sea-surface surveys.
- The mesh origin and extent are derived automatically from the GPS positions in the AirData CSV or SRT files selected in the input section
- The generated `flat_surface_dem.glb` / `flat_surface_dem.json` are written to the target folder and automatically set as the DEM input, so they are used like any other DEM

## 5. Manual DEM

Provide a `.gltf` or `.glb` file and its companion `.json` metadata file directly.

## Troubleshooting

- **Wrong origin location after conversion**: If the converted DEM places the origin in the wrong country, the GeoTIFF likely has incorrect CRS metadata. Enter the correct CRS in the **Source CRS** field before converting. Verify the correct CRS by loading the file in QGIS and checking its reported coordinate system.
- **Elevation meshes made with 9.0 or earlier are half a cell off**: a cell of an elevation model holds the height at its centre; the mesh put it at the cell's south-west corner, so the whole terrain lay half a cell too far to the south-west - 0.5 m with the Austrian one-metre model, 5 m with a ten-metre GeoTIFF. On flat ground that changes nothing; on a slope every position gets a wrong height (2 m on a 20 degree slope with ten-metre cells), and positions of animals near the edge of a frame move with it. Meshes made now are correct and say so in their `.json` (`"heights_at": "cell centres"`). For an older mesh the geo-referencing steps write a note with the size of the shift into the log; create the mesh again (**Download DEM** or **Convert a GeoTIFF**) and run *Geo-Reference Detections* again to correct it - with the one-metre model the difference is below what the drone's own position is good for, with coarser models it is worth it.
- **Geo-referencing issues**: Ensure the DEM covers your entire flight area, the metadata JSON has correct origin coordinates, and the target CRS matches your DEM projection.
