# Environmental Context

Publicly available data for the ground a flight monitored, on or near the day it flew: greenness, vegetation moisture, open water, snow, land cover, tree cover, terrain wetness and the weather. It is there to help explain *where* animals were (clustering, habitat use) and *how well* a camera could see them (detectability) - and to give every flight a short record of the conditions it was flown in.

It comes in two parts:

- **Environmental Context** on the *Pre-Processing* tab - four per-flight sub-steps, one per data source, that fetch or derive the layers.
- **Habitat Context** on the *Survey Analytics* tab - joins the layers with the tracked animals of a survey: a table with one row per animal, charts of the animals against the monitored ground, and a table per transect.

None of the sources needs an account. E1, E2 and E4 need an internet connection; E3 works offline.

## The four sources

| Step | Source | What you get | Cell size | For the flight's date? |
|------|--------|--------------|-----------|------------------------|
| **E1** | Sentinel-2 L2A via [Earth Search](https://earth-search.aws.element84.com/v1) | NDVI (greenness), NDMI (vegetation moisture), NDWI (open water), NDSI (snow) | 10 m (NDMI and NDSI from 20 m bands) | Nearest clear scene, before or after |
| **E2** | [ESA WorldCover](https://esa-worldcover.org) 2021 v200 | Land-cover class, tree cover (%) | 10 m | No - one map for 2021 |
| **E3** | The flight's own DEM | Topographic wetness index (TWI) | 10 m | No - terrain only |
| **E4** | [Open-Meteo](https://open-meteo.com) | Hourly weather at flight time and over the 24 h before | one value for the flight | Yes |

**Each source is its own sub-step.** They do not depend on each other and can be run, re-run and fail separately: a weather service that is down does not cost you the satellite layers. Within one sub-step there is no partial result - a source that cannot be reached fails its step with the reason, and what an earlier run of *that* source left behind is removed first, so old rasters never sit beside a failed re-run looking current. **→ Run All Four** runs E1 to E4 one after the other and carries on past a failure; *Abort* stops the whole queue.

The environment belongs to the **flight**, not to a camera, so there is no camera selector: thermal and RGB look at the same meadow.

### Extent and grid

The layers cover the ground the flight saw - its field-of-view footprints (P3), or the flight path when P3 has not run yet - plus a buffer (200 m by default) so the neighbouring habitat is visible. When the survey area is switched on (Survey Analytics tab) and drawn, the extent is widened to include it, so the layers show the whole plot and not only the strip flown.

All rasters share one 10 m grid in the flight's target CRS, snapped to multiples of 10 m. A flight in the same UTM zone as its Sentinel-2 tile therefore gets the satellite's own pixels, not resampled ones.

### E1. Satellite indices (Sentinel-2)

Sentinel-2 passes every two to five days, and cloud decides what it sees. The step searches the scenes within the **scene window** (±10 days by default) of the flight and tries them **nearest first, before or after the flight alike**. A scene is used when the share of the *flight's extent* that is free of cloud, cloud shadow, cirrus and missing data reaches the **required clear share** (80 % by default) - judged from the scene's own classification over the flight, not from the cloud percentage of the 100 km tile. Cloudy cells of the chosen scene are written as no-data.

The scene, its date, its distance from the flight in days (negative = before) and its clear share are recorded in `sentinel2.json`, shown in the step's status and carried into the habitat tables - a scene taken eight days before a flight is not the same evidence as one from the day before.

If no scene in the window is clear enough the step fails and names the best candidate; widen the window or lower the required share on **Configuration → Environment**.

| Index | Formula | Reads as |
|-------|---------|----------|
| NDVI | (NIR − red) / (NIR + red) | Dense green vegetation high, bare ground and water low. Forage, cover, leaf-on state of deciduous canopy. |
| NDMI | (narrow NIR − SWIR) / (narrow NIR + SWIR) | Water content of the vegetation. A **proxy for how wet a patch is** - not air humidity, and under closed forest it says little about the ground. Most meaningful as a comparison within one scene. |
| NDWI | (green − NIR) / (green + NIR) | Positive over open water: ponds, flooded ground, streams wider than a cell. |
| NDSI | (green − SWIR) / (green + SWIR) | Above about 0.4 over snow. |

A sudden drop in NDVI between two flights' scenes over a meadow is usually mowing; over a field, harvest. The plugin fetches one scene per flight and does not compute change layers itself.

### E2. Land cover (ESA WorldCover)

The class of every 10 m cell - tree cover, shrubland, grassland, cropland, built-up, bare, snow and ice, water, wetland, mangroves, moss and lichen - with the map's own colours. **Tree cover** is derived from it: the share of tree cells within 50 m of each cell, which is what separates closed forest from a hedge and what the canopy does to detectability.

WorldCover maps 2021. Meadows and forest do not move between flights, but a clear-cut or a new field after 2021 is not in it.

### E3. Terrain wetness (DEM)

The topographic wetness index, `ln(a / tan β)`, with *a* the upslope area draining through a cell and *β* its slope: high in flat hollows and valley floors where water collects, low on ridges and steep slopes. It is static, needs no download and works under closed forest, which makes it the complement to NDMI.

It is computed from the DEM mesh selected on the Input tab, sampled onto the grid with a 500 m margin around the flight (a cell's wetness depends on the slope above it). Depressions are filled and the flow is routed to all lower neighbours. The step fails when the DEM covers less than half of the extent, and on a flat-surface DEM (aquatic surveys), where there is no terrain to derive anything from.

### E4. Weather (Open-Meteo)

Not satellite data, but for a thermal flight the strongest explanation of detectability: what the air and the ground were at when the drone flew, and how much sun, rain and wind the hours before put into them.

- **At flight time** (interpolated to the middle of the flight): air temperature, relative humidity, dew point and its spread to the air temperature (small = dew, mist, fog), wind speed, gusts and direction, cloud cover, shortwave radiation, surface pressure, snow depth, soil temperature and moisture of the top layer.
- **Previous 24 hours** (before the first frame): precipitation and snowfall sums, sunshine hours, air and soil temperature range, mean cloud cover, radiation, wind and humidity, strongest gust, hours since the last rain.

Flights older than a week are answered by Open-Meteo's archive (the ERA5 reanalysis family, back to 1940), newer ones by its forecast service, because the reanalysis lags a few days behind. These are **model values for a grid cell of one to a few tens of kilometres** - one value for the whole flight, not per patch - and all times are UTC. `weather_hourly.csv` keeps the hourly rows the summary was derived from (72 h before the flight to its end).

Open-Meteo's free service is rate-limited per network address and for non-commercial use; when the limit is reached the step fails with the service's message and can be re-run later.

## Layers in QGIS

**→ Add Environment Layers to QGIS** adds every raster of the active flight as a styled layer in a **BAMBI Environment** group. The group sits *below* the flight's Thermal and RGB groups - these are base layers - and shows one raster at a time, since stacked opaque rasters would only hide each other.

## Habitat context (Survey Analytics)

**→ Summarise Habitat** works on the survey - the ticked flights, the camera, the species filter and the survey area, like every other analytic. Each flight needs tracks and at least one environment source; sources a flight lacks are named in the log and their columns stay empty.

**Per animal** (`habitat_tracks.csv` / `.json`): one row per counted track - the same animals as the census - with its position, species, sex and age, the mean of every layer within the **radius** (30 m by default) of its position, the most common land-cover class there with its share, the scene's distance from the flight in days, and the flight's weather. Satellite cells are 10-20 m, so a value describes **the patch an animal was in, not the spot it stood on**; a radius of 0 takes the single cell under it. **→ Add Habitat Tracks to QGIS** loads the table as a point layer in the BAMBI Survey group, to filter, style and export.

**Animals against the ground** (`habitat.json`, **→ Show Habitat Charts**): a value on its own says little - greener, wetter or more wooded *than the area at large* is the signal. For every layer the charts show its distribution over the **monitored ground** (grey) and at the animals (colour), with both means; the monitored ground is what the cameras actually saw (the field-of-view footprints, inside the survey area when one is set), because an animal could only be found where somebody looked. For land cover each class shows its share of the ground, its share of the animals and their ratio (the *selection ratio*: above 1 a class is used more than its area would suggest). A species selector redraws the animals' side. A table lists each flight's scene and weather.

With few animals the distributions are noise; the charts state *n*. The ratio is descriptive, not a test, and tree cover cuts both ways: animals under canopy are also harder to detect.

**Per transect** (`habitat_transects.csv`): when the [population estimate](pipeline.md#population-estimation-transects) has written a flight's transect areas, the mean of every layer over each transect's monitored area and its dominant land cover, next to the transect's count and area - covariates for a model fitted outside the plugin. The plugin's own population estimate is unchanged.

## Configuration

**Configuration → Environment** (saved with the project):

| Setting | Default | Meaning |
|---------|---------|---------|
| Scene window (±) | 10 days | How far from the flight a Sentinel-2 scene may be |
| Required clear share | 80 % | Usable share of the flight's extent a scene needs |
| Buffer around the flight | 200 m | Ground fetched around the footprint |
| Extend to the survey area | on | Widen the extent to the survey area when one is defined |

The radius around each animal is set in the **Habitat Context** group.

## Output

```
target_folder/environment/                   # per flight, camera-independent
├── ndvi.tif  ndmi.tif  ndwi.tif  ndsi.tif   # E1, float32, no-data = NaN
├── sentinel2.json                           # scene, date, offset, clear share, candidates tried
├── landcover.tif                            # E2, class values with colour table
├── tree_cover.tif                           # E2, percent
├── landcover.json                           # tiles, composition, class names
├── twi.tif                                  # E3
├── terrain.json
├── weather.json                             # E4: at_flight, previous_24h, hourly
└── weather_hourly.csv

survey_folder/analytics_{t|w}/
├── habitat_tracks.csv / .json               # one row per track
├── habitat_transects.csv                    # one row per transect (if any)
└── habitat.json                             # distributions behind the charts
```

A source counts as done when its `.json` and the rasters it names exist; deleting them resets the step.

## Limits

- **Scale.** 10-20 m cells against centimetre drone imagery: patches, not positions.
- **Date.** "On the flight's date" is in practice "the nearest clear scene"; the offset is always recorded.
- **Moisture.** NDMI and TWI are proxies for wetness of vegetation and ground. There is no per-patch air humidity; humidity and dew point come from the weather model, one value per flight.
- **Water level.** Satellites give the *extent* of small water bodies (NDWI), not their level.
- **Coverage.** All three online sources are global. Other services - including ones that need credentials - can be added in `core/env_sentinel2.py` (`Catalogue`) without touching the rest.

## Attribution

Contains modified Copernicus Sentinel data (Sentinel-2, served by Element 84 Earth Search). ESA WorldCover 10 m 2021 v200 © ESA WorldCover project, CC BY 4.0. Weather data by Open-Meteo.com, CC BY 4.0. Cite them when you publish results derived from these layers.
