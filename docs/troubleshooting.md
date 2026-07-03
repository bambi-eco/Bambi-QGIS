# Troubleshooting

## `No module named 'bambi'` / `No module named 'alfspy'`

You have not installed the `bambi_detection` / `alfs_py` dependency, or you have not restarted QGIS after installing it. Install both required packages via the [Dependency Manager](installation.md#dependency-manager) and restart QGIS — the Python environment is only loaded at startup.

## Problems installing dependencies

See [Installation problems](installation.md#installation-problems) for Git/PATH issues in the OSGeo4W Shell and offline installation from local paths.

## Model download fails

- Check your internet connection
- Download manually from HuggingFace: [Detection (thermal_animal_detector.pt)](https://huggingface.co/cpraschl/bambi-thermal-detection), [Re-ID (osnet_x0_5_bambi_thermal_omni.pt)](https://huggingface.co/cpraschl/bambi-thermal-omni)
- Place `thermal_animal_detector.pt` in `%APPDATA%\QGIS\QGIS3\profiles\default\bambi_deps\models\` (or specify a custom path in the **Model Path** field of the Detection configuration)

## Geo-referencing issues

- Ensure the DEM covers your entire flight area
- Verify the DEM metadata JSON has correct origin coordinates
- Check that the target CRS matches your DEM projection
- If poses look shifted or rotated, determine correction factors with the [Correction Wizard](correction-wizard.md)

## GeoTIFF conversion: wrong origin location

- If the converted DEM places the origin in the wrong country, the GeoTIFF likely has incorrect CRS metadata
- Enter the correct CRS in the **Source CRS** field (e.g. `EPSG:3006` for SWEREF99TM) before converting
- Verify the correct CRS by loading the file in QGIS and checking its reported coordinate system

## Photo mode: images not matched to GPS

- Ensure image timestamps (EXIF) are in UTC or that the correct timezone offset is configured in the **Configuration** tab
- Verify the AirData CSV covers the same time range as the photo session

## No tracks generated

- Lower the detection confidence threshold
- Adjust the IoU threshold for tracking
- Verify that detections were successfully generated in step 3

## Perpendicular distances not calculated

- Run **2. Generate Flight Route** before **→ Calculate Perpendicular** — the AirData GPS `LineString` is required
- Ensure the AirData CSV contains enough GPS records (at least 2 valid positions)
- For track perpendicular: ensure **4. Track Animals Or Import** has been completed first

## BoxMOT import errors

- Ensure you're installing in the correct Python environment (OSGeo4W Shell)
- Try: `pip install --upgrade boxmot`
- Check for conflicting package versions

## Memory issues with large datasets

- Reduce the number of frames processed at each step; back up intermediate results and rerun with different settings
- For ALFS and GeoTIFFs: reduce resolution or decrease the max tile size parameter
