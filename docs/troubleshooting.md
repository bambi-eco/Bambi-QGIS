# Troubleshooting

## Where to look first

When a result is missing or not what you expected, open QGIS' **Log Messages** panel (*View > Panels > Log Messages*) and its **BAMBI** tab. The plugin carries on after many small failures - metadata it could not read, a raster whose CRS could not be set, a correction file that did not parse - and says there what happened and why, with the original error. What matters for the result is written as a warning and repeated in the plugin's own log panel, starting with `Note:`; fallbacks and clean-up that did not work are written as information. (Up to 9.0 these 62 places said nothing at all: an unreadable `correction.json`, for instance, simply meant an uncorrected flight.)

## `No module named 'bambi'` / `No module named 'alfspy'`

You have not installed the `bambi_detection` / `alfs_py` dependency, or you have not restarted QGIS after installing it. Install both required packages via the [Dependency Manager](installation.md#dependency-manager) and restart QGIS; the Python environment is only loaded at startup.

## Problems installing dependencies

See [Installation problems](installation.md#installation-problems) for Git/PATH issues in the OSGeo4W Shell and offline installation from local paths.

## Model download fails

- Check your internet connection
- Download manually from HuggingFace: [Thermal detection (thermal_animal_detector.pt)](https://huggingface.co/cpraschl/bambi-thermal-detection), [RGB detection (rgb/weights/best.pt)](https://huggingface.co/cpraschl/bambi-models), [Re-ID (osnet_x0_5_bambi_thermal_omni.pt)](https://huggingface.co/cpraschl/bambi-thermal-omni)
- Place `thermal_animal_detector.pt` (thermal) or the RGB weights renamed to `rgb_animal_detector.pt` in `%APPDATA%\QGIS\QGIS3\profiles\default\bambi_deps\models\` (or specify a custom path in the **Thermal/RGB Model Path** fields of the Detection configuration)
- The error message names the exact file to fetch: the default models are the ones this plugin release was tested with, identified by the commit of their repository and their SHA-256. A file placed by hand that is another version is set aside as `*.previous` and the tested one downloaded; without a network the file that is there is used, and the log says that it is not the tested version. A model of your own goes into the **Model Path** fields, where nothing is checked.

## "... is not the published file"

A download arrived, but it is not the model that was published - it was cut off, or a proxy answered with a page of its own. Nothing of it is kept. Try again; if it persists, download the file in a browser from the address in the message and place it as described above.

## Geo-referencing issues

- Ensure the DEM covers your entire flight area
- Verify the DEM metadata JSON has correct origin coordinates
- Check that the target CRS matches your DEM projection
- If poses look shifted or rotated, determine correction factors with the [Correction Wizard](correction-wizard.md)

## GeoTIFF conversion: wrong origin location

- If the converted DEM places the origin in the wrong country, the GeoTIFF likely has incorrect CRS metadata
- Enter the correct CRS in the **Source CRS** field (e.g. `EPSG:3006` for SWEREF99TM) before converting
- Verify the correct CRS by loading the file in QGIS and checking its reported coordinate system

## Frames do not line up with the flight log (time zone)

The cameras write local time, the flight log UTC; **Timezone → Auto** finds the offset between them by looking for the one - any quarter of an hour from UTC-12 to UTC+14 - under which the recording falls on the log rows flagged as video (or photo). It reads the SRT files of either camera, and the photos' EXIF times where there are none, and shows what it found as *Detected offset*; the label follows the files you choose.

Up to 9.0 the offset was the difference between the mean *times of day* of the two, rounded to whole hours. That failed at night: a video across midnight came out as UTC-10, one after midnight as UTC-22 (where UTC+2 was right), and the frames were then matched to the wrong part of the log, or to none of it. Offsets that are no whole hours (India, Nepal, parts of Australia) were half an hour off. If the detected offset looks wrong - a camera clock that was never set, say - untick **Auto** and choose the time zone by name.

## Photo mode: images not matched to GPS

- Ensure image timestamps (EXIF) are in UTC or that the correct timezone offset is configured in the **Configuration** tab
- Verify the AirData CSV covers the same time range as the photo session

## No tracks generated

- Lower the detection confidence threshold
- Adjust the IoU threshold for tracking
- Verify that detections were successfully generated in step A1

## Perpendicular distances not calculated

- Run **2. Generate Flight Route** before **→ Calculate Perpendicular**, since the AirData GPS `LineString` is required
- Ensure the AirData CSV contains enough GPS records (at least 2 valid positions)
- For track perpendicular: ensure **4. Track Animals Or Import** has been completed first

## BoxMOT import errors

- Ensure you're installing in the correct Python environment (OSGeo4W Shell)
- Try: `pip install --upgrade boxmot`
- Check for conflicting package versions

## Memory issues with large datasets

- Reduce the number of frames processed at each step; back up intermediate results and rerun with different settings
- For ALFS and GeoTIFFs: reduce resolution or decrease the max tile size parameter
- **Graphics memory filling up during ALFS, GeoTIFF export or in the 3D Flight Viewer** (up to 9.0): every tile, sample or frame left its framebuffer on the graphics card - 24 bytes per pixel, about 400 MB for a 4096 px tile - until the step ended, and a step that failed or the viewer's ALFS tab never gave it back at all, so memory grew until QGIS was restarted. Fixed: each render gives back what it took as soon as it is done. If you worked around it with a smaller tile size or by restarting QGIS between runs, that is no longer needed
- **Memory growing while tools are opened and closed** (up to 9.0): a tool - the 3D Flight Viewer, the Thermal Viewer, the AGL tool, the Flight Planner, the Video Creator, the wizards - tidied up only when it was closed with the window's close button. Closed with `Esc` or with its own *Close* button it was merely hidden: its background work went on, its files stayed open, and what it had loaded stayed in memory; and every tool that had been closed, in whatever way, stayed in QGIS as a hidden window until QGIS was quit, so opening it again added another. Fixed: a tool ends its work and gives back what it holds however it is closed, and is removed once it is closed
