# -*- coding: utf-8 -*-
"""The Dataset Export step of the BAMBI processor (see
:mod:`core.dataset_export`).

Like the survey analytics it works on a *survey* - the flights ticked on
the Survey Analytics tab and one camera - and writes them as one dataset,
each flight in the split it was given. Mixed into :class:`BambiProcessor`.
"""

from typing import Any, Dict


class SurveyExportMixin:
    """The flights of a survey as a dataset with train, validation and
    test splits."""

    def run_dataset_export(self, config: Dict[str, Any], progress_fn=None,
                           log_fn=None, cancel_check=None):
        """Export the survey's flights in one of the export formats, each
        flight whole in one split.

        Settings (``config["dataset_export"]``): ``format`` - a key of
        ``core.exporters.EXPORTERS``; ``output`` - the dataset's folder;
        ``splits`` - ``{flight name: "train" | "val" | "test" | ""}``
        (``""`` leaves a flight out); ``include_images``;
        ``include_not_an_animal`` (``None``: as the format does by
        default).

        What is exported is what the flights hold - every labelled box of
        the survey's camera. The species filter, the survey area and "only
        confirmed by the other camera" are questions of an analysis and do
        not apply.
        """
        from .bambi_processing import CancelledException
        from .core import dataset_export, exporters

        survey = self._survey(config)
        settings = dict(config.get("dataset_export") or {})
        key = settings.get("format") or "coco"
        splits = settings.get("splits") or {}
        epsg = survey.epsg
        if key in exporters.NEEDS_CRS and not epsg:
            raise RuntimeError(
                f"{exporters.EXPORTERS[key][0]} publishes latitude and "
                "longitude, so the flights' target CRS must be set before "
                "exporting.")
        flights = [{"name": flight.name,
                    "target_folder": flight.target_folder,
                    "epsg": flight.epsg or epsg,
                    "split": splits.get(flight.name, dataset_export.LEFT_OUT)}
                   for flight in survey.flights]
        if log_fn:
            log_fn(f"Exporting {self._flights_note(survey)} "
                   f"({survey.camera_label}) as {exporters.EXPORTERS[key][0]}"
                   f": {dataset_export.describe({f['name']: f['split'] for f in flights})}")

        stopped = []

        def cancelled():
            if cancel_check and cancel_check():
                stopped.append(True)
            return bool(stopped)

        try:
            return dataset_export.export(
                key, flights, survey.modality,
                (settings.get("output") or "").strip(),
                include_images=settings.get("include_images"),
                include_not_an_animal=settings.get("include_not_an_animal"),
                progress=progress_fn, cancelled=cancelled, log=log_fn)
        except dataset_export.DatasetExportError as exc:
            if stopped:
                raise CancelledException("Dataset export cancelled") from exc
            raise RuntimeError(str(exc)) from exc
