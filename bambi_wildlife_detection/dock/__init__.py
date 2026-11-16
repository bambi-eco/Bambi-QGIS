# -*- coding: utf-8 -*-
"""The main panel, by tab.

``bambi_dock_widget.BambiDockWidget`` is one class when the plugin runs -
every control is an attribute of the dock widget itself - and its code is
kept here, one module per tab or concern. Each module holds a class that is
mixed into ``BambiDockWidget``; none of them is a widget of its own. The one
exception is the step runner, which is an object the dock has.

The tabs
    ``input_tab``                the Input tab
    ``configuration_tab``        the Configuration tab and most of its sub-tabs
    ``correction_settings``      Configuration > Correction
    ``tracking_settings``        Configuration > Tracking
    ``classification_settings``  Configuration > Classification
    ``step_tabs``                the Pre-Processing and Processing tabs
    ``accuracy``                 Processing > Accuracy: the report against
                                 hand labels
    ``analytics_tab``            the Survey Analytics tab
    ``grid_density``             Survey Analytics > Grid Density, and the
                                 grid on the map
    ``trend``                    Survey Analytics > Compare Visits: the same
                                 ground, surveyed again
    ``reaction``                 Survey Analytics > Reaction to the Drone,
                                 and the movements on the map
    ``position``                 Survey Analytics > Position Uncertainty, and
                                 the circles on the map
    ``review_queue``             Survey Analytics > Review Queue: the tracks
                                 in doubt, one after another

What they share
    ``run_panel``                starting a step, its progress, the log
    ``step_runner``              the step runner: one step at a time, on a
                                 thread of its own (an object, ``dock.steps``)
    ``flights``                  the flights of a project, switching between them
    ``settings``                 reading the form (``get_config``), keeping it
    ``step_status``              what has been done, Reset Step
    ``inspectors``               the map tools of the toolbar

On the map
    ``result_layers``            the "Add ... to QGIS" buttons of the steps
    ``flight_route_layers``      the flight route and its markers
    ``transect_layers``          the transects of a survey
    ``group_layers``             the groups the census found
    ``layer_groups``             where a layer goes in the layer tree

``range_dialogs`` and ``common`` hold the dialogs and helpers that more than
one of them uses. The Environment sub-tab and the groups that go with it are
in ``bambi_environment_panel``, which is mixed in the same way.

A method of one part calls those of another through ``self``, as it did when
they were in one file. A name that a test replaces has to be replaced in
the module that uses it.
"""
