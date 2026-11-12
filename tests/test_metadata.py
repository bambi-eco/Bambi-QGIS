# -*- coding: utf-8 -*-
"""metadata.txt must load the way plugins.qgis.org (and QGIS) read it.

Both use ``configparser.ConfigParser`` with its default interpolation, where
a bare ``%`` in any value (say, "33 %" in the changelog) is a syntax error
that rejects the upload.
"""
import configparser
import os

METADATA = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "bambi_wildlife_detection", "metadata.txt")


def test_metadata_is_readable_with_interpolation():
    parser = configparser.ConfigParser()
    parser.optionxform = str
    with open(METADATA, encoding="utf-8") as fh:
        parser.read_file(fh)
    # reading a value is what triggers interpolation
    values = dict(parser.items("general"))
    for key in ("name", "version", "qgisMinimumVersion", "description",
                "about", "changelog"):
        assert values.get(key), key
