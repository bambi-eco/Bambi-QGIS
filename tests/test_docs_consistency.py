# -*- coding: utf-8 -*-
"""The documentation names things as the plugin does.

Three kinds of drift, each found once and corrected by hand:

* a step heading in the docs that is not what the step's button says
  ("E1. Satellite indices" for the button "E1. Fetch Satellite Indices"),
  and a toolbar tool under a name it does not have ("Select Detection or
  Track" for "Inspector: Click Detection / Track");
* a link to a heading that was renamed;
* comments pointing to design documents that are not in the repository -
  55 of them, with a couple of hundred bare "(section 8.1)" marks that meant
  the same documents.
"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "bambi_wildlife_detection")
DOCS = sorted(glob.glob(os.path.join(ROOT, "docs", "*.md"))) \
    + [os.path.join(ROOT, "README.md")]


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _sources():
    paths = glob.glob(os.path.join(PACKAGE, "**", "*.py"), recursive=True)
    return [path for path in paths if os.sep + "ext" + os.sep not in path]


def _plain(text):
    return re.sub(r"\s+", " ", text.replace("…", "")).strip()


def _step_buttons():
    """``{"P1": "Extract Frames", ...}`` from the buttons' own texts."""
    found = {}
    for path in _sources():
        for code, name in re.findall(
                r'"\s*([PACSE]\d)\.\s+([^"\n]+?)…?"', _read(path)):
            found.setdefault(code, _plain(name))
    return found


def test_the_steps_are_found_in_the_code():
    steps = _step_buttons()
    assert steps["P1"] == "Extract Frames"
    assert steps["A2"] == "Track Animals Or Import"
    assert steps["E3"] == "Compute Terrain Wetness (DEM)"
    assert len(steps) >= 19


def test_a_step_heading_says_what_the_button_says():
    steps = _step_buttons()
    wrong = []
    for path in DOCS:
        for code, name in re.findall(
                r"^#{2,5}\s+([PACSE]\d)\.\s+(.+?)\s*$", _read(path),
                flags=re.M):
            if _plain(name) != steps.get(code):
                wrong.append(f"{os.path.basename(path)}: {code}. {name} "
                             f"(button: {steps.get(code)})")
    assert wrong == []


def test_no_step_is_referred_to_that_does_not_exist():
    steps = _step_buttons()
    unknown = []
    for path in DOCS:
        for code in set(re.findall(r"\b([PACSE]\d)\b", _read(path))):
            if code not in steps:
                unknown.append(f"{os.path.basename(path)}: {code}")
    assert unknown == []


def test_the_toolbar_tools_are_called_by_their_names():
    plugin = _read(os.path.join(PACKAGE, "bambi_wildlife_detection.py"))
    tools = re.findall(r"text=self\.tr\('(Inspector: [^']+)'\)", plugin)
    assert len(tools) == 3
    documented = _read(os.path.join(ROOT, "docs", "tools.md"))
    for tool in tools:
        assert f"**{tool}**" in documented, tool


def _anchor(heading):
    anchor = re.sub(r"[^\w\s-]", "", heading.strip().lower(), flags=re.U)
    return re.sub(r"\s", "-", anchor)


def test_links_between_the_documents_lead_somewhere():
    anchors = {os.path.basename(path): {
        _anchor(heading) for heading in re.findall(
            r"^#{1,6}\s+(.+)$", _read(path), flags=re.M)} for path in DOCS}
    broken = []
    for path in DOCS:
        here = os.path.basename(path)
        for file_name, anchor in re.findall(
                r"\]\((?:docs/)?([a-zA-Z-]+\.md)?#([^)]+)\)", _read(path)):
            target = file_name or here
            if target in anchors and anchor not in anchors[target]:
                broken.append(f"{here} -> {target}#{anchor}")
    assert broken == []


def test_documents_that_are_pointed_to_exist():
    """A comment that says "see SOME_PLAN.md" is no use without the file."""
    missing = set()
    paths = _sources() + glob.glob(os.path.join(ROOT, "tests*", "*.py")) \
        + glob.glob(os.path.join(ROOT, "*.sh"))
    for path in paths:
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue
        for name in re.findall(r"\b([A-Z][A-Z_]+_PLAN\.md)\b", _read(path)):
            if not os.path.isfile(os.path.join(ROOT, name)):
                missing.add(f"{os.path.relpath(path, ROOT)}: {name}")
    assert sorted(missing) == []


def test_a_section_number_says_of_what():
    """"(section 8.1)" on its own meant a document that is not there. What
    is left names its source: the paper, or the plan that is in the
    repository."""
    bare = []
    paths = _sources() + glob.glob(os.path.join(ROOT, "tests*", "*.py"))
    for path in paths:
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue
        lines = _read(path).split("\n")
        for number, line in enumerate(lines, 1):
            if "§" not in line:
                continue
            nearby = " ".join(lines[max(0, number - 3):number + 1])
            if not re.search(r"paper|When One Modality|CLASSIFICATION_PLAN",
                             nearby):
                bare.append(f"{os.path.relpath(path, ROOT)}:{number}")
    assert bare == []
