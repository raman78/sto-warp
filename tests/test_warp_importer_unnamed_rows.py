"""Equipment rows the detector could not name are reported, not dropped.

Such a row gets no boxes. The importer turns `LayoutDetector.last_unnamed_rows`
into a line in `ImportResult.errors`, which WARP CORE shows under the review
list — unless boxes already cover the row (the user drew them, and they come
back merged as confirmed), in which case nothing is missing any more.

Offline: a stand-in detector, no image.
"""
from __future__ import annotations

from types import SimpleNamespace

import warp.warp_importer as wi

ROW = {'row': 8, 'cy': 415, 'filled': 2, 'y0': 394, 'y1': 436, 'x0': -10, 'x1': 196}


def _importer(rows):
    imp = wi.WarpImporter.__new__(wi.WarpImporter)
    imp._layout = SimpleNamespace(last_unnamed_rows=rows)
    return imp


def test_an_unnamed_row_with_no_box_is_reported():
    assert _importer([ROW])._unnamed_rows_without_boxes({}) == [ROW]


def test_a_row_the_user_has_boxed_is_no_longer_reported():
    layout = {'Universal Consoles': [(163, 395, 32, 42)]}
    assert _importer([ROW])._unnamed_rows_without_boxes(layout) == []


def test_a_box_in_another_panel_on_the_same_line_does_not_cover_it():
    layout = {'Personal Space Traits': [(400, 395, 33, 45)]}
    assert _importer([ROW])._unnamed_rows_without_boxes(layout) == [ROW]


def test_a_detector_that_never_measured_reports_nothing():
    assert _importer(None)._unnamed_rows_without_boxes({}) == []


def test_row_numbers_are_written_as_ranges():
    assert wi._row_ranges([8]) == 'row 8'
    assert wi._row_ranges([1, 2, 4, 5, 6, 12]) == 'rows 1-2, 4-6, 12'
