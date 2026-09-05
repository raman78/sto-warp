"""One EasyOCR reader per process, not one per component.

Three were being built during a single recognition run: `TextExtractor`'s,
`LayoutDetector`'s, and a third behind a module-global `TextExtractor` inside
`eq_geometry` that existed only to reach its reader. Each loads the text
detection network and the recognition network, so two of the three paid that
cost for nothing. Measured 2026-09-05 by counting distinct reader objects seen
calling `readtext` during a folder pass.

Sharing is safe because `readtext` does not mutate the reader — it is a forward
pass over the caller's pixels — and the pipeline is sequential, one recognition
worker walking a folder image by image.

Offline: the reader factory is replaced, so no model is ever loaded.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from warp.recognition import eq_geometry, text_extractor
from warp.recognition.layout_detector import LayoutDetector
from warp.recognition.text_extractor import TextExtractor


class _FakeReader:
    def __init__(self):
        self.calls = 0

    def readtext(self, *a, **kw):
        self.calls += 1
        return []


@pytest.fixture
def built(monkeypatch):
    """Count how many readers get constructed, whoever asks."""
    made: list[_FakeReader] = []

    class _Easy:
        @staticmethod
        def Reader(*a, **kw):
            made.append(_FakeReader())
            return made[-1]

    monkeypatch.setattr(text_extractor, '_SHARED_READER', None)
    monkeypatch.setitem(__import__('sys').modules, 'easyocr', _Easy)
    return made


def test_two_extractors_share_one_reader(built):
    a, b = TextExtractor(), TextExtractor()
    assert a._get_ocr() is b._get_ocr()
    assert len(built) == 1


def test_the_layout_detector_shares_it_too(built):
    te = TextExtractor()
    ld = LayoutDetector.__new__(LayoutDetector)   # no __init__: no Qt, no disk
    ld._ocr = None
    assert ld._get_ocr() is te._get_ocr()
    assert len(built) == 1


def test_eq_geometry_shares_it_too(built):
    te = TextExtractor()
    assert eq_geometry._get_easyocr_reader() is te._get_ocr()
    assert len(built) == 1


def test_all_three_together_build_exactly_one(built):
    te = TextExtractor()
    ld = LayoutDetector.__new__(LayoutDetector)
    ld._ocr = None
    readers = {id(te._get_ocr()), id(ld._get_ocr()),
               id(eq_geometry._get_easyocr_reader())}
    assert len(readers) == 1
    assert len(built) == 1


def test_the_reader_is_built_only_on_first_use(built):
    """Constructing a TextExtractor must not load the models — importing the
    trainer would otherwise pay for OCR it may never run."""
    TextExtractor()
    assert built == []


def test_eq_geometry_no_longer_keeps_a_text_extractor():
    """It built a whole TextExtractor purely to reach its reader."""
    assert not hasattr(eq_geometry, '_TEXT_EXTRACTOR')
