"""One read of the screenshot, shared by everything that needs text.

The pipeline used to read the same picture two or three times, in three
different ways: `scan_image` in five horizontal strips, `eq_geometry` in one
whole frame, and `_ocr_section_labels` in one whole frame with the channels in
BGR order handed to a reader that expects RGB.

Every reason recorded for the split turned out to be false — resolution (the
detector never upscales below 2560 px, and a strip has the frame's width),
speed (the loop had no early exit), and accuracy (the apparent edge came from
ground labels being matched by exact string equality, so one mistyped
character dropped a row).

Measured over the store after unifying them: the space grid scores 2736 of
2931 confirmed boxes both before and after, with every slot identical to a
tenth of a percent and no screenshot changing its box count; the ground grid
stays at 298 of 299. OCR per screenshot falls from 7-8 calls to 1-2.

Offline: the reader is replaced, so nothing runs a model.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from warp.recognition import eq_geometry as eg
from warp.recognition.layout_detector import LayoutDetector
from warp.recognition.text_extractor import TextExtractor


class _Reader:
    """Records every image it is asked to read."""

    def __init__(self):
        self.shapes: list[tuple] = []

    def readtext(self, image, *a, **kw):
        self.shapes.append(image.shape[:2])
        return []


@pytest.fixture
def reader(monkeypatch):
    r = _Reader()
    monkeypatch.setattr('warp.recognition.text_extractor.shared_reader',
                        lambda: r)
    return r


# ── scan_image ────────────────────────────────────────────────────────────

def test_scan_image_reads_the_frame_once(reader):
    te = TextExtractor()
    img = np.zeros((900, 1600, 3), dtype=np.uint8)
    te.scan_image(img)
    assert reader.shapes == [(900, 1600)]


def test_it_does_not_slice_the_frame(reader):
    """Five strips of a 900 px frame were 195 px tall."""
    te = TextExtractor()
    img = np.zeros((900, 1600, 3), dtype=np.uint8)
    te.scan_image(img)
    assert all(h == 900 for h, _w in reader.shapes)


def test_a_second_call_is_served_from_the_cache(reader):
    te = TextExtractor()
    img = np.zeros((900, 1600, 3), dtype=np.uint8)
    te.scan_image(img)
    te.scan_image(img)
    assert len(reader.shapes) == 1


def test_clearing_the_cache_makes_it_read_again(reader):
    te = TextExtractor()
    img = np.zeros((900, 1600, 3), dtype=np.uint8)
    te.scan_image(img)
    te.clear_scan_cache()
    te.scan_image(img)
    assert len(reader.shapes) == 2


def test_a_read_that_raises_yields_no_tokens_and_is_reported(reader, monkeypatch,
                                                              capfd):
    def _boom(*a, **kw):
        raise RuntimeError('reader exploded')
    monkeypatch.setattr(reader, 'readtext', _boom)
    te = TextExtractor()
    assert te.scan_image(np.zeros((50, 50, 3), dtype=np.uint8)) == []
    assert 'scan_image read failed' in capfd.readouterr().err


# ── The space panel takes the shared tokens ───────────────────────────────

def test_space_geometry_uses_tokens_when_given(monkeypatch):
    """It must not open the screenshot a second time."""
    called = []
    monkeypatch.setattr(eg, '_run_ocr', lambda img: called.append(1) or [])
    eg.detect_eq_geometry(np.zeros((100, 100, 3), dtype=np.uint8),
                          ocr_tokens=[])
    assert called == []


def test_space_geometry_still_reads_for_itself_without_tokens(monkeypatch):
    """Kept for callers outside detect() — the dev probes rely on it."""
    called = []
    monkeypatch.setattr(eg, '_run_ocr', lambda img: called.append(1) or [])
    eg.detect_eq_geometry(np.zeros((100, 100, 3), dtype=np.uint8))
    assert called == [1]


def test_the_detector_hands_its_tokens_to_the_space_panel(monkeypatch):
    seen = {}

    def _fake(img, ocr_tokens=None):
        seen['tokens'] = ocr_tokens
        return None

    monkeypatch.setattr('warp.recognition.layout_detector.detect_eq_geometry',
                        _fake)
    det = LayoutDetector.__new__(LayoutDetector)
    det._eq_geom_cache = {}
    det._img_key_memo = (None, '')
    det._ocr_tokens = [{'text': 'Fore'}]
    det._get_eq_geometry(np.zeros((40, 40, 3), dtype=np.uint8))
    assert seen['tokens'] == [{'text': 'Fore'}]


def test_a_detector_outside_a_detect_call_has_no_tokens():
    """`_ocr_tokens` starts as None so the standalone path is the fallback,
    not an accidental empty list that would look like 'read nothing'."""
    det = LayoutDetector.__new__(LayoutDetector)
    LayoutDetector.__init__(det)
    assert det._ocr_tokens is None
