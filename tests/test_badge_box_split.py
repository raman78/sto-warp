"""The tier badge gets a box of its own, even when it shares a line.

Anchors 1 and 1b find the tier inside a token like
`'Verne Temporal Science Vessel [T6-X2]'` and used to give the whole token's
rectangle to *both* `ship_tier_bbox` and `ship_type_bbox`. The tier string was
right and the rectangle was a picture of a ship's name.

Measured 2026-09-05: 57 % of confirmed `Ship Tier` rows in one maintainer's
store carried a box shared with `Ship Type`, and 73 of 109 tier boxes over the
172-screenshot corpus. Not an edge case — the normal one. Downstream it is a
crop of the wrong thing, and because identical pixels give an identical hash it
merged the tier and class ballots on the server, which is how
`Fleet Yamaguchi Support Cruiser` came to be published as a `Ship Tier`.

The separation does not come from a bigger picture — that is pass 1d's trick
and it works only because there the two lines are stacked vertically. Here the
badge sits *beside* the class name, so the reader is right to return one line.
It comes from asking for a different grouping: EasyOCR merges neighbouring
detections whose horizontal gap is under `width_ths` of their height.

Offline: the reader is replaced, so no OCR runs.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from warp.recognition.text_extractor import TextExtractor


def _tok(text: str, x0: int, y0: int, x1: int, y1: int, conf: float) -> dict:
    return {
        'text': text, 'low': text.lower(), 'conf': conf,
        'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
        'cx': (x0 + x1) // 2, 'cy': (y0 + y1) // 2,
        'w': x1 - x0, 'h': y1 - y0,
    }


# Top band of 1-ba54d6e861e08f02.png, verbatim. The class name and the badge
# came back as one 292 px token.
_FUSED = _tok('Verne Temporal Science Vessel [T6-X2]', 49, 25, 341, 45, 0.66)
_TOKENS = [
    _tok('0.8.5. ILLINOIS', 48, 0, 234, 26, 0.55),
    _tok('Fore', 517, 7, 553, 23, 1.00),
    _tok('Weapons', 482, 22, 555, 42, 1.00),
    _FUSED,
    _tok('{NCV-93676)', 47, 43, 151, 63, 0.68),
]

# What the reader returns for the fused token's pixels at width_ths=0.05,
# in full-image coordinates. The badge is the last box.
_SPLIT = [
    _tok('Verne', 49, 27, 100, 44, 1.00),
    _tok('Temporal', 105, 27, 180, 44, 0.74),
    _tok('Science', 185, 27, 245, 44, 1.00),
    _tok('Vessel', 250, 27, 300, 44, 1.00),
    _tok('[T6-X2]', 280, 27, 340, 44, 0.61),
]

_BADGE_BOX = (280, 27, 60, 17)
_FUSED_BOX = (49, 25, 292, 20)


def _extract(monkeypatch, reread=_SPLIT, calls=None):
    img = np.zeros((849, 902, 3), dtype=np.uint8)
    te = TextExtractor()
    te._scan_cache_key = id(img)
    te._scan_cache_tokens = _TOKENS

    def _fake_rescan(_img, bbox, scale=2.0, **kw):
        if calls is not None:
            calls.append((tuple(bbox), scale, kw))
        return reread

    monkeypatch.setattr(te, 'rescan_region', _fake_rescan)
    return te.extract_ship_info(img)


@pytest.fixture
def info(monkeypatch):
    return _extract(monkeypatch)


def test_the_tier_is_still_read(info):
    assert info['ship_tier'] == 'T6-X2'


def test_the_tier_gets_the_badge_box_not_the_whole_line(info):
    assert tuple(info['ship_tier_bbox']) == _BADGE_BOX


def test_the_tier_box_is_no_longer_the_class_box(info):
    """The defect itself: two rows, one rectangle, one crop hash."""
    assert tuple(info['ship_tier_bbox']) != tuple(info['ship_type_bbox'])


def test_the_class_line_keeps_its_own_box(info):
    """Only the tier box was wrong. The class line really is that wide."""
    assert tuple(info['ship_type_bbox']) == _FUSED_BOX
    assert info['ship_type'] == 'Verne Temporal Science Vessel'


def test_the_re_read_asks_for_a_narrower_grouping(monkeypatch):
    """Upscaling alone leaves the line intact — it is `width_ths` that splits
    text sitting side by side."""
    calls: list[tuple] = []
    _extract(monkeypatch, calls=calls)
    assert len(calls) == 1
    bbox, scale, kw = calls[0]
    assert bbox == _FUSED_BOX
    assert scale > 1.0
    assert kw.get('width_ths', 1.0) < 0.5      # below EasyOCR's default


def test_a_re_read_that_cannot_separate_them_keeps_the_old_box(monkeypatch):
    """The measured failure of the first attempt: at the default grouping the
    reader hands back the same fused line. Keeping the wide box is wrong but
    it is what we had; swapping it for an equally wide one gains nothing."""
    info = _extract(monkeypatch, reread=[
        _tok('Verne Temporal Science Vessel [T6-X2]', 49, 27, 340, 44, 0.89)])
    assert tuple(info['ship_tier_bbox']) == _FUSED_BOX


def test_a_re_read_with_no_bracket_changes_nothing(monkeypatch):
    info = _extract(monkeypatch, reread=[_tok('Verne', 49, 27, 100, 44, 1.0)])
    assert tuple(info['ship_tier_bbox']) == _FUSED_BOX
    assert info['ship_tier'] == 'T6-X2'


def test_a_bracket_that_is_not_a_tier_is_refused(monkeypatch):
    """A narrow box is not enough — its text has to snap to a real tier."""
    info = _extract(monkeypatch, reread=[_tok('[MK XV]', 280, 27, 340, 44, 0.9)])
    assert tuple(info['ship_tier_bbox']) == _FUSED_BOX


def test_a_badge_alone_in_its_token_is_not_re_read(monkeypatch):
    """Nothing to separate: the token is the badge, so the box is already
    right and an OCR call would buy nothing."""
    calls: list[tuple] = []
    img = np.zeros((849, 902, 3), dtype=np.uint8)
    te = TextExtractor()
    te._scan_cache_key = id(img)
    te._scan_cache_tokens = [
        _tok('Verne Temporal Science Vessel', 49, 25, 300, 45, 0.9),
        _tok('[T6-X2]', 310, 25, 370, 45, 0.7),
    ]
    monkeypatch.setattr(te, 'rescan_region',
                        lambda *a, **k: calls.append(a) or [])
    info = te.extract_ship_info(img)
    assert info['ship_tier'] == 'T6-X2'
    assert calls == []


def test_split_badge_box_returns_none_when_nothing_is_narrower():
    te = TextExtractor()
    te.rescan_region = lambda *a, **k: [
        _tok('Verne Temporal Science Vessel [T6-X2]', 49, 27, 340, 44, 0.9)]
    assert te.split_badge_box(np.zeros((10, 10, 3), dtype=np.uint8),
                              {'x': 49, 'y': 25, 'w': 292, 'h': 20}) is None


def test_split_badge_box_prefers_the_narrowest_candidate():
    te = TextExtractor()
    te.rescan_region = lambda *a, **k: [
        _tok('Vessel [T6-X2]', 250, 27, 340, 44, 0.6),   # 90 px
        _tok('[T6-X2]', 280, 27, 340, 44, 0.6),          # 60 px
    ]
    assert te.split_badge_box(np.zeros((10, 10, 3), dtype=np.uint8),
                              {'x': 49, 'y': 25, 'w': 292, 'h': 20}) \
        == (280, 27, 60, 17)
