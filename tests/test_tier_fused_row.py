"""A tier badge fused with the line under it must still be read, and boxed.

Regression, the mirror image of `test_tier_badge_split`: instead of cutting the
badge in half, the whole-screen scan merged `[T6-X2]` with the registry line
`(NCC-93015-B)` directly below it into one 50 px box reading `'{TEX23015-8)'`.
That token has no closed bracket for the bracket anchors and no `T<digit>` for
the loose one, so the screenshot fell through to the anchorless path: the tier
was never read at all, the importer inferred it from the slot counts instead,
and the review box was drawn around the class line — which is not where the
tier is.

Tokens below are the real OCR output for that screenshot (902x849); the
re-read fixture is the real output of `rescan_region` over the fused token.
Feeding both in exercises the whole path without running OCR.
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


# Top band of STO_screens/screeny2/image-4073e52ef5e3376f.png, verbatim.
# The ship name above the class line came back at conf 0.01 and is dropped by
# the band filter, which is why no name anchor rescues this screenshot either.
_TOKENS = [
    _tok('Fore', 527, 0, 573, 19, 1.00),
    _tok('Weapons', 483, 17, 574, 43, 0.99),
    _tok('Terran Lexington Dreadnought Cruiser', 40, 22, 396, 50, 0.79),
    _tok('{TEX23015-8)', 36, 42, 190, 92, 0.53),
    _tok('Deflector', 482, 70, 574, 94, 1.00),
    _tok('Impulse', 495, 133, 574, 161, 1.00),
    _tok('7puis8', 497, 144, 573, 159, 0.25),
]

# What the reader returns for the fused token's pixels alone, at 2x.
_REREAD = [
    _tok('[T6-X2]', 41, 47, 115, 67, 0.80),
    _tok('(NCC-93015-8)', 40, 67, 186, 89, 0.97),
]

_FUSED_BOX = (36, 42, 154, 50)       # the token the scan produced
_BADGE_BOX = (41, 47, 74, 20)        # where '[T6-X2]' actually sits
_CLASS_BOX = (40, 22, 356, 28)       # where the class line actually sits


def _extract(monkeypatch, tokens=_TOKENS, reread=_REREAD, calls=None):
    img = np.zeros((849, 902, 3), dtype=np.uint8)
    te = TextExtractor()
    te._scan_cache_key = id(img)          # feed tokens, skip OCR
    te._scan_cache_tokens = tokens

    def _fake_rescan(_img, bbox, scale=2.0):
        if calls is not None:
            calls.append(tuple(bbox))
        return reread

    monkeypatch.setattr(te, 'rescan_region', _fake_rescan)
    return te.extract_ship_info(img)


@pytest.fixture
def info(monkeypatch):
    return _extract(monkeypatch)


def test_fused_badge_is_read_as_a_tier(info):
    assert info['ship_tier'] == 'T6-X2'


def test_tier_bbox_is_the_badge_alone(info):
    """Not the fused token's box — that one also covers the registry line."""
    assert tuple(info['ship_tier_bbox']) == _BADGE_BOX


def test_tier_bbox_stops_above_the_registry_line(info):
    _x, y, _w, h = info['ship_tier_bbox']
    assert y + h <= 67          # '(NCC-93015-8)' starts at y=67


def test_tier_bbox_is_not_the_class_line(info):
    """The old behaviour: no badge box, so the tier borrowed the type's."""
    assert tuple(info['ship_tier_bbox']) != _CLASS_BOX


def test_class_line_is_still_read_from_the_row_above(info):
    assert info['ship_type'] == 'Terran Lexington Dreadnought Cruiser'
    assert tuple(info['ship_type_bbox']) == _CLASS_BOX


def test_registry_line_does_not_become_the_type(info):
    assert 'NCC' not in info['ship_type']
    assert 'TEX' not in info['ship_type']


def test_equipment_column_does_not_leak_into_the_type(info):
    for hud in ('Fore', 'Weapons', 'Deflector', 'Impulse'):
        assert hud not in info['ship_type']


def test_no_anchorless_fallback_needed(info):
    assert info['anchorless_candidates'] == []


def test_only_the_over_tall_token_is_re_read(monkeypatch):
    """One OCR call, spent on the one token that cannot be a single line."""
    calls: list[tuple] = []
    _extract(monkeypatch, calls=calls)
    assert calls == [_FUSED_BOX]


def test_normal_height_tokens_are_never_re_read(monkeypatch):
    """Same screenshot with the fused token trimmed to one line's height:
    nothing is tall enough to suspect, so no re-read is attempted."""
    tokens = [t if t['text'] != '{TEX23015-8)'
              else _tok(t['text'], 36, 42, 190, 68, 0.53)
              for t in _TOKENS]
    calls: list[tuple] = []
    info = _extract(monkeypatch, tokens=tokens, calls=calls)
    assert calls == []
    assert info['ship_tier'] == ''


def test_a_re_read_without_a_badge_invents_no_tier(monkeypatch):
    """Height alone never establishes a tier — the re-read has to show one."""
    reread = [_tok('(NCC-93015-8)', 40, 67, 186, 89, 0.97)]
    info = _extract(monkeypatch, reread=reread)
    assert info['ship_tier'] == ''
    assert info['ship_tier_bbox'] is None


def test_a_re_read_that_is_not_a_real_tier_is_refused(monkeypatch):
    """A bracket is not enough; the content must snap to SHIP_TIER_VALUES."""
    reread = [_tok('[MK XV]', 41, 47, 115, 67, 0.80)]
    info = _extract(monkeypatch, reread=reread)
    assert info['ship_tier'] == ''


def test_rescan_region_returns_full_image_coordinates():
    """The re-read runs on an upscaled crop; its boxes have to come back in
    the same coordinate space as the tokens they are compared against."""
    class _FakeReader:
        def readtext(self, _img, **_kw):
            # A 2x crop taken at (36, 42): a box at (10, 10)-(158, 50) in crop
            # space is (41, 47)-(115, 67) in the full image.
            return [([[10, 10], [158, 10], [158, 50], [10, 50]], '[T6-X2]', 0.8)]

    te = TextExtractor()
    te._ocr = _FakeReader()
    img = np.zeros((849, 902, 3), dtype=np.uint8)
    out = te.rescan_region(img, _FUSED_BOX)
    assert len(out) == 1
    assert (out[0]['x0'], out[0]['y0'], out[0]['w'], out[0]['h']) == _BADGE_BOX


def test_rescan_region_survives_an_out_of_frame_box():
    te = TextExtractor()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert te.rescan_region(img, (500, 500, 40, 20)) == []
