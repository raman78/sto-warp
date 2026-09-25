"""What the matcher is shown must be the cell the grid reports.

The trainer draws each item's bbox over the screenshot, and that picture is
the user's only view of what was recognised. The importer used to add a
per-image Y offset to every crop after an "anchor" slot (the P5 recalibration)
while reporting the unshifted bbox. When the grid-aligned Deflector scored
low and some crop up to 40 px away hit anything at 1.00 — in every measured
case a pHash collision — the whole rest of the panel was read up to 16 px
off the cell that was drawn.

Run standalone:
    python -m pytest tests/test_warp_importer_crops.py -v
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('cv2')


@pytest.fixture(autouse=True)
def _isolate_user_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_CACHE_DIR', str(tmp_path / 'cargo-cache'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    from warp.data import cargo
    monkeypatch.setattr(cargo, '_fetch', lambda *a, **k: (None, None, None))
    cargo._MEMO.clear()
    cargo._BUCKET_MEMO.clear()
    yield
    cargo._MEMO.clear()
    cargo._BUCKET_MEMO.clear()


_SHIP = 'Deimos Pilot Destroyer'

# One column of cells, 50 px pitch — the shape of the real equipment panel.
_GRID = {
    'Fore Weapons': [(200, 10, 35, 44)],
    'Deflector':    [(200, 60, 35, 44)],
    'Engines':      [(200, 110, 35, 44)],
    'Warp Core':    [(200, 160, 35, 44)],
    'Shield':       [(200, 210, 35, 44)],
}


class _FakeText:
    def extract_ship_info(self, img):
        return {'ship_name': '', 'ship_type': _SHIP, 'ship_tier': 'T6',
                'ship_type_bbox': (10, 10, 100, 20),
                'ship_tier_bbox': (120, 10, 40, 20), 'build_type': 'SPACE'}

    def refine_ship_info(self, *a, **k):
        return self.extract_ship_info(None)

    def scan_image(self, img):
        return []


class _FakeLayout:
    def __init__(self):
        counts = {slot: len(b) for slot, b in _GRID.items()}
        self.last_row_pixel_counts = dict(counts)
        self.last_row_cell_counts = dict(counts)
        self.last_trait_icon_counts: dict = {}

    def detect(self, img, build_type, profile=None, **k):
        return {slot: list(b) for slot, b in _GRID.items()}

    def __getattr__(self, name):
        return lambda *a, **k: {}


class _FakeMatcher:
    """Weak on every grid cell, certain on anything else.

    That is the condition under which an anchor search would move the grid:
    the cell itself is doubtful, and a crop beside it looks perfect.
    """

    def __init__(self, img):
        self._cells = {img[y:y + h, x:x + w].tobytes()
                       for b in _GRID.values() for (x, y, w, h) in b}
        self.seen: list[bytes] = []
        # Diagnostics the importer reads back after every match.
        self._last_stage_scores: dict = {}
        self._last_match_src = ''
        self._last_match_origin = ''
        self._last_match_variant = ''

    def match(self, crop, candidate_names=None):
        key = crop.tobytes()
        self.seen.append(key)
        if key in self._cells:
            return 'Grid Item', 0.50, None, False
        return 'Off Grid Item', 1.00, None, False

    def __getattr__(self, name):
        return lambda *a, **k: None


def _run(tmp_path):
    from warp.warp_importer import ShipDB, WarpImporter

    (tmp_path / 'ship_list.json').write_text(json.dumps([{
        'Page': _SHIP, 'name': _SHIP, 'type': 'Destroyer',
        'boffs': 'Commander Tactical-Pilot,Ensign Universal', 'tier': '6',
        'fore': '1', 'devices': '3', 'hangars': '',
    }]), encoding='utf-8')
    shot = tmp_path / 'shot.png'
    shot.write_bytes(b'\x89PNG\r\n\x1a\n' + b'pretend pixels')

    # Distinct noise everywhere, so no two crops are byte-identical and every
    # cell reads as holding an icon.
    img = np.random.default_rng(7).integers(0, 256, (300, 300, 3), dtype=np.uint8)

    importer = WarpImporter(build_type='SPACE', from_trainer=False)
    importer._text = _FakeText()
    importer._shipdb = ShipDB(tmp_path)
    importer._layout = _FakeLayout()
    importer._classify_screen = lambda img: ('SPACE_EQ', 0.96)
    matcher = _FakeMatcher(img)
    importer._matcher = matcher

    result = importer._process_image(img, str(shot))
    return img, matcher, result


def test_every_crop_the_matcher_sees_is_a_grid_cell(tmp_path, monkeypatch):
    import warp.recognition.layout_detector as ld
    monkeypatch.setattr(ld.LayoutDetector, '_classify_cell',
                        staticmethod(lambda crop: 'active'))
    img, matcher, _ = _run(tmp_path)
    grid_cells = {img[y:y + h, x:x + w].tobytes()
                  for b in _GRID.values() for (x, y, w, h) in b}

    assert matcher.seen, 'the matcher was never called — test exercises nothing'
    assert all(c in grid_cells for c in matcher.seen)


def test_a_doubtful_anchor_does_not_move_the_cells_below_it(tmp_path, monkeypatch):
    """The reported failure: after a weak Deflector, Engines to Shield were
    read from crops below their drawn boxes and named from those."""
    import warp.recognition.layout_detector as ld
    monkeypatch.setattr(ld.LayoutDetector, '_classify_cell',
                        staticmethod(lambda crop: 'active'))
    _, _, result = _run(tmp_path)
    below = [it for it in result.items
             if it.slot in ('Engines', 'Warp Core', 'Shield')]

    assert below
    assert all(it.name == 'Grid Item' for it in below)
