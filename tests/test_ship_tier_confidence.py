"""The Ship Tier row must show the tier the build was actually sized from.

The tier is not a label, it is a slot count: `_apply_ship_and_tier_bonuses`
grants +1 Universal Console, Device and Starship Trait for `-X` and +2 for
`-X2`. Three sources can produce it — the OCR badge, a tier the user
confirmed in WARP CORE, and a raise argued from the measured rows — and the
importer already prefers them in that order for everything downstream.

The review row did not follow. It read the raw OCR field and reported a flat
1.00, so a screenshot whose badge read `T1` while the layout was built from
the user's confirmed `T6` offered `T1` as certain: with auto-accept enabled
that writes T1 back over the confirmed row, and the next Auto-Detect sizes
the grid from it.

Run standalone:
    python -m pytest tests/test_ship_tier_confidence.py -v
"""
from __future__ import annotations

import hashlib
import json

import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('cv2')

# The lowest value WARP CORE's auto-accept spinbox allows
# (`trainer_window.py`: `_spin_auto_conf.setRange(0.5, 1.0)`). Anything a user
# must review by hand has to sit strictly below this, not at it.
MIN_AUTO_ACCEPT_SETTING = 0.5


@pytest.fixture(autouse=True)
def _isolate_user_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_CACHE_DIR', str(tmp_path / 'cargo-cache'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    from warp.data import cargo
    # Never reach for the network: an empty cache would otherwise send cargo
    # to GitHub. Failing the fetch drops it to the committed baseline.
    monkeypatch.setattr(cargo, '_fetch', lambda *a, **k: (None, None, None))
    cargo._MEMO.clear()
    cargo._BUCKET_MEMO.clear()
    yield
    cargo._MEMO.clear()
    cargo._BUCKET_MEMO.clear()


# The ship is a real one from the committed cargo baseline: `_compose_
# inferred_tier` reads its base tier out of `ships.json`, so a made-up name
# would leave the raise unnamed and the row unchanged.
_SHIP = 'Deimos Pilot Destroyer'

# Counts the row detector reports off the panel. The Deimos carries three
# devices and no universal console, so five and two are a +2 surplus.
_MEASURED = {'Fore Weapons': 5, 'Devices': 5, 'Universal Consoles': 2,
             'Deflector': 1, 'Engines': 1, 'Warp Core': 1, 'Shield': 1}

_NO_SURPLUS = {'Fore Weapons': 5, 'Devices': 3, 'Deflector': 1,
               'Engines': 1, 'Warp Core': 1, 'Shield': 1}


class _FakeText:
    """Stands in for TextExtractor: hands back a fixed OCR reading."""

    def __init__(self, ship_type: str, ship_tier: str):
        self._info = {
            'ship_name': '', 'ship_type': ship_type, 'ship_tier': ship_tier,
            'ship_type_bbox': (10, 10, 100, 20),
            'ship_tier_bbox': (120, 10, 40, 20) if ship_tier else None,
            'build_type': 'SPACE',
        }

    def extract_ship_info(self, img):
        return dict(self._info)

    def refine_ship_info(self, *a, **k):
        return dict(self._info)

    def scan_image(self, img):
        return []


class _FakeLayout:
    """Stands in for the row detector: reports fixed pixel counts."""

    def __init__(self, measured: dict):
        self.last_row_pixel_counts = dict(measured)

    def detect(self, img, build_type, profile=None, **k):
        profile = profile or {}
        return {slot: [(0, 0, 10, 10)] * profile.get(slot, 0)
                for slot in self.last_row_pixel_counts
                if profile.get(slot, 0) > 0}

    def __getattr__(self, name):
        return lambda *a, **k: {}


def _run(tmp_path, *, ocr_tier: str, measured: dict = None,
         confirmed_tier: str = '') -> dict:
    """Drive `_process_image` and return its items keyed by slot.

    A confirmed tier switches the call to the trainer path, which is the only
    one allowed to read annotations.json.
    """
    from warp.warp_importer import ShipDB, WarpImporter

    (tmp_path / 'ship_list.json').write_text(json.dumps([{
        'Page': _SHIP, 'name': _SHIP, 'type': 'Destroyer',
        'boffs': 'Commander Tactical-Pilot,Ensign Universal', 'tier': '6',
        'fore': '5', 'devices': '3', 'hangars': '',
    }]), encoding='utf-8')

    shot = tmp_path / 'shot.png'
    shot.write_bytes(b'\x89PNG\r\n\x1a\n' + b'pretend pixels')
    if confirmed_tier:
        from warp import userdata
        sha16 = hashlib.sha256(shot.read_bytes()).hexdigest()[:16]
        (userdata.training_data_dir() / 'annotations.json').write_text(
            json.dumps({sha16: {'filename': shot.name, 'annotations': [{
                'slot': 'Ship Tier', 'name': confirmed_tier,
                'state': 'confirmed', 'auto_confirmed': False,
                'bbox': [120, 10, 40, 20]}]}}), encoding='utf-8')

    importer = WarpImporter(build_type='SPACE',
                            from_trainer=bool(confirmed_tier))
    importer._text = _FakeText(_SHIP, ocr_tier)
    importer._shipdb = ShipDB(tmp_path)
    importer._layout = _FakeLayout(measured or _NO_SURPLUS)
    importer._classify_screen = lambda img: ('SPACE_EQ', 0.96)

    result = importer._process_image(
        np.zeros((561, 305, 3), dtype=np.uint8), str(shot))
    return {item.slot: item for item in result.items}


# ── What the row shows ─────────────────────────────────────────────────────

def test_a_tier_the_user_confirmed_is_what_the_row_shows(tmp_path):
    """The reported failure. `[T6]` misread as `T1`, `T6` confirmed by the
    user: the layout was sized from T6, so T6 is what the review row has to
    offer — otherwise accepting it overwrites the confirmation."""
    items = _run(tmp_path, ocr_tier='T1', confirmed_tier='T6')

    assert items['Ship Tier'].name == 'T6'


def test_a_confirmed_tier_is_reported_as_certain(tmp_path):
    """Pins behaviour that predates the fix — the emitted tier was already
    reported at 1.00, wrongly for every source. It is the value above that
    changed. Kept so a future grading pass cannot quietly demote the one
    source that really is ground truth."""
    items = _run(tmp_path, ocr_tier='T1', confirmed_tier='T6')

    assert items['Ship Tier'].confidence == 1.0


def test_a_raised_tier_is_what_the_row_shows(tmp_path):
    """Five devices against a three-device ship is +2, so the build was
    sized from `-X2` and the row may not still say `T6`."""
    items = _run(tmp_path, ocr_tier='T6', measured=_MEASURED)

    assert items['Ship Tier'].name.endswith('-X2')


# ── What the row claims ────────────────────────────────────────────────────

def test_a_badge_read_is_not_reported_as_certain(tmp_path):
    """Measured over the 80 screenshots in annotations.json carrying a
    user-confirmed tier, the badge read agrees with the user on 74 of the 79
    it answered (73, plus one where the store is the wrong side) — `T1` for
    `[T6]` being the commonest miss."""
    items = _run(tmp_path, ocr_tier='T6')

    assert items['Ship Tier'].confidence < 1.0


def test_a_raised_tier_must_be_reviewed_by_hand(tmp_path):
    """The raise is a lower bound argued from pixel counts and its accuracy
    has never been measured, so auto-accept may not take it."""
    items = _run(tmp_path, ocr_tier='T6', measured=_MEASURED)

    assert items['Ship Tier'].confidence < MIN_AUTO_ACCEPT_SETTING


def test_a_raised_tier_is_marked_as_inferred(tmp_path):
    """`src='inferred'` is what puts the marker on the review row."""
    items = _run(tmp_path, ocr_tier='T6', measured=_MEASURED)

    assert items['Ship Tier'].src == 'inferred'
