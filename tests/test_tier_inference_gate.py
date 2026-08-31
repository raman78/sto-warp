"""A tier upgrade may only be inferred when a ship was actually identified.

When the tier badge is cropped out, `_infer_x_bonus` recovers the -X / -X2
upgrade by comparing measured slot counts against the ship's profile. Every
reading is a surplus over that profile, which is sound while the profile came
from ship_list.json and meaningless when it came from the keyword fallback:
the fallback carries `Universal Consoles = 0`, meaning "no ship was matched",
not "this ship has none.

Reported on a cropped SPACE_EQ screenshot with no ship header. Evidence came
out [Devices 5-4 = +1, Universal Consoles 2-0 = +2]; the max promoted the ship
to -X2, Devices grew from 4 to 6, and the row detector padded a sixth slot in
front of the five real ones. That phantom matched as `__inactive__` at
confidence 1.00 and was auto-accepted.

Run standalone:
    python -m pytest tests/test_tier_inference_gate.py -v
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
    # Never reach for the network: an empty cache would otherwise send cargo
    # to GitHub. Failing the fetch drops it to the committed baseline.
    monkeypatch.setattr(cargo, '_fetch', lambda *a, **k: (None, None, None))
    cargo._MEMO.clear()
    cargo._BUCKET_MEMO.clear()
    yield
    cargo._MEMO.clear()
    cargo._BUCKET_MEMO.clear()


# The counts from the reported screenshot: five devices and two universal
# consoles actually on screen.
_MEASURED = {'Fore Weapons': 5, 'Devices': 5, 'Universal Consoles': 2,
             'Aft Weapons': 3, 'Deflector': 1, 'Engines': 1, 'Warp Core': 1,
             'Shield': 1}

_TEXT_INFO = {'ship_name': '', 'ship_tier': '', 'ship_type_bbox': None,
              'ship_tier_bbox': None, 'build_type': 'SPACE'}


class _FakeText:
    def __init__(self, ship_type: str):
        self._info = dict(_TEXT_INFO, ship_type=ship_type)

    def extract_ship_info(self, img):
        return dict(self._info)

    def refine_ship_info(self, *a, **k):
        return dict(self._info)

    def scan_image(self, img):
        return []


class _FakeLayout:
    """Reports the measured counts the tier inference reads."""

    def __init__(self):
        self.last_row_pixel_counts = dict(_MEASURED)

    def detect(self, *a, **k):
        return {}

    def __getattr__(self, name):
        return lambda *a, **k: {}


def _run(tmp_path, ocr_ship_type: str):
    from warp.warp_importer import ShipDB, WarpImporter

    # A real ship so the identified case has something to match against.
    (tmp_path / 'ship_list.json').write_text(json.dumps([{
        'Page': 'Test Escort', 'name': 'Test Escort', 'type': 'Escort',
        'boffs': 'Commander Tactical,Ensign Science', 'tier': '6',
        'fore': '4', 'devices': '4', 'hangars': '',
    }]), encoding='utf-8')

    importer = WarpImporter(build_type='SPACE', from_trainer=False)
    importer._text = _FakeText(ocr_ship_type)
    importer._shipdb = ShipDB(tmp_path)
    importer._layout = _FakeLayout()
    importer._classify_screen = lambda img: ('SPACE_EQ', 0.96)

    return importer._process_image(
        np.zeros((561, 305, 3), dtype=np.uint8), 'test.png')


def test_devices_are_not_padded_past_what_was_measured(tmp_path):
    """The reported failure, at its visible consequence.

    Five devices on screen, no ship identified, so nothing may push the row
    past five and invent a slot in front of them.
    """
    profile = _run(tmp_path, '').ship_profile or {}

    assert profile.get('Devices', 0) <= _MEASURED['Devices'], (
        f"asked for {profile.get('Devices')} devices "
        f"with {_MEASURED['Devices']} on screen")


def test_universal_consoles_are_not_invented_either(tmp_path):
    """Two measured against a fallback zero is what produced the +2."""
    profile = _run(tmp_path, '').ship_profile or {}

    assert not profile.get('Universal Consoles')


def test_inference_still_runs_for_an_identified_ship(tmp_path):
    """The case the inference exists for — a known ship whose tier badge was
    cropped away — must keep working. Here the same measurements do raise the
    row, because the profile they are compared against is the ship's own.
    """
    result = _run(tmp_path, 'Test Escort')

    assert result.ship_type == 'Test Escort'
    assert (result.ship_profile or {}).get('Devices') == 6
