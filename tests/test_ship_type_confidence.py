"""Ship Type confidence must reflect what the lookup actually established.

Ship Type used to be emitted at a flat 1.0 regardless of how ShipDB reached
it. Four of the lookup strategies read `_by_type`, which is keyed on the
generic `type` field and therefore holds an arbitrary member of the class —
measured, its slot profile is wrong on 4.9 of 11 slots for the ship really on
screen. Reporting that as certain meant WARP CORE auto-accepted it into
`annotations.json` as ground truth.

Run standalone:
    python -m pytest tests/test_ship_type_confidence.py -v
"""
from __future__ import annotations

import json

import pytest

# The lowest value WARP CORE's auto-accept spinbox allows
# (`trainer_window.py`: `_spin_auto_conf.setRange(0.5, 1.0)`). Anything a user
# must review by hand has to sit strictly below this, not at it.
MIN_AUTO_ACCEPT_SETTING = 0.5


@pytest.fixture(autouse=True)
def _isolate_user_dirs(monkeypatch, tmp_path):
    """Keep the importer off the user's real data.

    `WarpImporter.__init__` builds a cargo cache view, and `_process_image`
    appends to the recognition history under `training_data_dir()` and writes
    the detection log — all of which resolve through the XDG basedirs in
    `warp.userdata`. Without redirecting every one of them, running this file
    edits the maintainer's own `recog_history.json`.
    """
    monkeypatch.setenv('WARP_CACHE_DIR', str(tmp_path / 'cargo-cache'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    from warp.data import cargo
    cargo._MEMO.clear()
    cargo._BUCKET_MEMO.clear()
    yield
    cargo._MEMO.clear()
    cargo._BUCKET_MEMO.clear()


def _resolution(strategy: str, matched: bool = True):
    from warp.warp_importer import ShipResolution

    return ShipResolution(
        name='', type='Some Cruiser', tier='T6', profile={},
        strategy=strategy, matched=matched, ocr_name='', ocr_type='',
    )


@pytest.mark.parametrize('strategy', [
    'exact-type', 'word-subset', 'word-subset-best', 'fuzzy-type',
])
def test_class_only_strategies_are_not_auto_acceptable(strategy):
    """These four resolve to a class and hand back an arbitrary member."""
    from warp.warp_importer import ship_type_confidence

    assert ship_type_confidence(_resolution(strategy)) < MIN_AUTO_ACCEPT_SETTING


@pytest.mark.parametrize('strategy', [
    'display-name', 'display-name-best', 'fuzzy-display',
    'token-overlap', 'anchorless-rescue',
])
def test_ship_level_strategies_stay_fully_confident(strategy):
    """These identify a specific ship_list.json entry — unchanged behaviour."""
    from warp.warp_importer import ship_type_confidence

    assert ship_type_confidence(_resolution(strategy)) == 1.0


def test_unmatched_lookup_is_unverified():
    from warp.warp_importer import ship_type_confidence

    conf = ship_type_confidence(_resolution('keyword-fallback', matched=False))

    assert conf < MIN_AUTO_ACCEPT_SETTING


def test_absent_resolution_is_unverified():
    """Trait-only panels skip the ShipDB lookup entirely."""
    from warp.warp_importer import ship_type_confidence

    assert ship_type_confidence(None) < MIN_AUTO_ACCEPT_SETTING


def test_class_only_still_outranks_a_bare_ocr_read():
    """Knowing the right class is worth more than knowing nothing, so the two
    low bands stay ordered — the review list sorts on this."""
    from warp.warp_importer import ship_type_confidence

    assert (ship_type_confidence(_resolution('fuzzy-type'))
            > ship_type_confidence(None))


def _ship(name: str, stype: str) -> dict:
    return {
        'Page': name, 'name': name, 'type': stype,
        'boffs': 'Commander Tactical,Ensign Science',
        'tier': '6', 'fore': '4', 'hangars': '',
    }


def test_a_real_class_read_lands_below_auto_accept(tmp_path):
    """End to end through ShipDB: reading only the class must not produce an
    auto-acceptable Ship Type, whichever ship the index happens to hold."""
    from warp.warp_importer import ShipDB, ship_type_confidence

    (tmp_path / 'ship_list.json').write_text(json.dumps([
        _ship('Zahl Heavy Cruiser', 'Cruiser'),
        _ship('Excelsior Cruiser', 'Cruiser'),
    ]), encoding='utf-8')

    resolution = ShipDB(tmp_path).resolve('', 'Cruiser', 'T6')

    assert resolution.strategy in {'exact-type', 'word-subset',
                                   'word-subset-best', 'fuzzy-type'}
    assert ship_type_confidence(resolution) < MIN_AUTO_ACCEPT_SETTING


def test_a_named_ship_read_stays_confident(tmp_path):
    from warp.warp_importer import ShipDB, ship_type_confidence

    (tmp_path / 'ship_list.json').write_text(json.dumps([
        _ship('Zahl Heavy Cruiser', 'Cruiser'),
        _ship('Excelsior Cruiser', 'Cruiser'),
    ]), encoding='utf-8')

    resolution = ShipDB(tmp_path).resolve('', 'Excelsior Cruiser', 'T6')

    assert resolution.type == 'Excelsior Cruiser'
    assert ship_type_confidence(resolution) == 1.0


# ── The call site ──────────────────────────────────────────────────────────
#
# Everything above tests the mapping. These drive `_process_image` itself, so
# that reverting the emission line back to a flat 1.0 fails a test rather than
# passing silently.
#
# `_process_image` normally needs a real screenshot and the OCR/ML stack. All
# of that reaches it through lazy getters which return a pre-set attribute
# untouched, so the collaborators can simply be assigned before the call.

np = pytest.importorskip('numpy')


_TEXT_INFO = {
    'ship_name': '',
    'ship_tier': 'T6',
    'ship_type_bbox': (10, 10, 100, 20),
    'ship_tier_bbox': (120, 10, 40, 20),
    'build_type': 'SPACE',
}


class _FakeText:
    """Stands in for TextExtractor: hands back a fixed OCR reading."""

    def __init__(self, ship_type: str):
        self._info = dict(_TEXT_INFO, ship_type=ship_type)

    def extract_ship_info(self, img):
        return dict(self._info)

    def refine_ship_info(self, *args, **kwargs):
        return dict(self._info)

    def scan_image(self, img):
        return []                      # no OCR tokens; layout stays empty


class _FakeLayout:
    """Stands in for LayoutDetector: finds no equipment rows.

    `last_row_pixel_counts` and `last_trait_icon_counts` are read as mappings
    by `_infer_x_bonus`, so they have to be real dicts rather than another
    stub method.
    """

    last_row_pixel_counts: dict = {}
    last_trait_icon_counts: dict = {}

    def __getattr__(self, name):
        return lambda *args, **kwargs: {}


def _run_importer(tmp_path, ocr_ship_type: str):
    from warp.warp_importer import ShipDB, WarpImporter

    (tmp_path / 'ship_list.json').write_text(json.dumps([
        _ship('Zahl Heavy Cruiser', 'Cruiser'),
        _ship('Excelsior Cruiser', 'Cruiser'),
    ]), encoding='utf-8')

    importer = WarpImporter(build_type='SPACE', from_trainer=False)
    importer._text = _FakeText(ocr_ship_type)
    importer._shipdb = ShipDB(tmp_path)
    importer._layout = _FakeLayout()
    importer._classify_screen = lambda img: ('SPACE_EQ', 0.99)

    result = importer._process_image(
        np.zeros((600, 700, 3), dtype=np.uint8), 'test.png')
    return {item.slot: item for item in result.items}


def test_emitted_class_only_ship_type_is_not_auto_acceptable(tmp_path):
    """The l1.png failure, at the point where confidence is actually set."""
    items = _run_importer(tmp_path, 'Cruiser')

    assert items['Ship Type'].confidence < MIN_AUTO_ACCEPT_SETTING


def test_emitted_named_ship_type_keeps_full_confidence(tmp_path):
    items = _run_importer(tmp_path, 'Excelsior Cruiser')

    assert items['Ship Type'].name == 'Excelsior Cruiser'
    assert items['Ship Type'].confidence == 1.0


def test_emitted_ship_tier_is_unaffected_by_the_class_lookup(tmp_path):
    """The tier comes off the badge and never consults ShipDB, so a class-only
    class match may not drag its confidence down with it.

    The badge has a grade of its own — see tests/test_ship_tier_confidence.py,
    which owns what that grade is.
    """
    from warp.warp_importer import SHIP_TIER_CONF_BADGE

    items = _run_importer(tmp_path, 'Cruiser')

    assert items['Ship Tier'].confidence == SHIP_TIER_CONF_BADGE
