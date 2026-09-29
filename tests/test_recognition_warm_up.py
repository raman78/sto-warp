"""The first recognition's one-off loading happens at start, in the background.

On a fresh process the first Auto-Detect took 19.6 s on
image-cda05d5238072b99.png, ~13 s of it loading that every later recognition
reuses (OCR networks, icon index, models, confirmed crops, their stacked
copy). `SETSIconMatcher.warm_up` does that at start — as the `warm` step of
the launcher's sync cycle and from a standalone WARP CORE — under the lock
the builders take, so a recognition started meanwhile waits rather than
repeating the work.

Offline: no network; OCR and models are stubbed where they would load.

Run standalone:
    python -m pytest tests/test_recognition_warm_up.py -v
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Class state and the OCR loader, restored afterwards."""
    from warp.recognition import icon_matcher as IM
    from warp.recognition import text_extractor as TE
    C = IM.SETSIconMatcher
    saved = (C._session_examples, C._sess_stack, C._index_cache,
             C._seeded_from_training_data, C._seeded_from_community, C._warming)
    C._session_examples, C._sess_stack, C._index_cache = [], None, None
    monkeypatch.setattr(TE, 'shared_reader', lambda: None)
    monkeypatch.setattr('warp.data.cargo.canonical_names', lambda: set())
    yield
    (C._session_examples, C._sess_stack, C._index_cache,
     C._seeded_from_training_data, C._seeded_from_community, C._warming) = saved


def _crop(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (4, 4, 3), dtype=np.uint8)
    return cv2.resize(base, (49, 64), interpolation=cv2.INTER_NEAREST)


@pytest.fixture
def icons(tmp_path, monkeypatch):
    d = tmp_path / 'icons'
    d.mkdir()
    for i in range(5):
        cv2.imwrite(str(d / f'Icon {i}.png'), _crop(i))
    monkeypatch.setattr('warp.data.cargo.icons_dir', lambda: d)
    from warp.recognition.icon_matcher import SETSIconMatcher
    monkeypatch.setattr(SETSIconMatcher, '_get_ml_session', lambda self: None)
    return d


# ── What the warm-up leaves behind ─────────────────────────────────────────

def test_after_warm_up_a_new_matcher_reuses_the_index(icons):
    from warp.recognition.icon_matcher import SETSIconMatcher
    SETSIconMatcher.warm_up()
    built = SETSIconMatcher._index_cache
    assert built is not None
    assert SETSIconMatcher()._tmpl_mat64 is built[3]


def test_after_warm_up_the_session_stack_is_ready(icons, monkeypatch):
    from warp.recognition.icon_matcher import SETSIconMatcher

    def _seed(*a, **k):
        SETSIconMatcher.add_session_example(_crop(7), 'Community Item', origin='community')
        return 1
    monkeypatch.setattr(SETSIconMatcher, '_seed_from_community_crops', classmethod(_seed))
    SETSIconMatcher.warm_up()
    stack = SETSIconMatcher._sess_stack
    assert stack is not None and len(stack[1]) == 1
    assert SETSIconMatcher._session_stack() is stack


def test_a_failing_step_does_not_stop_the_rest(icons, monkeypatch):
    """Everything here is what the first recognition would do itself, so a
    failure costs the head start and nothing else."""
    from warp.recognition import text_extractor as TE
    from warp.recognition.icon_matcher import SETSIconMatcher

    def _boom():
        raise RuntimeError('no OCR here')
    monkeypatch.setattr(TE, 'shared_reader', _boom)
    parts = SETSIconMatcher.warm_up()
    assert 'icon index' in parts and SETSIconMatcher._index_cache is not None
    assert SETSIconMatcher.is_warming() is False


def test_the_users_crops_are_seeded_only_when_asked(icons, monkeypatch, tmp_path):
    """Only WARP CORE may read the user's confirmed crops."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    calls = []
    monkeypatch.setattr(SETSIconMatcher, '_seed_from_training_data',
                        classmethod(lambda cls, d: calls.append(d) or 0))
    SETSIconMatcher.warm_up()
    assert calls == []
    td = tmp_path / 'td'
    td.mkdir()
    SETSIconMatcher.warm_up(td)
    assert calls == [td]


# ── A recognition started during the warm-up ───────────────────────────────

def _slow_ocr(monkeypatch, started: threading.Event, release: threading.Event):
    """Hold the warm-up in its first step, the OCR load, which takes no lock of
    its own — so only the warm-up's outer hold of the prep lock can make a
    concurrent caller wait there."""
    from warp.recognition import text_extractor as TE

    def _load():
        started.set()
        release.wait(5)
    monkeypatch.setattr(TE, 'shared_reader', _load)


def test_a_recognition_waits_for_the_warm_up_instead_of_repeating_it(icons, monkeypatch):
    from warp.recognition import icon_matcher as IM
    from warp.recognition.icon_matcher import SETSIconMatcher
    builds = []
    real = IM.SETSIconMatcher._build_index_unlocked

    def _counting(self):
        before = SETSIconMatcher._index_cache
        real(self)
        if SETSIconMatcher._index_cache is not before:
            builds.append(1)
    monkeypatch.setattr(IM.SETSIconMatcher, '_build_index_unlocked', _counting)
    started, release = threading.Event(), threading.Event()
    _slow_ocr(monkeypatch, started, release)

    warm = threading.Thread(target=SETSIconMatcher.warm_up)
    done = []
    rec = threading.Thread(target=lambda: done.append(SETSIconMatcher()))
    try:
        warm.start()
        assert started.wait(5)
        assert SETSIconMatcher.is_warming()
        rec.start()
        time.sleep(0.2)
        waited = not done
    finally:
        # Never let a thread outlive the test: it would carry on after the
        # monkeypatches are undone, against the user's real dirs.
        release.set()
        warm.join(10)
        if rec.is_alive() or rec.ident:
            rec.join(10)
    assert waited, 'the recognition must wait on the warm-up'
    assert done and builds == [1], 'the index was built once, by the warm-up'


def test_warp_dropping_the_users_crops_is_not_undone_by_the_warm_up(icons, monkeypatch, tmp_path):
    """The WARP-vs-CORE rule: WARP drops trainer seeds before it matches. A
    drop that arrives before the warm-up has seeded them must still win."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    td = tmp_path / 'td'
    td.mkdir()

    def _seed_td(cls, d):
        SETSIconMatcher.add_session_example(_crop(3), 'Own Item', origin='trainer_td')
        return 1
    monkeypatch.setattr(SETSIconMatcher, '_seed_from_training_data', classmethod(_seed_td))
    started, release = threading.Event(), threading.Event()
    _slow_ocr(monkeypatch, started, release)

    warm = threading.Thread(target=SETSIconMatcher.warm_up, args=(td,))
    drop = threading.Thread(target=lambda: SETSIconMatcher.reset_ml_session(
        keep_origins={'user', 'community'}))
    try:
        warm.start()
        assert started.wait(5)
        drop.start()
        time.sleep(0.2)
    finally:
        release.set()
        warm.join(10)
        if drop.ident:
            drop.join(10)
    assert not any(e.get('origin') == 'trainer_td'
                   for e in SETSIconMatcher._session_examples)


# ── Where it runs ──────────────────────────────────────────────────────────

def test_the_sync_cycle_warms_up_after_the_seed_and_before_done(monkeypatch):
    pytest.importorskip('PySide6')
    from warp.gui import sync_coordinator as SC
    from warp.recognition.icon_matcher import SETSIconMatcher
    order = []
    monkeypatch.setattr('warp.data.cargo.refresh_all', lambda force=False: None)
    monkeypatch.setattr('warp.data.asset_sync.AssetSyncManager.run', lambda self: None)
    monkeypatch.setattr('warp.trainer.model_updater.ModelUpdater._bg_check',
                        lambda self, on_updated=None: None)
    monkeypatch.setattr('warp.knowledge.community_crops.CommunityCropsClient.fetch',
                        lambda self: None)
    monkeypatch.setattr(SETSIconMatcher, 'seed_from_community_crops',
                        classmethod(lambda cls, force=False: order.append('seed-call')))
    monkeypatch.setattr(SETSIconMatcher, 'warm_up',
                        classmethod(lambda cls, td=None: order.append('warm-call')))
    w = SC._RefreshWorker(None, None, None, False)
    w.step.connect(order.append)
    w.run()
    assert order.index('seed') < order.index('warm') < order.index('done')
    assert order.index('seed-call') < order.index('warm-call')
