"""Auto-Detect is greyed out while the recognition warm-up runs, and says why.

After the warm-up landed (d346676), pressing Auto-Detect Slots in its first
seconds did nothing visible: the run queued behind the warm-up's lock, while
the tool's own status bar went on saying "Ready." and the launcher's only
progress was a small line of text below it. Now the warm-up reports each
part, both tools show it as a bar next to their Auto-Detect Slots, and the
button stays greyed out until the last report — which comes also when the
warm-up fails, or the button would never come back.

Offscreen Qt; the warm-up itself is stubbed wherever a window starts one.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip('PySide6')

from PySide6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


# ── What the warm-up reports ───────────────────────────────────────────────

@pytest.fixture
def quiet_matcher(monkeypatch):
    """Every warm-up part a no-op, so only the reporting is exercised."""
    from warp.recognition import text_extractor as TE
    from warp.recognition.icon_matcher import SETSIconMatcher as C
    monkeypatch.setattr(TE, 'shared_reader', lambda: None)
    monkeypatch.setattr(C, '__init__', lambda self: None)
    monkeypatch.setattr(C, '_get_ml_session', lambda self: None, raising=False)
    monkeypatch.setattr(C, 'seed_from_training_data', classmethod(lambda cls, d: 0))
    monkeypatch.setattr(C, 'seed_from_community_crops', classmethod(lambda cls, force=False: 0))
    monkeypatch.setattr(C, '_session_stack', classmethod(lambda cls: None))
    return C


def test_each_part_is_reported_before_it_runs_and_the_end_last(quiet_matcher, tmp_path):
    reports = []
    quiet_matcher.warm_up(tmp_path, progress=lambda *r: reports.append(r))
    assert reports == [
        (0, 6, 'ocr'), (1, 6, 'icon index'), (2, 6, 'models'),
        (3, 6, 'own crops'), (4, 6, 'community crops'), (5, 6, 'session stack'),
        (6, 6, ''),
    ]


def test_without_the_users_store_there_are_five_parts(quiet_matcher):
    reports = []
    quiet_matcher.warm_up(progress=lambda *r: reports.append(r))
    assert [r[1] for r in reports] == [5] * 6 and reports[-1] == (5, 5, '')


def test_the_end_is_reported_even_when_the_warm_up_breaks(quiet_matcher, monkeypatch):
    """Otherwise Auto-Detect would stay greyed out for the whole session."""
    import warp.recognition.icon_matcher as IM

    class _Broken:
        def __enter__(self):
            raise RuntimeError('lock gone')

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(IM, '_PREP_LOCK', _Broken())
    reports = []
    with pytest.raises(RuntimeError):
        quiet_matcher.warm_up(progress=lambda *r: reports.append(r))
    assert reports[-1][0] == reports[-1][1]


def test_a_failing_progress_report_does_not_stop_the_warm_up(quiet_matcher):
    def _boom(*r):
        raise RuntimeError('window gone')
    parts = quiet_matcher.warm_up(progress=_boom)
    assert 'session stack' in parts


# ── The bar ────────────────────────────────────────────────────────────────

def test_the_bar_shows_while_preparing_and_hides_at_the_end(qapp):
    from warp.gui.progress_bar import RecognitionPrepBar
    bar = RecognitionPrepBar()
    assert bar.report(1, 5, 'icon index') is True
    assert not bar.isHidden() and 'icon library (2/5)' in bar.format()
    assert bar.report(5, 5, '') is False
    assert bar.isHidden()


# ── WARP ───────────────────────────────────────────────────────────────────

@pytest.fixture
def warp_win(qapp):
    from warp.gui import warp_window as ww
    w = ww.WarpWindow()
    yield w
    w.close()


def test_warp_greys_auto_detect_out_until_the_end(warp_win, tmp_path):
    warp_win._last_folder = tmp_path
    warp_win._set_controls_enabled(True)
    assert warp_win._rerun_btn.isEnabled()
    warp_win.set_recognition_prep(0, 5, 'ocr')
    assert not warp_win._rerun_btn.isEnabled()
    warp_win.set_recognition_prep(5, 5, '')
    assert warp_win._rerun_btn.isEnabled()


def test_warp_does_not_let_a_folder_open_re_enable_it_meanwhile(warp_win, tmp_path):
    """Opening a folder enables Auto-Detect Slots; mid-warm-up it must wait."""
    warp_win.set_recognition_prep(0, 5, 'ocr')
    warp_win._last_folder = tmp_path
    warp_win._set_controls_enabled(True)
    assert not warp_win._rerun_btn.isEnabled()
    warp_win.set_recognition_prep(5, 5, '')
    assert warp_win._rerun_btn.isEnabled()


def test_warp_withdraws_ready_and_puts_it_back(warp_win):
    sb = warp_win.statusBar()
    warp_win.set_recognition_prep(0, 5, 'ocr')
    assert sb.currentMessage() == ''
    warp_win.set_recognition_prep(5, 5, '')
    assert sb.currentMessage() == 'Ready.'


def test_warp_keeps_a_message_that_replaced_ready_meanwhile(warp_win):
    sb = warp_win.statusBar()
    warp_win.set_recognition_prep(0, 5, 'ocr')
    sb.showMessage('Loaded /x — classifying screen types…')
    warp_win.set_recognition_prep(5, 5, '')
    assert sb.currentMessage().startswith('Loaded')


# ── WARP CORE ──────────────────────────────────────────────────────────────

@pytest.fixture
def core_win(qapp):
    from warp.trainer import trainer_window as tw
    w = tw.WarpCoreWindow(embed=True)
    yield w
    w.close()


def test_core_greys_auto_detect_out_until_the_end(core_win):
    core_win._set_auto_detect_enabled(True)
    core_win.set_recognition_prep(2, 6, 'models')
    a = core_win._action_auto_detect
    assert not a.isEnabled() and 'prepared' in a.toolTip()
    core_win.set_recognition_prep(6, 6, '')
    assert a.isEnabled()


def test_core_keeps_a_done_screenshot_locked_after_the_end(core_win):
    """The warm-up ending must not unlock a screenshot marked Done."""
    core_win.set_recognition_prep(0, 6, 'ocr')
    core_win._set_auto_detect_enabled(False, 'Screenshot is marked Done')
    core_win.set_recognition_prep(6, 6, '')
    a = core_win._action_auto_detect
    assert not a.isEnabled() and 'Done' in a.toolTip()


def test_standalone_core_shows_its_own_warm_up(core_win, qapp, monkeypatch):
    from warp.recognition.icon_matcher import SETSIconMatcher
    seen = []
    monkeypatch.setattr(core_win, 'set_recognition_prep',
                        lambda *r: seen.append(r))
    core_win._recognition_prep_report.disconnect()
    core_win._recognition_prep_report.connect(core_win.set_recognition_prep)

    def _warm(cls, td=None, progress=None):
        progress(0, 1, 'ocr')
        progress(1, 1, '')
    monkeypatch.setattr(SETSIconMatcher, 'warm_up', classmethod(_warm))
    core_win._start_recognition_warm_up()
    for t in threading.enumerate():
        if t.name == 'warp-recognition-warm-up':
            t.join(5)
    qapp.processEvents()
    assert seen == [(0, 1, 'ocr'), (1, 1, '')]


# ── The launcher's sync ────────────────────────────────────────────────────

def test_a_cycle_after_the_splash_still_warms_up(qapp, monkeypatch):
    """The cold-start splash runs the sync phases itself and has no warm-up,
    so the launcher only arms the timer afterwards — which left a first run's
    first Auto-Detect doing all the loading itself."""
    from warp.gui import sync_coordinator as SC
    from warp.recognition.icon_matcher import SETSIconMatcher
    ran = threading.Event()
    monkeypatch.setattr(SETSIconMatcher, 'warm_up', classmethod(
        lambda cls, td=None, progress=None: ran.set()))
    coord = SC.SyncCoordinator.__new__(SC.SyncCoordinator)
    SC.QObject.__init__(coord)
    from PySide6.QtCore import QTimer
    coord._timer = QTimer(coord)
    coord.arm_periodic_only()
    coord._timer.stop()
    assert ran.wait(5)


def test_the_sync_cycle_passes_its_progress_on(monkeypatch):
    from warp.gui import sync_coordinator as SC
    from warp.recognition.icon_matcher import SETSIconMatcher
    monkeypatch.setattr('warp.data.cargo.refresh_all', lambda force=False: None)
    monkeypatch.setattr('warp.data.asset_sync.AssetSyncManager.run', lambda self: None)
    monkeypatch.setattr('warp.trainer.model_updater.ModelUpdater._bg_check',
                        lambda self, on_updated=None: None)
    monkeypatch.setattr('warp.knowledge.community_crops.CommunityCropsClient.fetch',
                        lambda self: None)
    monkeypatch.setattr(SETSIconMatcher, 'seed_from_community_crops',
                        classmethod(lambda cls, force=False: 0))
    monkeypatch.setattr(SETSIconMatcher, 'warm_up', classmethod(
        lambda cls, td=None, progress=None: progress(1, 1, '')))
    w = SC._RefreshWorker(None, None, None, False)
    seen = []
    w.recognition_prep.connect(lambda *r: seen.append(r))
    w.run()
    assert seen == [(1, 1, '')]


def test_the_launchers_first_cycle_warms_up_alongside_not_after(qapp, monkeypatch):
    """The cycle's `warm` step came 4-11 s in, after the network steps, and
    the button stayed live until then. The warm-up needs no network."""
    from warp.gui import sync_coordinator as SC
    from warp.recognition.icon_matcher import SETSIconMatcher
    order = []
    ran = threading.Event()

    def _warm(cls, td=None, progress=None):
        order.append('warm-up')
        ran.set()
    monkeypatch.setattr(SETSIconMatcher, 'warm_up', classmethod(_warm))
    coord = SC.SyncCoordinator.__new__(SC.SyncCoordinator)
    SC.QObject.__init__(coord)
    from PySide6.QtCore import QTimer
    coord._timer = QTimer(coord)
    monkeypatch.setattr(coord, 'request_refresh',
                        lambda force=True: order.append('cycle'))
    coord.start()
    coord._timer.stop()
    assert ran.wait(5)
    assert 'cycle' in order
