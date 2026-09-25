"""What recognition could not do reaches the window, from the right run.

Two failures that were only in the log on 2026-09-25:

* The OCR read failed ("CUDA error: out of memory" with a game holding the
  card). `scan_image` turned the exception into an empty token list, the
  layout ran without labels, and the screenshot came back with no equipment
  and no word about why.
* Starting recognition again interrupted the running one, and the
  interrupted worker's 'Cancelled' arrived after the new run had started.
  The window took it for the new run: "Recognition cancelled", progress
  stopped, while the new run carried on unseen.

Run standalone:
    python -m pytest tests/test_recognition_failures_shown.py -v
"""
from __future__ import annotations

import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('cv2')


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


# ── The text read ──────────────────────────────────────────────────────────

class _FailingReader:
    def readtext(self, *a, **k):
        raise RuntimeError('CUDA error: out of memory\nSearch for cudaErrorMemoryAllocation')


class _Reader:
    def readtext(self, *a, **k):
        return []


def _extractor(reader):
    from warp.recognition.text_extractor import TextExtractor
    t = TextExtractor()
    t._get_ocr = lambda: reader
    return t


def test_a_failed_read_is_remembered_with_its_reason():
    t = _extractor(_FailingReader())
    t.scan_image(np.zeros((50, 50, 3), dtype=np.uint8))

    assert t.last_read_error == 'CUDA error: out of memory'


def test_a_read_that_works_leaves_no_error_behind():
    first = np.zeros((50, 50, 3), dtype=np.uint8)
    second = np.zeros((60, 60, 3), dtype=np.uint8)   # both alive: scan_image caches by id()
    t = _extractor(_FailingReader())
    t.scan_image(first)
    t._get_ocr = lambda: _Reader()
    t.scan_image(second)

    assert t.last_read_error == ''


def test_the_import_result_says_the_text_could_not_be_read(tmp_path, monkeypatch):
    from warp.data import cargo
    monkeypatch.setattr(cargo, '_fetch', lambda *a, **k: (None, None, None))
    from warp.warp_importer import WarpImporter
    shot = tmp_path / 'shot.png'
    shot.write_bytes(b'\x89PNG\r\n\x1a\n' + b'pretend pixels')

    importer = WarpImporter(build_type='SPACE', from_trainer=False)
    importer._text = _extractor(_FailingReader())
    importer._classify_screen = lambda img: ('SPACE_EQ', 0.96)
    result = importer._process_image(np.zeros((300, 300, 3), dtype=np.uint8), str(shot))

    assert any('could not be read' in e and 'CUDA error: out of memory' in e
               for e in result.errors)


# ── Which run the window listens to ────────────────────────────────────────

class _Any:
    """Stands in for any widget the method touches: every attribute is one
    more of these, and calling it returns one."""

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()

PySide6 = pytest.importorskip('PySide6')


def _window(monkeypatch):
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication, QLabel
    import warp.trainer.trainer_window as tw
    QApplication.instance() or QApplication([])

    class _Worker(QObject):
        progress = Signal(int, str)
        finished = Signal(list)
        error = Signal(str)

        def __init__(self, *a, **k):
            super().__init__()
            self.errors = []

        def isRunning(self):
            return False

        def start(self):
            pass

    monkeypatch.setattr(tw, 'RecognitionWorker', _Worker)

    class _Win:
        _recog_worker = None
        _sets = None

        def __init__(self):
            self._review_summary = QLabel()
            self._recog_warning = QLabel()
            self.errors_seen: list[str] = []
            self.done: list = []

        def _on_recognition_error(self, msg):
            self.errors_seen.append(msg)

        def _on_recognition_done(self, *a, **k):
            self.done.append(a)

        def __getattr__(self, name):
            return _Any()

        _start_recognition = tw.WarpCoreWindow._start_recognition
        _show_recognition_warnings = tw.WarpCoreWindow._show_recognition_warnings

    return _Win()


def test_a_replaced_runs_cancel_does_not_reach_the_window(monkeypatch, tmp_path):
    from pathlib import Path
    w = _window(monkeypatch)
    w._start_recognition(Path('a.png'), 'SPACE_EQ')
    old = w._recog_worker
    w._start_recognition(Path('a.png'), 'SPACE_EQ')
    old.error.emit('Cancelled')

    assert w.errors_seen == []


def test_the_current_runs_signals_still_do(monkeypatch):
    from pathlib import Path
    w = _window(monkeypatch)
    w._start_recognition(Path('a.png'), 'SPACE_EQ')
    w._recog_worker.error.emit('boom')
    w._recog_worker.finished.emit([])

    assert w.errors_seen == ['boom'] and len(w.done) == 1


def test_a_failure_is_shown_and_cleared_on_the_next_run():
    from PySide6.QtWidgets import QApplication, QLabel
    import warp.trainer.trainer_window as tw
    QApplication.instance() or QApplication([])

    class _Win:
        _recog_warning = QLabel()
        show = tw.WarpCoreWindow._show_recognition_warnings

    w = _Win()
    w.show(['Text on this screenshot could not be read (CUDA error: out of memory).'])
    assert 'could not be read' in w._recog_warning.text()
    assert not w._recog_warning.isHidden()
    w.show([])
    assert w._recog_warning.isHidden()
