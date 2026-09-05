"""A confirmation records who made it: the person, or the detector.

`auto_confirmed` is not a colour. It is what keeps the detector's own answers
out of the session-example seed — `SETSIconMatcher.seed_from_community_crops`
skips them so today's high-confidence match cannot become tomorrow's perfect
self-match — and out of what `WarpImporter._user_confirmed` reads as ground
truth.

Three internal callers reach `_on_accept` after the detector cleared the
auto-accept threshold, and until 2026-09-05 all three recorded the result as
though the user had pressed Enter. Reported from a real session: correcting a
seat to Intel re-matched the icon and marked it confirmed-by-user, which it
was not.

Run standalone:
    python -m pytest tests/test_accept_provenance.py -v
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip('PySide6')

from warp.trainer.trainer_window import WarpCoreWindow


# ── The signature ──────────────────────────────────────────────────────────

def test_accept_takes_who_decided():
    sig = inspect.signature(WarpCoreWindow._on_accept)

    assert 'auto' in sig.parameters
    assert sig.parameters['auto'].default is False


def test_a_signal_cannot_fill_it_positionally():
    """`clicked` hands a slot a `checked` bool. If `auto` were positional,
    clicking Accept would file the row as a detector answer."""
    sig = inspect.signature(WarpCoreWindow._on_accept)

    assert sig.parameters['auto'].kind is inspect.Parameter.KEYWORD_ONLY
    assert any(p.kind is inspect.Parameter.VAR_POSITIONAL
               for p in sig.parameters.values())


def test_qt_style_positional_call_leaves_it_a_user_decision():
    """The behaviour the previous test protects, driven rather than inspected.
    Binds the real function to a stub that records what it was told."""
    seen = {}

    class _Stub:
        def _is_current_locked(self):
            seen['auto'] = self._captured_auto
            return True                      # bail immediately after the guard

    def _capture(self, *args, auto=False):
        self._captured_auto = auto
        return WarpCoreWindow._on_accept(self, *args, auto=auto)

    stub = _Stub()
    stub.statusBar = lambda: type('B', (), {'showMessage': lambda *a: None})()
    _capture(stub, False)                    # exactly what `clicked` sends

    assert seen['auto'] is False


# ── The call sites ─────────────────────────────────────────────────────────

def _calls_in(fn_name: str) -> list[str]:
    src = inspect.getsource(getattr(WarpCoreWindow, fn_name))
    return [ln.strip() for ln in src.splitlines() if '_on_accept(' in ln]


@pytest.mark.parametrize('fn', [
    '_on_ocr_finished', '_rematch_current_item', '_rematch_with_slot',
])
def test_the_threshold_driven_callers_say_it_was_automatic(fn):
    """All three are gated on the auto-accept checkbox and its threshold —
    they are the detector accepting its own answer."""
    calls = _calls_in(fn)

    assert calls, f'{fn} no longer calls _on_accept'
    assert all('auto=True' in c for c in calls), calls


@pytest.mark.parametrize('fn', ['_on_enter', '_on_completer_activated'])
def test_the_user_driven_callers_stay_user_decisions(fn):
    """Enter, and picking from the autocomplete list. Marking these automatic
    would throw away every real confirmation the seed depends on."""
    calls = _calls_in(fn)

    assert calls, f'{fn} no longer calls _on_accept'
    assert all('auto=True' not in c for c in calls), calls


def test_the_flag_written_follows_the_argument():
    """The call sites are only half of it: the body has to key the recorded
    flag on `auto` rather than hardcoding it. Reverting that line broke no
    test until this one existed."""
    src = inspect.getsource(WarpCoreWindow._on_accept)

    assert "ri['auto_confirmed'] = auto" in src
    assert 'auto_confirmed=auto' in src
    assert 'auto_confirmed=False' not in src
