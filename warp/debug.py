"""
Standalone debug logger for sto-warp.

Replaces upstream `src.setsdebug` import surface. Records are split into
two channels written to separate rotated files under `$WARP_LOG_DIR`:

  - `warp_detection.log` — recognition, OCR, layout, importer, trainer.
    Imported as the bare `log` symbol (default).
  - `warp_system.log`    — asset sync, model updater, knowledge client,
    desktop integration, XDG migration. Imported as `syslog`.

Each channel keeps its own `.bak` rotation across restarts. Subscribers
receive `(channel, level, line)` so the launcher's two Logs tabs can
filter on the channel they belong to.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable


# 'detection'      — WARP recognition pipeline (default)
# 'detection_core' — WARP CORE trainer detection / classification
# 'system'         — sync, model updates, infrastructure
#
# WARP and WARP CORE share the same underlying recognition code, but each
# tool's GUI shows its own Detection Logs tab. To keep the two tabs from
# polluting each other we route writes from `log.*` through a thread-local
# channel override (`use_detection_channel`) so each tool's worker thread
# writes to its own file/subscription stream.
CHANNELS = ('detection', 'detection_core', 'system')


def _default_log_dir() -> Path:
    env = os.environ.get('WARP_LOG_DIR')
    if env:
        return Path(env)
    xdg = os.environ.get('XDG_CONFIG_HOME')
    if xdg:
        return Path(xdg) / 'warp'
    return Path.home() / '.config' / 'warp'


_log_dir = _default_log_dir()
try:
    _log_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    _log_dir = Path.home()


_paths: dict[str, tuple[Path, Path]] = {}
_files: dict[str, object | None]    = {}
_locks: dict[str, threading.Lock]   = {}

for _ch in CHANNELS:
    _cur = _log_dir / f'warp_{_ch}.log'
    _bak = _cur.with_suffix('.log.bak')
    _paths[_ch] = (_cur, _bak)
    _locks[_ch] = threading.Lock()
    try:
        if _cur.exists():
            try:
                if _bak.exists():
                    _bak.unlink()
                _cur.rename(_bak)
            except Exception:
                pass
        _files[_ch] = open(_cur, 'w', buffering=1)
    except Exception as e:
        _files[_ch] = None
        print(f'[WARP-LOG] WARNING: cannot open {_cur}: {e}', flush=True)


# Subscribers receive (channel, level, formatted_line) for every record.
# Used by the launcher's two Logs tabs; intentionally Qt-free so non-GUI
# callers (CLI, tests) don't pay for it.
_subscribers: list[Callable[[str, str, str], None]] = []


def subscribe(cb: Callable[[str, str, str], None]) -> None:
    """Register `cb(channel, level, line)` for live log records. Idempotent."""
    if cb not in _subscribers:
        _subscribers.append(cb)


def unsubscribe(cb: Callable[[str, str, str], None]) -> None:
    try:
        _subscribers.remove(cb)
    except ValueError:
        pass


def log_paths(channel: str = 'detection') -> tuple[Path, Path]:
    """Return (current_session_log, previous_session_log) for `channel`."""
    return _paths.get(channel, _paths['detection'])


# ── Thread-local detection-channel routing ────────────────────────────────
# `log` (channel='detection') resolves the active channel on every write
# via this thread-local, so a worker thread can opt into 'detection_core'
# without every call site needing to know about it. Main-thread default
# can be set via `set_main_detection_channel` (e.g. tied to the active
# launcher tab).

_thread_local = threading.local()
_main_default = 'detection'


def set_main_detection_channel(name: str) -> None:
    """Sticky channel for any thread that hasn't pushed its own override.
    Intended for the launcher: switch when the user moves between tabs."""
    global _main_default
    if name not in CHANNELS:
        raise ValueError(f'unknown channel {name!r}')
    _main_default = name


def _resolve_detection_channel() -> str:
    return getattr(_thread_local, 'detection_channel', _main_default)


@contextlib.contextmanager
def use_detection_channel(name: str):
    """Route `log.*` writes on the current thread to `name` until exit.
    No-op for non-detection loggers (syslog is unaffected)."""
    if name not in CHANNELS:
        raise ValueError(f'unknown channel {name!r}')
    prev = getattr(_thread_local, 'detection_channel', None)
    _thread_local.detection_channel = name
    try:
        yield
    finally:
        if prev is None:
            try:
                del _thread_local.detection_channel
            except AttributeError:
                pass
        else:
            _thread_local.detection_channel = prev


def _write(channel: str, level: str, msg: str) -> None:
    ts  = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    line = f'{ts}  [{level}]  {msg}'
    if level != 'DEBUG':
        print(f'[WARP/{channel[:3]}] {line}', file=sys.stderr, flush=True)
    fh = _files.get(channel)
    if fh is not None:
        with _locks[channel]:
            try:
                fh.write(line + '\n')
                fh.flush()
                os.fsync(fh.fileno())
            except Exception:
                pass
    if _subscribers:
        for cb in list(_subscribers):
            try:
                cb(channel, level.strip(), line)
            except Exception:
                pass


def clear_logs(channel: str = 'detection') -> None:
    """Emit a special CLEAR signal to UI log viewers for the given channel."""
    if _subscribers:
        for cb in list(_subscribers):
            try:
                cb(channel, 'CLEAR', '')
            except Exception:
                pass


class _Log:
    """Channel-bound logger facade. `log.info(...)` writes to one file.

    Special case: when constructed with channel='detection', writes go
    to whichever detection channel the thread (or launcher tab) has
    selected — see `use_detection_channel` and
    `set_main_detection_channel`. All other channels are static."""

    def __init__(self, channel: str):
        self._ch = channel
        self._dynamic = (channel == 'detection')

    def _resolve(self) -> str:
        return _resolve_detection_channel() if self._dynamic else self._ch

    def info(self, msg):    _write(self._resolve(), 'INFO ', str(msg))
    def debug(self, msg):   _write(self._resolve(), 'DEBUG', str(msg))
    def warning(self, msg): _write(self._resolve(), 'WARN ', str(msg))
    def error(self, msg):   _write(self._resolve(), 'ERROR', str(msg))


# Default channel: detection. Use `from warp.debug import log`.
log    = _Log('detection')
# System channel: sync, model updates, infrastructure.
# Use `from warp.debug import syslog`.
syslog = _Log('system')


for _ch, (_cur, _bak) in _paths.items():
    _Log(_ch).info(f'=== warp.debug initialized  pid={os.getpid()}  log={_cur} ===')
