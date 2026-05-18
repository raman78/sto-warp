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

import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable


CHANNELS = ('detection', 'system')


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


def _write(channel: str, level: str, msg: str) -> None:
    ts  = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    tid = threading.current_thread().name
    line = f'{ts}  [{level}]  [{tid}]  {msg}'
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


class _Log:
    """Channel-bound logger facade. `log.info(...)` writes to one file."""

    def __init__(self, channel: str):
        self._ch = channel

    def info(self, msg):    _write(self._ch, 'INFO ', str(msg))
    def debug(self, msg):   _write(self._ch, 'DEBUG', str(msg))
    def warning(self, msg): _write(self._ch, 'WARN ', str(msg))
    def error(self, msg):   _write(self._ch, 'ERROR', str(msg))


# Default channel: detection. Use `from warp.debug import log`.
log    = _Log('detection')
# System channel: sync, model updates, infrastructure.
# Use `from warp.debug import syslog`.
syslog = _Log('system')


for _ch, (_cur, _bak) in _paths.items():
    _Log(_ch).info(f'=== warp.debug initialized  pid={os.getpid()}  log={_cur} ===')
