"""
Standalone debug logger for sto-warp.

Replaces upstream `src.setsdebug` import surface. Writes one rotated
log file to `$WARP_LOG_DIR/warp_debug.log` (defaults to the platform
user-config dir) and mirrors INFO/WARN/ERROR to stderr.

Public API: `log.info / debug / warning / error`.
"""
from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable


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

_log_path = _log_dir / 'warp_debug.log'
_lock = threading.Lock()

try:
    _bak_path = _log_path.with_suffix('.log.bak')
    if _log_path.exists():
        try:
            if _bak_path.exists():
                _bak_path.unlink()
            _log_path.rename(_bak_path)
        except Exception:
            pass
    _fh = open(_log_path, 'w', buffering=1)
    _file_ok = True
except Exception as e:
    _fh = None
    _file_ok = False
    print(f'[WARP-LOG] WARNING: cannot open log file {_log_path}: {e}', flush=True)


# GUI tap — subscribers receive (level, formatted_line) for every record.
# Used by the launcher's Logs tab; intentionally Qt-free so non-GUI callers
# (CLI, tests) don't pay for it.
_subscribers: list[Callable[[str, str], None]] = []


def subscribe(cb: Callable[[str, str], None]) -> None:
    """Register `cb(level, line)` for live log records. Idempotent."""
    if cb not in _subscribers:
        _subscribers.append(cb)


def unsubscribe(cb: Callable[[str, str], None]) -> None:
    try:
        _subscribers.remove(cb)
    except ValueError:
        pass


def log_paths() -> tuple[Path, Path]:
    """Return (current_session_log, previous_session_log) paths."""
    return _log_path, _bak_path


def _write(level: str, msg: str) -> None:
    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    tid = threading.current_thread().name
    line = f'{ts}  [{level}]  [{tid}]  {msg}'
    if level != 'DEBUG':
        print(f'[WARP] {line}', file=sys.stderr, flush=True)
    if _file_ok and _fh:
        with _lock:
            try:
                _fh.write(line + '\n')
                _fh.flush()
                os.fsync(_fh.fileno())
            except Exception:
                pass
    if _subscribers:
        for cb in list(_subscribers):
            try:
                cb(level.strip(), line)
            except Exception:
                pass


class _Log:
    def info(self, msg):    _write('INFO ', str(msg))
    def debug(self, msg):   _write('DEBUG', str(msg))
    def warning(self, msg): _write('WARN ', str(msg))
    def error(self, msg):   _write('ERROR', str(msg))


log = _Log()
log.info(f'=== warp.debug initialized  pid={os.getpid()}  log={_log_path} ===')
