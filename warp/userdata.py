"""XDG-compliant runtime path resolution + legacy migration.

Single source of truth for *every* place sto-warp persists state at
runtime. Splits files across the three XDG basedirs based on intent:

  - **config_dir** (`$XDG_CONFIG_HOME/warp`, default `~/.config/warp`)
    User-controlled identity / config — `install_id.txt`,
    `backend.json`. Survives `pipx uninstall`, survives reinstalls,
    never deleted by `pip`. Pre-1.0.5 installs also stored
    `hub_token.txt` here; that file is purged on first run after upgrade
    because the HF write token now lives only as a server-side secret.

  - **data_dir**   (`$XDG_DATA_HOME/warp`,   default `~/.local/share/warp`)
    Persistent user-generated state we cannot regenerate — annotations,
    confirmed crops, learned anchors, training screenshots, sync logs.

  - **cache_dir**  (`$XDG_CACHE_HOME/warp`,  default `~/.cache/warp`)
    Disposable downloads / derived files — community knowledge.json,
    rate-limit counters, model weights downloaded from HF, model
    version cache.

Pre-XDG installs kept everything next to the source tree
(`<repo>/warp/training_data/`, `<repo>/warp/models/`,
`<repo>/warp/knowledge/*.json`). On first call to any path helper we
copy missing files from those legacy locations so an existing user's
training data survives the move. The migration is one-shot and
idempotent.

Pipx + AUR installs land in read-only site-packages, so attempting to
write next to the source would either fail outright or silently land
in some unrelated cwd. Routing everything through this module keeps
those installs working.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path

try:
    from warp.debug import syslog as log
except Exception:
    log = logging.getLogger(__name__)


# ── XDG basedirs ───────────────────────────────────────────────────────────

def _xdg(env_var: str, default_rel: str) -> Path:
    raw = os.environ.get(env_var)
    if raw:
        p = Path(raw).expanduser()
    else:
        p = Path.home() / default_rel
    return p / 'warp'


def config_dir() -> Path:
    p = _xdg('XDG_CONFIG_HOME', '.config')
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir() -> Path:
    p = _xdg('XDG_DATA_HOME', '.local/share')
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = _xdg('XDG_CACHE_HOME', '.cache')
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Named subdirs / files ──────────────────────────────────────────────────

def training_data_dir() -> Path:
    p = data_dir() / 'training_data'
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = cache_dir() / 'models'
    p.mkdir(parents=True, exist_ok=True)
    return p


def install_id_file() -> Path:
    """Per-install UUID for deduplication on the WARP backend."""
    return config_dir() / 'install_id.txt'


def backend_config_file() -> Path:
    """Optional override of the WARP knowledge backend URL."""
    return config_dir() / 'backend.json'


def knowledge_cache_file() -> Path:
    return cache_dir() / 'knowledge_cache.json'


def icon_equivalence_cache_file() -> Path:
    """Curated icon-equivalence classes mirrored from HF.

    Maintainer-curated JSON (see ``warp.tools.icon_equivalence``) that
    lists groups of STO item names sharing identical icon art. The
    trainer reads this to suppress community-conflict prompts between
    items whose icons are visually indistinguishable — otherwise the
    user would be asked to disambiguate the same identical-icon pair on
    every restart and never get a stable answer.
    """
    return cache_dir() / 'icon_equivalence.json'


def rate_limit_file() -> Path:
    return cache_dir() / 'contribute_rate_limit.json'


def recognition_stats_file() -> Path:
    """Per-image icon-match stats sink used by WarpImporter."""
    return data_dir() / 'recognition_stats.json'


def contribute_queue_file() -> Path:
    """Pending WARPSync contributions persisted across app restarts."""
    return data_dir() / 'contribute_queue.jsonl'


def screen_type_stats_file() -> Path:
    """Validation-accuracy stats kept by the trainer."""
    return data_dir() / 'screen_type_stats.json'


# ── One-shot migration from legacy in-repo layout ─────────────────────────

_MIGRATION_LOCK = threading.Lock()
_MIGRATION_DONE = False


def _legacy_repo_root() -> Path | None:
    """Walk up from this file looking for a development checkout.

    Returns the directory containing `pyproject.toml` (and therefore the
    original `warp/` source tree), or None when running from an installed
    wheel where the package has been copied into site-packages.
    """
    p = Path(__file__).resolve().parent
    for _ in range(6):
        if (p / 'pyproject.toml').is_file():
            return p
        p = p.parent
    return None


def _purge_legacy_hub_token() -> bool:
    """Delete the shared HF write token left over from pre-1.0.5 installs.

    Every sto-warp install used to ship with the same write-scoped HF
    token in `~/.config/warp/hub_token.txt`, which meant a single leaked
    file could compromise the entire `sets-sto` dataset namespace.
    Starting with 1.0.5 the token lives only as a server-side secret
    in the backend (HF Space `sets-sto/warp-backend`); the on-disk file
    is no longer read by any code path, so we delete it on first run.

    Idempotent: returns True only the first time it actually removes a
    file, False otherwise (already gone, never existed, or unreadable
    filesystem). A removal failure is logged but never raised — losing
    the deletion is not worth crashing the app.
    """
    legacy_token = config_dir() / 'hub_token.txt'
    if not legacy_token.exists():
        return False
    try:
        legacy_token.unlink()
        log.info(
            f'userdata: removed legacy {legacy_token} — HF write token '
            f'now lives only on the backend (sets-sto/warp-backend Space)'
        )
        return True
    except Exception as e:
        log.warning(f'userdata: failed to purge legacy hub_token.txt: {e}')
        return False


def _copy_if_missing(src: Path, dst: Path) -> bool:
    """Copy file or directory tree from `src` to `dst` unless `dst` already
    exists. Returns True if a copy was actually performed."""
    if dst.exists():
        return False
    if not src.exists():
        return False
    try:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return True
    except Exception as e:
        log.warning(f'userdata: legacy copy {src} → {dst} failed: {e}')
        return False


def migrate_legacy(force: bool = False) -> dict[str, bool]:
    """Idempotent one-shot migration of in-repo data to XDG dirs.

    Safe to call repeatedly; only the first invocation per process does
    any work. Returns a dict mapping each migration entry to whether it
    actually moved anything this call.
    """
    global _MIGRATION_DONE
    with _MIGRATION_LOCK:
        if _MIGRATION_DONE and not force:
            return {}
        _MIGRATION_DONE = True

        repo = _legacy_repo_root()
        if repo is None:
            return {}

        legacy = repo / 'warp'
        moved: dict[str, bool] = {}

        # CONFIG — identity / runtime override
        moved['install_id'] = _copy_if_missing(legacy / 'knowledge' / 'install_id.txt',
                                               install_id_file())
        moved['backend']    = _copy_if_missing(legacy / 'knowledge' / 'config.json',
                                               backend_config_file())
        moved['hub_token_purged'] = _purge_legacy_hub_token()

        # CACHE — downloads
        moved['knowledge_cache'] = _copy_if_missing(
            legacy / 'knowledge' / 'knowledge_cache.json',
            knowledge_cache_file())
        moved['rate_limit'] = _copy_if_missing(
            legacy / 'knowledge' / 'rate_limit.json',
            rate_limit_file())
        moved['models'] = _copy_if_missing(legacy / 'models', models_dir().parent /
                                            models_dir().name) \
            if not models_dir().exists() else False
        # The previous line works around mkdir() being called eagerly in
        # `models_dir()`. Use a direct copytree when the target is still
        # essentially empty (no model files present yet).
        try:
            if not any(models_dir().iterdir()):
                src_models = legacy / 'models'
                if src_models.is_dir():
                    for f in src_models.iterdir():
                        _copy_if_missing(f, models_dir() / f.name)
                    moved['models'] = True
        except Exception as e:
            log.debug(f'userdata: model migration scan failed: {e}')

        # DATA — user state
        try:
            if not any(training_data_dir().iterdir()):
                src_td = legacy / 'training_data'
                if src_td.is_dir():
                    for f in src_td.iterdir():
                        _copy_if_missing(f, training_data_dir() / f.name)
                    moved['training_data'] = True
        except Exception as e:
            log.debug(f'userdata: training data migration scan failed: {e}')

        applied = [k for k, v in moved.items() if v]
        if applied:
            log.info(f'userdata: migrated legacy files → XDG ({", ".join(applied)})')

        return moved


def ensure_migrated() -> None:
    """Cheap idempotent wrapper — every userdata caller can invoke at will."""
    migrate_legacy(force=False)
