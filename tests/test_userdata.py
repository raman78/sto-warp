"""Round-trip tests for userdata path resolution and legacy migration.

Every path helper must honour the XDG environment overrides and create
the directory on first call.  The migration logic must be idempotent
and copy legacy files without clobbering existing XDG state.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def xdg_env(monkeypatch, tmp_path):
    """Point all XDG basedirs at isolated temp subdirectories."""
    config = tmp_path / 'config'
    data = tmp_path / 'data'
    cache = tmp_path / 'cache'
    monkeypatch.setenv('XDG_CONFIG_HOME', str(config))
    monkeypatch.setenv('XDG_DATA_HOME', str(data))
    monkeypatch.setenv('XDG_CACHE_HOME', str(cache))
    return config, data, cache


def test_config_dir_honours_xdg_override(xdg_env):
    config, _, _ = xdg_env
    from warp.userdata import config_dir
    result = config_dir()
    assert result == config / 'warp'
    assert result.is_dir()


def test_data_dir_honours_xdg_override(xdg_env):
    _, data, _ = xdg_env
    from warp.userdata import data_dir
    result = data_dir()
    assert result == data / 'warp'
    assert result.is_dir()


def test_cache_dir_honours_xdg_override(xdg_env):
    _, _, cache = xdg_env
    from warp.userdata import cache_dir
    result = cache_dir()
    assert result == cache / 'warp'
    assert result.is_dir()


def test_named_subdirs_exist_after_first_call(xdg_env):
    from warp.userdata import training_data_dir, models_dir
    td = training_data_dir()
    md = models_dir()
    assert td.is_dir()
    assert md.is_dir()


def test_named_files_live_under_correct_basedirs(xdg_env):
    config, data, cache = xdg_env
    from warp import userdata
    assert userdata.install_id_file().parent == config / 'warp'
    assert userdata.backend_config_file().parent == config / 'warp'
    assert userdata.knowledge_cache_file().parent == cache / 'warp'
    assert userdata.rate_limit_file().parent == cache / 'warp'
    assert userdata.recognition_stats_file().parent == data / 'warp'
    assert userdata.contribute_queue_file().parent == data / 'warp'


def test_legacy_migration_is_idempotent(xdg_env, monkeypatch):
    import warp.userdata as ud
    monkeypatch.setattr(ud, '_MIGRATION_DONE', False)
    first = ud.migrate_legacy(force=True)
    monkeypatch.setattr(ud, '_MIGRATION_DONE', False)
    second = ud.migrate_legacy(force=True)
    # Second run must not move anything new.
    assert all(v is False for v in second.values())


def test_purge_legacy_hub_token_removes_file(xdg_env):
    config, _, _ = xdg_env
    from warp.userdata import config_dir, _purge_legacy_hub_token
    token_file = config_dir() / 'hub_token.txt'
    token_file.write_text('fake-token')
    assert _purge_legacy_hub_token() is True
    assert not token_file.exists()
    # Second call is idempotent.
    assert _purge_legacy_hub_token() is False
