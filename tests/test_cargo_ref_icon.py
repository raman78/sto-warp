"""Tests for warp.data.cargo.ref_icon_path trait icon_name fallback.

Trait icons are cached under their `icon_name` (e.g. 'Hive Defenses
(space)'), which differs from the display `name` ('Hive Defenses').
ref_icon_path must fall back to that variant so a confirmed trait's
icon still resolves in tooltips.
"""
from __future__ import annotations

from urllib.parse import quote_plus

import warp.data.cargo as cargo


def _write_icon(icons_dir, name: str):
    (icons_dir / f'{quote_plus(name)}.png').write_bytes(b'\x89PNG\r\n')


def test_ref_icon_path_falls_back_to_trait_icon_name(monkeypatch, tmp_path):
    icons = tmp_path / 'icons'
    icons.mkdir()
    _write_icon(icons, 'Hive Defenses (space)')
    monkeypatch.setenv('WARP_ICONS_DIR', str(icons))

    # Stub the trait source instead of hitting the real cache/network.
    monkeypatch.setattr(cargo, '_load_raw', lambda src: (
        [{'name': 'Hive Defenses', 'icon_name': 'Hive Defenses (space)'}]
        if src == 'traits.json' else []
    ))
    monkeypatch.setattr(cargo, '_BUCKET_MEMO', {})

    p = cargo.ref_icon_path('Hive Defenses')
    assert p is not None
    assert p.name == f'{quote_plus("Hive Defenses (space)")}.png'


def test_ref_icon_path_direct_name_wins(monkeypatch, tmp_path):
    icons = tmp_path / 'icons'
    icons.mkdir()
    _write_icon(icons, 'Accurate')
    monkeypatch.setenv('WARP_ICONS_DIR', str(icons))
    monkeypatch.setattr(cargo, '_load_raw', lambda src: [])
    monkeypatch.setattr(cargo, '_BUCKET_MEMO', {})

    p = cargo.ref_icon_path('Accurate')
    assert p is not None
    assert p.name == 'Accurate.png'


def test_ref_icon_path_unknown_returns_none(monkeypatch, tmp_path):
    icons = tmp_path / 'icons'
    icons.mkdir()
    monkeypatch.setenv('WARP_ICONS_DIR', str(icons))
    monkeypatch.setattr(cargo, '_load_raw', lambda src: [])
    monkeypatch.setattr(cargo, '_BUCKET_MEMO', {})

    assert cargo.ref_icon_path('No Such Item') is None


def _stub_dual_env_trait(monkeypatch):
    """'Adaptive Offense' filed under both env icon_names (ground first)."""
    monkeypatch.setattr(cargo, '_load_raw', lambda src: (
        [{'name': 'Adaptive Offense', 'icon_name': 'Adaptive Offense (ground)'},
         {'name': 'Adaptive Offense', 'icon_name': 'Adaptive Offense (space)'}]
        if src == 'traits.json' else []
    ))
    monkeypatch.setattr(cargo, '_BUCKET_MEMO', {})


def test_ref_icon_path_env_picks_matching_variant(monkeypatch, tmp_path):
    icons = tmp_path / 'icons'
    icons.mkdir()
    _write_icon(icons, 'Adaptive Offense (ground)')
    _write_icon(icons, 'Adaptive Offense (space)')
    monkeypatch.setenv('WARP_ICONS_DIR', str(icons))
    _stub_dual_env_trait(monkeypatch)

    # env='space' must win even though the ground variant is listed first.
    p = cargo.ref_icon_path('Adaptive Offense', env='space')
    assert p is not None
    assert p.name == f'{quote_plus("Adaptive Offense (space)")}.png'

    p = cargo.ref_icon_path('Adaptive Offense', env='ground')
    assert p.name == f'{quote_plus("Adaptive Offense (ground)")}.png'


def test_ref_icon_path_env_falls_back_when_variant_missing(monkeypatch, tmp_path):
    icons = tmp_path / 'icons'
    icons.mkdir()
    _write_icon(icons, 'Adaptive Offense (ground)')  # only ground on disk
    monkeypatch.setenv('WARP_ICONS_DIR', str(icons))
    _stub_dual_env_trait(monkeypatch)

    # Requested space icon isn't cached — fall back to the available variant.
    p = cargo.ref_icon_path('Adaptive Offense', env='space')
    assert p is not None
    assert p.name == f'{quote_plus("Adaptive Offense (ground)")}.png'
