"""Product identity tests — guards PyPI metadata and entry points.

Catches accidental breakage of the package name, console script, or
version machinery before a release reaches PyPI.
"""
from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_package_name():
    with (PROJECT_ROOT / 'pyproject.toml').open('rb') as f:
        meta = tomllib.load(f)
    assert meta['project']['name'] == 'sto-warp'


def test_pyproject_console_script_entry_point():
    with (PROJECT_ROOT / 'pyproject.toml').open('rb') as f:
        meta = tomllib.load(f)
    scripts = meta['project']['scripts']
    assert 'sto-warp' in scripts
    assert scripts['sto-warp'] == 'warp.cli:main'


def test_pyproject_requires_python():
    with (PROJECT_ROOT / 'pyproject.toml').open('rb') as f:
        meta = tomllib.load(f)
    assert meta['project']['requires-python'] == '>=3.14'


def test_version_importable():
    from warp import __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0
    # Must not be the fallback for an installed package.
    # (In editable/dev mode '0.0.0+unknown' is acceptable.)


def test_pyproject_has_dev_dependencies():
    with (PROJECT_ROOT / 'pyproject.toml').open('rb') as f:
        meta = tomllib.load(f)
    dev_deps = meta['project']['optional-dependencies']['dev']
    dep_names = [d.split('>=')[0].split('<')[0].strip().lower() for d in dev_deps]
    assert 'pytest' in dep_names
    assert 'pytest-qt' in dep_names


def test_hatch_vcs_version_source():
    with (PROJECT_ROOT / 'pyproject.toml').open('rb') as f:
        meta = tomllib.load(f)
    assert meta['tool']['hatch']['version']['source'] == 'vcs'
    assert meta['tool']['hatch']['build']['hooks']['vcs']['version-file'] == 'warp/_version.py'
