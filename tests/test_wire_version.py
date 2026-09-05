"""The version recorded against every uploaded crop must be true, and must fit.

Two faults, found while adding the version to the window title.

The value was read from the installed distribution's metadata, which is
written once when a copy is installed. On the maintainer's editable install
that reported 1.0.18.dev1 while the package itself said 1.0.32.dev7 — so
every crop uploaded from that machine was filed against a version eighteen
releases old. An ordinary install has the two agree, which is why nothing
ever showed it.

And the backend declares the field as `max_length=20`. Pydantic refuses the
whole request when it is longer, so an over-long version does not mislabel
one crop — it drops the batch. A hatch-vcs dev build can produce
`1.0.36.dev1+g3a46caf.d20260905`, which is 30 characters.

Run standalone:
    python -m pytest tests/test_wire_version.py -v
"""
from __future__ import annotations

from warp.knowledge.sync_client import (
    _WIRE_VERSION_MAX, _fit_for_wire, _resolve_warp_version,
)


# ── Fitting the field ──────────────────────────────────────────────────────

def test_a_release_version_passes_through():
    assert _fit_for_wire('1.0.36') == '1.0.36'


def test_a_long_local_segment_is_dropped_not_truncated():
    """A truncated version reads as a real one and is not. The local segment
    carries the build noise, so it is what goes."""
    assert _fit_for_wire('1.0.36.dev1+g3a46caf.d20260905') == '1.0.36.dev1'


def test_what_survives_actually_fits_the_backend_field():
    for v in ('1.0.36', '1.0.36.dev1+g3a46caf.d20260905', 'x' * 40):
        assert len(_fit_for_wire(v)) <= _WIRE_VERSION_MAX


def test_an_unshortenable_version_says_so_rather_than_being_cut():
    assert _fit_for_wire('x' * 40) == 'unknown'


def test_an_empty_version_stays_empty():
    """The backend's own default is an empty string, which it accepts."""
    assert _fit_for_wire('') == ''


# ── Which version is reported ──────────────────────────────────────────────

def test_the_imported_package_is_preferred_over_distribution_metadata(monkeypatch):
    """The fault: the two disagree on an editable install, and the metadata
    is the staler of the two."""
    import warp

    monkeypatch.setattr(warp, '__version__', '1.0.32.dev7')
    monkeypatch.setattr('importlib.metadata.version', lambda _: '1.0.18.dev1')

    assert _resolve_warp_version() == '1.0.32.dev7'


def test_a_package_with_no_build_metadata_falls_back(monkeypatch):
    """`0.0.0+unknown` is the placeholder for a checkout the build hook never
    ran on — it is not a version, so the distribution is asked instead."""
    import warp

    monkeypatch.setattr(warp, '__version__', '0.0.0+unknown')
    monkeypatch.setattr('importlib.metadata.version', lambda _: '1.0.36')

    assert _resolve_warp_version() == '1.0.36'


def test_neither_source_available_is_not_an_error(monkeypatch):
    import warp

    monkeypatch.setattr(warp, '__version__', '0.0.0+unknown')

    def _boom(_):
        raise LookupError('no distribution')
    monkeypatch.setattr('importlib.metadata.version', _boom)

    assert _resolve_warp_version() == 'unknown'


def test_no_git_description_reaches_the_wire():
    """`display_version` may carry a commit hash and a dirty marker; neither
    belongs in a 20-character field, and neither describes a build anyone
    else can obtain."""
    from warp.knowledge.sync_client import WARP_VERSION

    assert '-dirty' not in WARP_VERSION
    assert len(WARP_VERSION) <= _WIRE_VERSION_MAX
