"""The tarball builder must settle the token question before the checkout.

Every read it performs works anonymously — the dataset is public — so a token
that cannot write stays invisible until `upload_file` fails at the very end.
Both halves of that were observed in September 2026: a revoked token 401'd
after ten minutes (2026-09-07, 2026-09-14), and a read-only one got past
`whoami` and 403'd after four (2026-09-16, twice).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

pytest.importorskip('huggingface_hub')

_SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'build_crops_tarball.py'


def _load():
    spec = importlib.util.spec_from_file_location('build_crops_tarball', _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Stop(Exception):
    """Raised by the stub once main() has got past every token check."""


def _stub_hub(monkeypatch, mod, *, whoami, write_probe):
    """Stub the Hub: `dataset_info` marks 'got past the token checks'.

    `write_probe` stands in for the Xet write-credential request — return a
    value to grant it, raise to refuse it.
    """
    calls: list[str] = []

    class _Api:
        def __init__(self, token=None):
            pass

        def whoami(self):
            calls.append('whoami')
            return whoami()

        def dataset_info(self, *a, **kw):
            calls.append('dataset_info')
            raise _Stop

    def _urlopen(req, timeout=None):
        calls.append('write_probe')
        return write_probe()

    monkeypatch.setattr(mod, 'HfApi', _Api)
    monkeypatch.setattr(mod, 'urlopen', _urlopen)
    return calls


def _http_error(code: int) -> HTTPError:
    return HTTPError('https://huggingface.co/x', code, 'refused', {}, None)


def test_revoked_token_stops_before_the_dataset(monkeypatch):
    mod = _load()
    monkeypatch.setenv('HF_TOKEN', 'hf_revoked')

    def _reject():
        raise RuntimeError('401 Client Error: Invalid username or password.')

    calls = _stub_hub(monkeypatch, mod, whoami=_reject, write_probe=lambda: None)

    assert mod.main() == 2
    assert calls == ['whoami']


def test_read_only_token_stops_before_the_checkout(monkeypatch):
    mod = _load()
    monkeypatch.setenv('HF_TOKEN', 'hf_readonly')

    def _refuse():
        raise _http_error(403)

    calls = _stub_hub(monkeypatch, mod, whoami=lambda: {'name': 'someone'},
                      write_probe=_refuse)

    assert mod.main() == 2
    assert calls == ['whoami', 'write_probe']


def test_write_token_proceeds_to_the_dataset(monkeypatch):
    mod = _load()
    monkeypatch.setenv('HF_TOKEN', 'hf_write')

    calls = _stub_hub(monkeypatch, mod, whoami=lambda: {'name': 'someone'},
                      write_probe=lambda: object())

    with pytest.raises(_Stop):
        mod.main()
    assert calls == ['whoami', 'write_probe', 'dataset_info']


def test_an_unreadable_probe_never_blocks_the_build(monkeypatch):
    """Only an explicit 403 refuses; a moved or broken endpoint must not."""
    mod = _load()
    monkeypatch.setenv('HF_TOKEN', 'hf_write')

    def _break():
        raise _http_error(404)

    calls = _stub_hub(monkeypatch, mod, whoami=lambda: {'name': 'someone'},
                      write_probe=_break)

    with pytest.raises(_Stop):
        mod.main()
    assert calls == ['whoami', 'write_probe', 'dataset_info']


def test_an_unchanged_tarball_is_not_published_again(tmp_path, monkeypatch):
    """The second build of identical crops must not commit the same bytes.

    The published manifest handed to the second run is the one the first run
    produced, so nothing here recomputes what the script computes.
    """
    mod = _load()
    monkeypatch.setenv('HF_TOKEN', 'hf_write')

    uploads: list[str] = []
    built_manifest: dict[str, str] = {}

    class _Api:
        def __init__(self, token=None):
            pass

        def whoami(self):
            return {'name': 'someone'}

        def dataset_info(self, *a, **kw):
            return SimpleNamespace(sha='c0ffee' * 7)

        def upload_file(self, *, path_or_fileobj, path_in_repo, **kw):
            uploads.append(path_in_repo)
            if path_in_repo == mod.MANIFEST_FILE:
                built_manifest['json'] = Path(path_or_fileobj).read_text()

    def _no_manifest(**kw):
        raise FileNotFoundError('404')

    monkeypatch.setattr(mod, 'HfApi', _Api)
    monkeypatch.setattr(mod, 'urlopen', lambda *a, **kw: object())
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(mod, 'hf_hub_download', _no_manifest)

    assert mod.main() == 0
    assert uploads == [mod.TARBALL_FILE, mod.MANIFEST_FILE]

    published = tmp_path / 'crops_manifest.json'
    published.write_text(built_manifest['json'])
    monkeypatch.setattr(mod, 'hf_hub_download', lambda **kw: str(published))
    uploads.clear()

    assert mod.main() == 0
    assert uploads == []


def test_absent_token_never_reaches_the_hub(monkeypatch):
    mod = _load()
    monkeypatch.delenv('HF_TOKEN', raising=False)

    calls = _stub_hub(monkeypatch, mod, whoami=lambda: {'name': 'someone'},
                      write_probe=lambda: object())

    assert mod.main() == 2
    assert calls == []
