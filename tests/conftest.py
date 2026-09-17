"""Shared pytest fixtures for sto-warp tests."""
from __future__ import annotations

import os
import socket

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# Loopback and local IPC stay open: the offscreen Qt platform, D-Bus and
# anything the runner itself needs speak over those, and none of them is the
# network this guard exists to keep out.
_LOCAL_HOSTS = {'127.0.0.1', '::1', 'localhost', '0.0.0.0'}


def _is_local(address) -> bool:
    if isinstance(address, (str, bytes)):
        return True                      # AF_UNIX path
    try:
        return address[0] in _LOCAL_HOSTS
    except (TypeError, IndexError):
        return False


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Fail the test that opens a connection to the outside world.

    The rule that tests do not touch the network was written down long
    before anything enforced it, and on 2026-09-17 it turned out not to
    hold: `test_standalone_upload.py` ticked the trainer's sync timer,
    which called `WARPSyncClient.refresh_knowledge`, which connected to
    54.246.238.99:443 — and the test passed, because that code catches its
    own failure, logs it and returns. A suite that quietly behaves one way
    with a network and another without it cannot tell anyone what it
    proved, and it is what made the interpreter's shutdown crash come and
    go between runs.

    The guard reports the test and the address, because "a test used the
    network" is not something anyone can act on.

    It watches Python's socket layer, which is where `urllib`, `requests`
    and `httpx` end up. A native extension that opens its own sockets —
    `hf_xet` does — is invisible to it, so a green run is evidence, not
    proof.
    """
    def _refuse(kind):
        def _blocked(self, address, *a, **kw):
            if _is_local(address):
                return kind(self, address, *a, **kw)
            raise AssertionError(
                f'{request.node.nodeid} opened a network connection to '
                f'{address!r}. Tests must stub whatever reaches out — see '
                f'the isolation rules in CLAUDE.md.'
            )
        return _blocked

    monkeypatch.setattr(socket.socket, 'connect', _refuse(socket.socket.connect))
    monkeypatch.setattr(socket.socket, 'connect_ex',
                        _refuse(socket.socket.connect_ex))
