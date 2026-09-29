"""Every log line reaches the file at once; only warnings and errors are
forced to the disk.

`os.fsync` after every line cost ~3 ms each — 0.8 s of an 8.2 s recognition.
It guards only against the whole system going down; a crash of the program
itself loses nothing once a line is flushed to the OS. So it stays for the
lines that matter after such an event, and goes for the rest.
"""
from __future__ import annotations

import threading

import pytest

import warp.debug as D


@pytest.fixture
def debug(monkeypatch, tmp_path):
    """The shipped `_write`, with the detection channel pointed at a file here
    and `os.fsync` counted instead of performed."""
    path = tmp_path / 'warp_detection.log'
    fh = open(path, 'a', encoding='utf-8')
    monkeypatch.setitem(D._files, 'detection', fh)
    monkeypatch.setitem(D._locks, 'detection', threading.Lock())
    synced = []
    monkeypatch.setattr(D.os, 'fsync', lambda fd: synced.append(fd))
    yield D, synced, path
    fh.close()


def test_an_info_line_is_in_the_file_immediately(debug):
    D, _, path = debug
    D.log.info('first line')
    assert 'first line' in path.read_text()


def test_an_info_line_is_not_forced_to_the_disk(debug):
    D, synced, _ = debug
    D.log.info('routine')
    D.log.debug('detail')
    assert synced == []


@pytest.mark.parametrize('level', ['warning', 'error'])
def test_warnings_and_errors_are_forced_to_the_disk(debug, level):
    D, synced, path = debug
    getattr(D.log, level)('something went wrong')
    assert len(synced) == 1
    assert 'something went wrong' in path.read_text()
