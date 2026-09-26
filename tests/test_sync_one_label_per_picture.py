"""One picture is sent under one label, and only once.

The upload cache keys the label it last sent by the picture's hash, and the
server keeps one label per picture per install (last wins). Identical pixels
confirmed in two places — the same inactive BOFF cell under two seats, the
same console as Engineering and as Universal — therefore flipped the cached
label on every sync, and every sync re-sent them as "corrections": measured
2026-09-25, the same 51 corrections at 00:39 and at 19:17.

Offline: the backend POST is stubbed.
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip('numpy')
cv2 = pytest.importorskip('cv2')
pytest.importorskip('PySide6')


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


def _png(path, value):
    img = np.full((40, 30, 3), value, dtype=np.uint8)
    img[5:35, 5:25] = (value, 255 - value, 60)
    cv2.imwrite(str(path), img)
    return str(path)


class _Mgr:
    def __init__(self, store, items):
        self._dir = store
        self._items = items

    def get_confirmed_crops(self):
        return [dict(i) for i in self._items]


def _sync(store, items, monkeypatch):
    """One upload pass; returns what was POSTed as (slot, name) pairs."""
    from warp.trainer import sync
    sent = []
    worker = sync.SyncWorker.__new__(sync.SyncWorker)
    sync.QThread.__init__(worker)
    worker._mgr = _Mgr(store, items)
    worker._url = 'http://127.0.0.1'
    monkeypatch.setattr(worker, '_post', lambda path, payload: (
        sent.extend((i['slot'], i['name']) for i in payload['items'])
        or {'accepted': len(payload['items'])}))
    monkeypatch.setattr(worker, '_fetch_staging_hashes', lambda *a: set())
    monkeypatch.setattr(worker, '_fetch_staging_labels', lambda *a: {})
    worker._upload()
    return sent


def _same_picture_twice(tmp_path, second_name='__inactive__'):
    store = tmp_path / 'store'
    (store / 'crops').mkdir(parents=True)
    a = _png(store / 'crops' / 'a.png', 30)
    b = _png(store / 'crops' / 'b.png', 30)        # identical bytes
    return store, [
        {'path': a, 'slot': 'Boff Tactical', 'name': '__inactive__', 'state': 'confirmed'},
        {'path': b, 'slot': 'Boff Temporal', 'name': second_name, 'state': 'confirmed'},
    ]


def test_a_picture_confirmed_in_two_slots_is_sent_once(tmp_path, monkeypatch):
    store, items = _same_picture_twice(tmp_path)

    assert len(_sync(store, items, monkeypatch)) == 1


def test_the_next_sync_sends_nothing_again(tmp_path, monkeypatch):
    """The reported loop: every sync re-sent the other label."""
    store, items = _same_picture_twice(tmp_path)
    _sync(store, items, monkeypatch)

    assert _sync(store, items, monkeypatch) == []


def test_the_label_chosen_does_not_depend_on_the_order(tmp_path, monkeypatch):
    store, items = _same_picture_twice(tmp_path)
    first = _sync(store, items, monkeypatch)
    for f in store.glob('.sync_*.json'):
        f.unlink()

    assert _sync(store, list(reversed(items)), monkeypatch) == first


def test_two_names_for_one_picture_are_reported(tmp_path, monkeypatch, capfd):
    """Not a slot difference but a disagreement about what the picture is —
    say so, with both names, instead of flipping between them."""
    store, items = _same_picture_twice(tmp_path, second_name='Tractor Beam')
    _sync(store, items, monkeypatch)
    err = capfd.readouterr().err

    assert 'Tractor Beam' in err and '__inactive__' in err


def test_different_pictures_are_all_sent(tmp_path, monkeypatch):
    store = tmp_path / 'store'
    (store / 'crops').mkdir(parents=True)
    items = [
        {'path': _png(store / 'crops' / f'{v}.png', v), 'slot': 'Devices',
         'name': f'Item {v}', 'state': 'confirmed'}
        for v in (10, 90, 170)
    ]

    assert len(_sync(store, items, monkeypatch)) == 3


# ── Tier badges are not refused as too small ───────────────────────────────

def _tier_badge(store, name='ship_tier__t6__k-a1.png'):
    img = np.full((14, 28, 3), 200, dtype=np.uint8)   # the smallest measured badge
    img[3:11, 4:24] = (20, 20, 20)
    cv2.imwrite(str(store / 'crops' / name), img)
    return {'path': str(store / 'crops' / name), 'slot': 'Ship Tier', 'name': 'T6',
            'state': 'confirmed'}


def test_a_short_tier_badge_is_sent(tmp_path, monkeypatch):
    store = tmp_path / 'store'
    (store / 'crops').mkdir(parents=True)

    assert _sync(store, [_tier_badge(store)], monkeypatch) == [('Ship Tier', 'T6')]


def test_a_badge_refused_under_the_old_rule_is_sent_now(tmp_path, monkeypatch):
    """The refusal was remembered in the file cache, so a corrected rule
    would never have reached the badges already refused."""
    store = tmp_path / 'store'
    (store / 'crops').mkdir(parents=True)
    item = _tier_badge(store)
    st = (store / 'crops' / 'ship_tier__t6__k-a1.png').stat()
    (store / '.sync_file_meta.json').write_text(json.dumps({item['path']: {
        'mtime_ns': st.st_mtime_ns, 'size': st.st_size, 'valid': False}}))

    assert _sync(store, [item], monkeypatch) == [('Ship Tier', 'T6')]
