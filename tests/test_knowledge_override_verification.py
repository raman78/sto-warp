"""A community pHash hit is used only when the picture matches too.

The knowledge table maps a 64-bit perceptual hash to the name the community
voted for. The hash is weak — a median of 12 bits set out of 64 — so two
different pictures can share one. A hit used to be a hard override at 1.00,
checked only for "is it a virtual class?". On the screenshot that exposed
this, an Omni-Directional Pahvan beam array read as `Phaser Turret` and a
deflector as the wrong deflector, both at 1.00.

A hit now says "the community voted *a* picture with this hash to be X". It
is used when the crop also resembles the gallery's pictures of X, which are
the crops the community confirmed and the wiki art.

Run standalone:
    python -m pytest tests/test_knowledge_override_verification.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('cv2')
pytest.importorskip('PySide6')


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


# Distinct noise, so the crop has a real hash and real structure.
CROP = np.random.default_rng(3).integers(0, 256, (44, 35, 3), dtype=np.uint8)


class _Knowledge:
    """The community table, holding one entry: this crop's hash → `name`."""

    def __init__(self, name):
        import cv2
        from warp.knowledge.sync_client import _compute_phash
        from warp.recognition.icon_matcher import MATCH_SIZE
        crop64 = cv2.resize(CROP, (MATCH_SIZE, MATCH_SIZE),
                            interpolation=cv2.INTER_AREA)
        self._table = {_compute_phash(crop64): name}

    def get_knowledge(self):
        return dict(self._table)


class _StubEmbedder:
    """Embeds every query as the vector the test chose."""

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def __call__(self, _tensor):
        import torch
        v = self.vector / np.linalg.norm(self.vector)
        return torch.from_numpy(v[None, :])


def _matcher(query, gallery: dict[str, list] | None, voted='Phaser Turret'):
    """`gallery` maps an item name to its picture vector; None = no embedder."""
    from warp.recognition.icon_matcher import SETSIconMatcher

    m = SETSIconMatcher(sync_client=_Knowledge(voted))
    m._index = []                      # no wiki templates in play
    if gallery is None:
        m._ml_disabled = True
        return m
    names = list(gallery)
    emb = np.asarray([gallery[n] for n in names], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    m._gallery_emb = emb
    m._gallery_lbl = np.arange(len(names), dtype=np.int32)
    m._gallery_is_art = np.zeros(len(names), dtype=bool)
    m._label_map = dict(enumerate(names))
    m._ml_kind = 'embedder'
    m._ml_session = _StubEmbedder(query)
    return m


SLOT = {'Phaser Turret', 'Omni-Directional Pahvan Proton Beam Array'}
TURRET = [1.0, 0.0, 0.0]
PAHVAN = [0.0, 1.0, 0.0]


def test_a_hit_on_a_picture_that_looks_like_it_is_used():
    m = _matcher(query=TURRET, gallery={'Phaser Turret': TURRET,
                                        'Omni-Directional Pahvan Proton Beam Array': PAHVAN})
    name, conf, _, _ = m.match(CROP, candidate_names=SLOT)

    assert (name, conf) == ('Phaser Turret', 1.0)
    assert m._last_match_src == 'knowledge'


def test_a_hit_on_a_picture_that_does_not_is_a_collision():
    """The reported case: the hash said Phaser Turret, the picture is the
    Pahvan array. The community voted on another picture."""
    m = _matcher(query=PAHVAN, gallery={'Phaser Turret': TURRET,
                                        'Omni-Directional Pahvan Proton Beam Array': PAHVAN})
    name, _, _, _ = m.match(CROP, candidate_names=SLOT)

    assert m._last_match_src != 'knowledge'
    assert name == 'Omni-Directional Pahvan Proton Beam Array'


def test_a_hit_naming_an_item_with_no_pictures_is_not_trusted():
    """Nothing to compare with means nothing confirms the claim — measured
    once in 357 wrong hits, never among the correct ones."""
    m = _matcher(query=PAHVAN,
                 gallery={'Omni-Directional Pahvan Proton Beam Array': PAHVAN})
    m.match(CROP, candidate_names=SLOT)

    assert m._last_match_src != 'knowledge'


def test_without_an_embedder_the_hit_is_offered_below_auto_accept():
    """No way to check the picture: keep the community's name, but not at a
    certainty WARP CORE would accept without a person."""
    from warp.recognition.icon_matcher import KNOWLEDGE_UNVERIFIED_CONF
    m = _matcher(query=TURRET, gallery=None)
    name, conf, _, _ = m.match(CROP, candidate_names=SLOT)

    assert name == 'Phaser Turret'
    assert conf == KNOWLEDGE_UNVERIFIED_CONF < 0.75


def test_checking_the_picture_costs_no_extra_forward_pass():
    """The knowledge cross-check already embeds the crop on every hit; the
    picture check reads those similarities. A second pass was ~45 ms per
    hit — seconds on a mixed screenshot."""
    m = _matcher(query=TURRET, gallery={'Phaser Turret': TURRET})
    calls = []
    real = m._ml_session
    m._ml_session = lambda t: (calls.append(1), real(t))[1]
    m.match(CROP, candidate_names=SLOT)

    assert len(calls) == 1


# ── One hash, several names ────────────────────────────────────────────────

class _Tally(_Knowledge):
    """The table leads with `lead`; the tally also names the others."""

    def __init__(self, lead, votes):
        super().__init__(lead)
        self._votes = votes

    def get_knowledge_votes(self, phash):
        return dict(self._votes) if phash in self._table else {}


def _tally_matcher(query, votes, lead):
    m = _matcher(query, {'Phaser Turret': TURRET,
                         'Omni-Directional Pahvan Proton Beam Array': PAHVAN})
    m._sync_client = _Tally(lead, votes)
    return m


def test_the_name_whose_pictures_match_is_chosen_not_the_leader():
    """Measured: one live hash carried votes for five different items. The
    table's leader is right only for its own picture."""
    m = _tally_matcher(PAHVAN, lead='Phaser Turret',
                       votes={'Phaser Turret': 30,
                              'Omni-Directional Pahvan Proton Beam Array': 2})
    name, conf, _, _ = m.match(CROP, candidate_names=SLOT)

    assert (name, conf) == ('Omni-Directional Pahvan Proton Beam Array', 1.0)
    assert m._last_match_src == 'knowledge'


def test_without_an_embedder_the_most_voted_name_is_offered():
    m = _tally_matcher(PAHVAN, lead='Phaser Turret',
                       votes={'Phaser Turret': 1,
                              'Omni-Directional Pahvan Proton Beam Array': 5})
    m._ml_disabled = True
    name, conf, _, _ = m.match(CROP, candidate_names=SLOT)

    assert name == 'Omni-Directional Pahvan Proton Beam Array'
    assert conf < 0.75


def test_a_server_without_a_tally_still_works():
    """Older backends send only `knowledge`; the leader stands alone."""
    m = _matcher(query=TURRET, gallery={'Phaser Turret': TURRET})
    name, _, _, _ = m.match(CROP, candidate_names=SLOT)

    assert name == 'Phaser Turret'


# ── The sync client keeps the tally ────────────────────────────────────────

def test_the_sync_client_keeps_the_tally_and_caches_it(monkeypatch, tmp_path):
    import io
    import json
    import urllib.request
    from warp import userdata
    from warp.knowledge.sync_client import WARPSyncClient

    payload = {'knowledge': {'aa': 'Phaser Turret'},
               'votes': {'aa': {'Phaser Turret': 3, 'Precision': 2}}}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda *a, **k: _Resp(json.dumps(payload).encode()))
    import threading
    # The constructor starts download threads; set the fields by hand, as
    # test_sets_gaps_push does.
    client = WARPSyncClient.__new__(WARPSyncClient)
    client._knowledge, client._knowledge_votes = {}, {}
    client._knowledge_lock = threading.Lock()
    client._url = 'http://127.0.0.1'
    client._download_knowledge_bg(force=True)

    assert client.get_knowledge_votes('aa') == {'Phaser Turret': 3, 'Precision': 2}
    cached = json.loads(userdata.knowledge_cache_file().read_text())
    assert cached['votes'] == payload['votes']
