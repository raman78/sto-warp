"""Session and template matching as matrix products, identical to OpenCV.

`_best_session_match` compared a crop with every session example — 22 000 on
the maintainer's machine — one `cv2.matchTemplate` and one `cv2.compareHist`
at a time; profiled on image-cda05d5238072b99.png that loop was 38.7 of 47.8 s
of recognition. `_template_scores` scored all 4 435 wiki icons and the caller
then discarded those outside the slot. And every recognition rebuilt the icon
index from the PNGs.

Each test compares against OpenCV computed here, in the test — the loop the
matrix products replace — never against the implementation itself.

Run standalone:
    python -m pytest tests/test_matcher_vectorised.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


@pytest.fixture(autouse=True)
def _empty_pool():
    """The pool and the caches are class state — leave them as found."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    saved = (SETSIconMatcher._session_examples, SETSIconMatcher._sess_stack,
             SETSIconMatcher._index_cache)
    SETSIconMatcher._session_examples = []
    SETSIconMatcher._sess_stack = None
    yield
    (SETSIconMatcher._session_examples, SETSIconMatcher._sess_stack,
     SETSIconMatcher._index_cache) = saved


def _crop(seed: int, h=40, w=34) -> np.ndarray:
    """An icon-like crop: blocks of colour plus noise, so it has structure."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (4, 4, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int16)
    img += rng.integers(-25, 25, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def _query(seed: int):
    from warp.recognition.icon_matcher import MATCH_SIZE, SETSIconMatcher
    c64 = cv2.resize(_crop(seed), (MATCH_SIZE, MATCH_SIZE), interpolation=cv2.INTER_AREA)
    return c64, SETSIconMatcher._hist_hsv(c64)


def _opencv(c64, q_hist, candidates):
    """The loop `_best_session_match` replaced, verbatim in behaviour."""
    from warp.recognition.icon_matcher import HIST_BINS, HIST_WEIGHT, SETSIconMatcher
    best = ('', 0.0, None)
    for e in SETSIconMatcher._session_examples:
        if candidates is not None and e['name'] not in candidates:
            continue
        if e['hist_hsv'].shape != tuple(HIST_BINS):
            continue
        tm = float(cv2.matchTemplate(c64, e['tmpl64'], cv2.TM_CCOEFF_NORMED).max())
        h = max(0.0, float(cv2.compareHist(q_hist, e['hist_hsv'], cv2.HISTCMP_CORREL)))
        combined = tm * (1.0 - HIST_WEIGHT) + h * HIST_WEIGHT
        if combined > best[1]:
            best = (e['name'], combined, e)
    return best


def _pool(n=60, names=12):
    from warp.recognition.icon_matcher import SETSIconMatcher
    for i in range(n):
        SETSIconMatcher.add_session_example(_crop(1000 + i), f'Item {i % names}',
                                            origin='community')


def _same(got, ref):
    return got[0] == ref[0] and got[2] is ref[2] and abs(got[1] - ref[1]) <= 1e-4


# ── Session examples ────────────────────────────────────────────────────────

def test_session_match_equals_opencv():
    from warp.recognition.icon_matcher import SETSIconMatcher
    _pool()
    m = SETSIconMatcher.__new__(SETSIconMatcher)
    for q in range(25):
        c64, qh = _query(q)
        assert _same(m._best_session_match(c64, qh, None), _opencv(c64, qh, None)), q


def test_session_match_equals_opencv_within_the_slot_names():
    from warp.recognition.icon_matcher import SETSIconMatcher
    _pool()
    m = SETSIconMatcher.__new__(SETSIconMatcher)
    names = {'Item 3', 'Item 7', 'Item 11'}
    for q in range(25):
        c64, qh = _query(q)
        got = m._best_session_match(c64, qh, names)
        assert _same(got, _opencv(c64, qh, names)), q
        assert got[0] in names | {''}


def test_a_query_matching_nothing_in_the_slot_returns_nothing():
    from warp.recognition.icon_matcher import SETSIconMatcher
    _pool()
    m = SETSIconMatcher.__new__(SETSIconMatcher)
    c64, qh = _query(1)
    assert m._best_session_match(c64, qh, {'Not In The Pool'}) == ('', 0.0, None)


def test_a_tie_goes_to_the_earliest_example():
    """The loop kept the first of equal scores (strict `>`)."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    crop = _crop(5)
    SETSIconMatcher.add_session_example(crop, 'First', origin='community')
    SETSIconMatcher.add_session_example(crop, 'Second', origin='community')
    m = SETSIconMatcher.__new__(SETSIconMatcher)
    c64, qh = _query(5)
    got = m._best_session_match(c64, qh, None)
    assert got[0] == 'First'
    assert _same(got, _opencv(c64, qh, None))


def test_a_flat_query_scores_as_opencv_does():
    """A flat crop against a real template is 0 in TM_CCOEFF_NORMED."""
    from warp.recognition.icon_matcher import MATCH_SIZE, SETSIconMatcher
    _pool(20, 5)
    m = SETSIconMatcher.__new__(SETSIconMatcher)
    c64 = np.full((MATCH_SIZE, MATCH_SIZE, 3), 90, np.uint8)
    qh = SETSIconMatcher._hist_hsv(c64)
    assert _same(m._best_session_match(c64, qh, None), _opencv(c64, qh, None))


def test_an_added_example_is_seen_at_once():
    """The stacked copy is rebuilt when the pool changes, never stale."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    _pool(20, 5)
    m = SETSIconMatcher.__new__(SETSIconMatcher)
    c64, qh = _query(77)
    before = m._best_session_match(c64, qh, None)
    SETSIconMatcher.add_session_example(_crop(77), 'Exact Match', origin='user')
    after = m._best_session_match(c64, qh, None)
    assert after[0] == 'Exact Match' != before[0]
    assert _same(after, _opencv(c64, qh, None))


def test_a_removed_example_is_gone_at_once():
    from warp.recognition.icon_matcher import SETSIconMatcher
    _pool(20, 5)
    SETSIconMatcher.add_session_example(_crop(88), 'Exact Match', origin='community')
    m = SETSIconMatcher.__new__(SETSIconMatcher)
    c64, qh = _query(88)
    assert m._best_session_match(c64, qh, None)[0] == 'Exact Match'
    SETSIconMatcher._session_examples = [e for e in SETSIconMatcher._session_examples
                                         if e['name'] != 'Exact Match']
    after = m._best_session_match(c64, qh, None)
    assert after[0] != 'Exact Match'
    assert _same(after, _opencv(c64, qh, None))


def test_a_replaced_example_is_seen_at_once():
    """Same length, one entry swapped in place: a length check would miss it."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    _pool(20, 5)
    m = SETSIconMatcher.__new__(SETSIconMatcher)
    c64, qh = _query(99)
    m._best_session_match(c64, qh, None)
    SETSIconMatcher.add_session_example(_crop(99), 'Swapped In', origin='community')
    new = SETSIconMatcher._session_examples.pop()
    SETSIconMatcher._session_examples[0] = new
    assert m._best_session_match(c64, qh, None)[0] == 'Swapped In'


# ── Wiki icon index ────────────────────────────────────────────────────────

def _icons(tmp_path, n=8):
    d = tmp_path / 'icons'
    d.mkdir(exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(d / f'Icon {i}.png'), _crop(500 + i, 64, 49))
    return d


def test_template_scores_for_slot_rows_equal_the_full_scores(tmp_path, monkeypatch):
    from warp.recognition.icon_matcher import SETSIconMatcher
    monkeypatch.setattr('warp.data.cargo.canonical_names', lambda: set())
    m = SETSIconMatcher(_icons(tmp_path))
    c64, _ = _query(3)
    full = m._template_scores(c64)
    rows = m._index_rows_for({'Icon 2', 'Icon 5'})
    part = m._template_scores(c64, rows)
    assert np.allclose(part[rows], full[rows], atol=1e-6)
    assert np.isneginf(np.delete(part, rows)).all()


def test_template_scores_equal_opencv(tmp_path, monkeypatch):
    """The matrix form already stood in for a cv2 loop; check it still does."""
    from warp.recognition.icon_matcher import (MATCH_SIZE, _TEMPLATE_SLIDE_SIZE,
                                               SETSIconMatcher)
    monkeypatch.setattr('warp.data.cargo.canonical_names', lambda: set())
    icons = _icons(tmp_path)
    m = SETSIconMatcher(icons)
    c64, _ = _query(4)
    got = m._template_scores(c64, m._index_rows_for({'Icon 1', 'Icon 6'}))
    for i, e in enumerate(m._index):
        if e['name'] not in {'Icon 1', 'Icon 6'}:
            continue
        orig = cv2.imread(str(icons / f"{e['name']}.png"))
        t64 = cv2.resize(orig, (MATCH_SIZE, MATCH_SIZE), interpolation=cv2.INTER_AREA)
        t58 = cv2.resize(orig, (_TEMPLATE_SLIDE_SIZE, _TEMPLATE_SLIDE_SIZE),
                         interpolation=cv2.INTER_AREA)
        ref = max(cv2.matchTemplate(c64, t64, cv2.TM_CCOEFF_NORMED).max(),
                  cv2.matchTemplate(c64, t58, cv2.TM_CCOEFF_NORMED).max())
        assert abs(got[i] - ref) <= 1e-4, e['name']


def test_the_icon_index_is_built_once_and_reused(tmp_path, monkeypatch):
    from warp.recognition.icon_matcher import SETSIconMatcher
    monkeypatch.setattr('warp.data.cargo.canonical_names', lambda: set())
    icons = _icons(tmp_path)
    first = SETSIconMatcher(icons)
    second = SETSIconMatcher(icons)
    assert second._tmpl_mat64 is first._tmpl_mat64


def test_new_art_rebuilds_the_icon_index(tmp_path, monkeypatch):
    from warp.recognition.icon_matcher import SETSIconMatcher
    monkeypatch.setattr('warp.data.cargo.canonical_names', lambda: set())
    icons = _icons(tmp_path)
    first = SETSIconMatcher(icons)
    cv2.imwrite(str(icons / 'Icon New.png'), _crop(999, 64, 49))
    second = SETSIconMatcher(icons)
    assert second._tmpl_mat64 is not first._tmpl_mat64
    assert 'Icon New' in {e['name'] for e in second._index}


def test_changed_art_of_the_same_name_rebuilds_the_icon_index(tmp_path, monkeypatch):
    """A file rewritten under the same name — an updated icon — is new art."""
    import os
    from warp.recognition.icon_matcher import SETSIconMatcher
    monkeypatch.setattr('warp.data.cargo.canonical_names', lambda: set())
    icons = _icons(tmp_path)
    first = SETSIconMatcher(icons)
    target = icons / 'Icon 0.png'
    cv2.imwrite(str(target), _crop(12345, 64, 49))
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    second = SETSIconMatcher(icons)
    assert second._tmpl_mat64 is not first._tmpl_mat64


def test_changed_cargo_names_rebuild_the_icon_index(tmp_path, monkeypatch):
    """Era-variant folding reads cargo's names, so they are part of the key."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    icons = _icons(tmp_path)
    monkeypatch.setattr('warp.data.cargo.canonical_names', lambda: set())
    first = SETSIconMatcher(icons)
    monkeypatch.setattr('warp.data.cargo.canonical_names', lambda: {'Icon 0'})
    second = SETSIconMatcher(icons)
    assert second._tmpl_mat64 is not first._tmpl_mat64


def test_a_flat_template_scores_as_opencv_does():
    """OpenCV resolves a flat template's 0/0 to 1.00 against any query.
    `add_session_example` refuses such crops, so no seeder can add one — this
    puts one in the pool directly, to pin the behaviour the refusal relies on."""
    from warp.recognition.icon_matcher import MATCH_SIZE, SETSIconMatcher
    _pool(10, 5)
    flat = np.full((MATCH_SIZE, MATCH_SIZE, 3), 60, np.uint8)
    SETSIconMatcher._session_examples.append({
        'name': 'Flat', 'tmpl64': flat, 'hist_hsv': SETSIconMatcher._hist_hsv(flat),
        'orig': flat, 'origin': 'community'})
    m = SETSIconMatcher.__new__(SETSIconMatcher)
    c64, qh = _query(2)
    got, ref = m._best_session_match(c64, qh, None), _opencv(c64, qh, None)
    assert ref[0] == 'Flat'
    assert _same(got, ref)
