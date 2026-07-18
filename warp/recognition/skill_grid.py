"""Space / ground skill-tree recognition.

Unlike equipment or traits, the skill tree is a **fixed grid**: every player's
tree has the same node layout; only each node's ON (trained) / OFF (untrained
or locked) state differs. Recognition therefore reduces to:

  1. anchor a fixed node template onto the screenshot, and
  2. read each node's ON/OFF state.

ON nodes render the icon in vivid career colour; OFF nodes are greyed and
locked nodes show a padlock. The two separate cleanly on the *fraction of
vivid pixels* (high saturation AND high value) inside the node tile — greyed /
padlock tiles have essentially none.

Output matches the SETS build schema so it can be exported to a skill-tree
JSON that SETS imports via ``import_skill_tree_file``:

    space_skills : {'eng': [30 bool], 'sci': [30 bool], 'tac': [30 bool]}
                   index i = rank*6 + sub*3 + node
    ground_skills: [[6 bool], [6 bool], [4 bool], [4 bool]]

The template lives in ``skill_template.json`` (calibrated once from a full
space skill-tree capture). numpy-only so the light test suite can exercise it
without cv2 / torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from warp.debug import log

_TEMPLATE_PATH = Path(__file__).with_name('skill_template.json')
_VIVID_SAT = 0.5      # a pixel counts as "vivid" above this saturation ...
_VIVID_VAL = 0.45     # ... and this brightness
_ON_FRACTION = 0.05   # ON if this fraction of the tile is vivid
_TILE_R = 13          # half-size (px) of the sampled node tile


def _load_template() -> dict:
    return json.loads(_TEMPLATE_PATH.read_text())


_TEMPLATE = _load_template()


# --- colour helpers -------------------------------------------------------

def _sat_val(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (saturation, value) in 0..1 for an HxWx3 uint8 RGB array."""
    a = rgb.astype(np.float32) / 255.0
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return sat, mx


def _is_on(sat: np.ndarray, val: np.ndarray, cx: int, cy: int) -> bool:
    """True if the node tile centred at (cx, cy) holds a trained (vivid) icon."""
    s = sat[cy - _TILE_R:cy + _TILE_R, cx - _TILE_R:cx + _TILE_R]
    v = val[cy - _TILE_R:cy + _TILE_R, cx - _TILE_R:cx + _TILE_R]
    if s.size == 0:
        return False
    return float(((s > _VIVID_SAT) & (v > _VIVID_VAL)).mean()) > _ON_FRACTION


# --- node-extent detection (anchors the template onto a full panel) -------

def _erode3(mask: np.ndarray) -> np.ndarray:
    e = mask.copy()
    e[1:, :] &= mask[:-1, :]
    e[:-1, :] &= mask[1:, :]
    e[:, 1:] &= mask[:, :-1]
    e[:, :-1] &= mask[:, 1:]
    return e


def _tile_centres(val: np.ndarray) -> list[tuple[int, int]]:
    """Square icon tiles sit on a near-black background — return their centres.

    Used only to find the *extent* of the node grid; individual misses don't
    matter as long as the corner nodes are found.
    """
    mask = _erode3(val > 0.10)
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    centres: list[tuple[int, int]] = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        y0 = y1 = sy
        x0 = x1 = sx
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            y0, y1 = min(y0, y), max(y1, y)
            x0, x1 = min(x0, x), max(x1, x)
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        hh, ww = y1 - y0 + 1, x1 - x0 + 1
        if 12 <= hh <= 70 and 12 <= ww <= 70 and 0.55 <= ww / hh <= 1.45 \
                and size / (hh * ww) >= 0.3:
            centres.append(((x0 + x1) // 2, (y0 + y1) // 2))
    return centres


def _robust_bound(vals: list[int], low: bool, k: float = 1.4) -> int:
    """Extent bound that ignores a lone off-grid outlier tile.

    Walks in from the extreme while the gap to the next tile exceeds ``k`` ×
    the typical (small) inter-tile spacing, so a spurious detection off the
    edge of the grid (e.g. a stray UI element) doesn't stretch the anchor and
    throw the whole template off.
    """
    import statistics
    s = sorted(vals)
    gaps = sorted(g for g in (s[i + 1] - s[i] for i in range(len(s) - 1)) if g > 0)
    unit = statistics.median(gaps[:max(3, len(gaps) // 2)]) if gaps else 0
    thr = max(k * unit, 15)
    if low:
        i = 0
        while i + 1 < len(s) and s[i + 1] - s[i] > thr:
            i += 1
        return s[i]
    j = len(s) - 1
    while j - 1 >= 0 and s[j] - s[j - 1] > thr:
        j -= 1
    return s[j]


def _node_extent(val: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) of node centres, or None if too few.

    Uses outlier-tolerant bounds so a stray detection doesn't distort the
    anchor (see :func:`_robust_bound`).
    """
    centres = _tile_centres(val)
    if len(centres) < 12:
        return None
    xs = [c[0] for c in centres]
    ys = [c[1] for c in centres]
    return (_robust_bound(xs, True), _robust_bound(ys, True),
            _robust_bound(xs, False), _robust_bound(ys, False))


# --- public API -----------------------------------------------------------

def detect_space(rgb: np.ndarray,
                 extent: tuple[int, int, int, int] | None = None) -> dict:
    """Recognise a space skill tree -> ``space_skills`` dict.

    *extent* is the (x0, y0, x1, y1) bounding box of node centres; when omitted
    it is auto-detected. Pass an explicit extent (e.g. from a user-confirmed
    layout) for scrolled / partial captures where auto-detection is unreliable.
    """
    sat, val = _sat_val(rgb)
    if extent is None:
        extent = _node_extent(val)
    if extent is None:
        log.warning('SkillGrid: could not anchor space template (too few tiles)')
        return {c: [False] * 30 for c in ('eng', 'sci', 'tac')}
    x0, y0, x1, y1 = extent
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    pos = _TEMPLATE['space']['positions']
    out: dict[str, list[bool]] = {}
    for career in ('eng', 'sci', 'tac'):
        states = []
        for nx, ny in pos[career]:
            cx = int(round(x0 + nx * w))
            cy = int(round(y0 + ny * h))
            states.append(_is_on(sat, val, cx, cy))
        out[career] = states
    log.info('SkillGrid: space eng=%d sci=%d tac=%d ON'
             % tuple(sum(out[c]) for c in ('eng', 'sci', 'tac')))
    return out


_GROUND_SHAPE = (6, 6, 4, 4)


def detect_ground(rgb: np.ndarray,
                  extent: tuple[int, int, int, int] | None = None) -> list:
    """Recognise a ground skill tree -> ``ground_skills`` = [[6],[6],[4],[4]].

    Trees are ordered top-left, top-right, bottom-left, bottom-right; node ids
    follow the SETS ground layout. *extent* as in :func:`detect_space`.
    """
    sat, val = _sat_val(rgb)
    if extent is None:
        extent = _node_extent(val)
    if extent is None:
        log.warning('SkillGrid: could not anchor ground template (too few tiles)')
        return [[False] * n for n in _GROUND_SHAPE]
    x0, y0, x1, y1 = extent
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    trees = _TEMPLATE['ground']['positions']
    out: list[list[bool]] = []
    for tree in trees:
        states = []
        for nx, ny in tree:
            cx = int(round(x0 + nx * w))
            cy = int(round(y0 + ny * h))
            states.append(_is_on(sat, val, cx, cy))
        out.append(states)
    log.info('SkillGrid: ground ON per tree = %s'
             % [sum(t) for t in out])
    return out


def detect(rgb: np.ndarray, env: str,
           extent: tuple[int, int, int, int] | None = None) -> dict:
    """Dispatch on *env* ('space' | 'ground')."""
    if env == 'space':
        return {'space_skills': detect_space(rgb, extent)}
    if env == 'ground':
        return {'ground_skills': detect_ground(rgb, extent)}
    raise ValueError(f'skill_grid.detect: unknown env {env!r}')


# --- SETS export ----------------------------------------------------------

def to_skill_tree(space_skills: dict | None = None,
                  ground_skills: list | None = None) -> dict:
    """Assemble a SETS-importable skill-tree dict from recognised skills.

    Built on the SETS-shaped ``empty_build('skills')`` skeleton, so any env we
    didn't recognise stays at its empty default. The result loads via SETS'
    ``File → Load Skill Tree`` (``import_skill_tree_file``) as JSON. Milestone
    unlock choices (``skill_unlocks``) and descriptions are left empty in v1 —
    SETS' ``merge_build`` fills them from defaults.
    """
    from warp.data.empty_build import empty_build
    tree = empty_build('skills')
    if space_skills is not None:
        tree['space_skills'] = space_skills
    if ground_skills is not None:
        tree['ground_skills'] = ground_skills
    return tree


def detect_boxes(rgb: np.ndarray, env: str,
                 extent: tuple[int, int, int, int] | None = None) -> list:
    """Per-node boxes for a canvas overlay: ``[(x, y, w, h, on), ...]``.

    Image-pixel coordinates (x, y = top-left). *on* is the recognised state.
    Env 'space' walks eng→sci→tac; 'ground' walks the 4 trees.
    """
    sat, val = _sat_val(rgb)
    if extent is None:
        extent = _node_extent(val)
    if extent is None:
        return []
    x0, y0, x1, y1 = extent
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    side = max(16, round(0.05 * w))
    boxes = []

    def _emit(nx, ny):
        cx = int(round(x0 + nx * w))
        cy = int(round(y0 + ny * h))
        boxes.append((cx - side // 2, cy - side // 2, side, side,
                      _is_on(sat, val, cx, cy)))

    if env == 'space':
        for career in ('eng', 'sci', 'tac'):
            for nx, ny in _TEMPLATE['space']['positions'][career]:
                _emit(nx, ny)
    elif env == 'ground':
        for tree in _TEMPLATE['ground']['positions']:
            for nx, ny in tree:
                _emit(nx, ny)
    return boxes


def env_of(rgb: np.ndarray) -> str | None:
    """Guess 'space' / 'ground' from the node-grid aspect, or None.

    The space tree is tall (w/h ≈ 0.62), the ground tree is wide (≈ 1.4), so
    the node-extent aspect separates them cleanly. Lets us recognise a screen
    the classifier only labelled generic ``SKILLS``.
    """
    _, val = _sat_val(rgb)
    ext = _node_extent(val)
    if ext is None:
        return None
    x0, y0, x1, y1 = ext
    h = y1 - y0
    if h <= 0:
        return None
    return 'space' if (x1 - x0) / h < 1.0 else 'ground'


_SKILL_STYPES = ('SKILLS', 'SPACE_SKILLS', 'GROUND_SKILLS')


def skills_from_files(typed_files: dict[str, str]) -> dict:
    """Recognise skill screens among *typed_files* ({path: screen_type}).

    Returns the SETS build fragments actually found — ``{'space_skills': ...}``
    and/or ``{'ground_skills': ...}``. Env comes from the screen type when it
    is SPACE_SKILLS / GROUND_SKILLS, else from :func:`env_of`. Lets WARP fold
    skills into its normal SETS-build export (the pipeline skips skill screens
    because they carry no items).
    """
    import numpy as _np
    from PIL import Image

    out: dict = {}
    for path, stype in typed_files.items():
        if stype not in _SKILL_STYPES:
            continue
        try:
            rgb = _np.asarray(Image.open(path).convert('RGB'))
        except Exception as e:  # noqa: BLE001
            log.warning(f'SkillGrid: cannot read {path}: {e}')
            continue
        env = ('space' if stype == 'SPACE_SKILLS' else
               'ground' if stype == 'GROUND_SKILLS' else env_of(rgb))
        if env == 'space' and 'space_skills' not in out:
            out['space_skills'] = detect_space(rgb)
        elif env == 'ground' and 'ground_skills' not in out:
            out['ground_skills'] = detect_ground(rgb)
    return out
