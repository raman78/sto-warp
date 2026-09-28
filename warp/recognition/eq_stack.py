"""Equipment-panel geometry when the screenshot carries no row labels.

`eq_geometry.detect_eq_geometry` places the panel from its OCR-read labels.
A screenshot cropped past the label column (or read by a failed OCR pass)
has none, and the equipment panel used to get no boxes at all. This module
finds the same grid from the cells alone.

The game right-justifies every equipment row, so the panel's rightmost
column holds a cell in every row, one row pitch apart, filled or not. And
every ship draws Deflector, Engines, Warp Core and Shield (Sec-Def on some)
as one-cell rows directly under Fore Weapons — a shape no other panel has.

  1. seed: icon-sized bright components (`trait_grid._detect_icon_ccs`),
     grouped by right edge and size, chained at whole multiples of one row
     pitch — bright detection misses dark or merged icons, so gaps occur
  2. every row inside the seed that no component covers must pass
     `LayoutDetector._cell_exists`, which answers for empty cells too
  3. extend up and down while `_cell_exists` still sees a cell
  4. panel_right = median right edge of the seed; cell pitch in x = the span
     of each row of contiguous cells over its steps (a fraction of a pixel)
  5. cells per row = the length of the `_count_icons_in_row` walk, called
     with the arguments `_detect_via_pixel_analysis` builds from a geometry
  6. the panel is one row, a run of 3..6 one-cell rows, then 3..8 rows
     (`_fit_panel`); rows outside that belong to something else
  7. rows named from that structure (Fore, the run, Aft) and, with a matcher,
     the rows under Aft named by what is in them (`_content_anchors`)

Measured on the 109 annotated space screenshots with OCR switched off
(docs/EQ_DETECTION.md, "When the panel has no labels").
"""
from __future__ import annotations

from statistics import median

from warp.debug import log
from warp.recognition import space_eq_rows as _ROWS
from warp.recognition.eq_geometry import EQGeometry, STD_ORDER
from warp.recognition.trait_grid import _detect_icon_ccs

MAX_ROWS = 16      # more rows than any panel draws: stop extending past it
SINGLES = (3, 6)   # one-cell rows under Fore: 4, 5 with Sec-Def; 3 and 6 tolerated
EDGE_TOL = 3       # px — right edges of one column
SIZE_TOL = 3       # px — cell size within one column
MAX_GAP = 4        # rows a seed may skip where no bright component was found
MIN_CONF = 0.40    # below this the trainer's match worker treats a match as unmatched
MAX_MISFIT = 0.10  # share of read cells an arrangement may leave out of place


def _groups(ccs):
    groups: list[list] = []
    for c in sorted(ccs, key=lambda c: c[0] + c[2]):
        for g in groups:
            r = g[0]
            if (abs((c[0] + c[2]) - (r[0] + r[2])) <= EDGE_TOL
                    and abs(c[2] - r[2]) <= SIZE_TOL and abs(c[3] - r[3]) <= SIZE_TOL):
                g.append(c)
                break
        else:
            groups.append([c])
    return groups


def _seed_chains(col, pitch):
    """Chains of components whose gaps are whole multiples (<= MAX_GAP) of pitch."""
    out, cur = [], [col[0]]
    for a, b in zip(col, col[1:]):
        k = round((b[1] - a[1]) / pitch)
        if 1 <= k <= MAX_GAP and abs((b[1] - a[1]) - k * pitch) <= 0.12 * pitch * k:
            cur.append(b)
        else:
            out.append(cur)
            cur = [b]
    out.append(cur)
    return [c for c in out if len(c) >= 3]


def _stack_from_chain(img, detector, ccs, chain, pitch):
    w = int(median(c[2] for c in chain))
    h = int(median(c[3] for c in chain))
    right = int(median(c[0] + c[2] for c in chain))
    y0 = chain[0][1]
    n = round((chain[-1][1] - y0) / pitch) + 1
    pitch = (chain[-1][1] - y0) / (n - 1)          # refit on the chain's span
    seen = {round((c[1] - y0) / pitch) for c in chain}

    def exists(y):
        return detector._cell_exists(img, right - w, int(round(y)), w, h, w * 1.05)

    for i in range(n):                              # interior rows must be cells
        if i not in seen and not exists(y0 + i * pitch):
            return None
    rows = [y0 + i * pitch for i in range(n)]
    while len(rows) < MAX_ROWS and exists(rows[0] - pitch):
        rows.insert(0, rows[0] - pitch)
    while len(rows) < MAX_ROWS and exists(rows[-1] + pitch):
        rows.append(rows[-1] + pitch)
    # Per row of contiguous cells, the span between its first and last cell
    # over the steps between them — a fraction of a pixel, where the median of
    # single steps (whole pixels, 34 or 35) drifted 2 px by the fifth cell.
    dxs = []
    for y in rows:
        xs = sorted({c[0] for c in ccs if abs(c[1] - y) <= SIZE_TOL + 2
                     and right - 7 * w <= c[0] + c[2] <= right + EDGE_TOL})
        if len(xs) >= 2 and all(0.9 * w <= b - a <= 1.5 * w for a, b in zip(xs, xs[1:])):
            dxs.append((xs[-1] - xs[0]) / (len(xs) - 1))
    dx = median(dxs) if dxs else w * 1.05
    return dict(rows=rows, pitch=pitch, right=right, dx=dx, w=w, h=h)


def _cells_per_row(img, detector, s):
    """Cells the game drew in each row — the length of the `_count_icons_in_row`
    walk, which stops where `_cell_exists` does. Keeps each row's cell states."""
    cell_w = max(20, int(round(s['dx'])))
    icon_h = max(20, int(round(s['pitch'] * 0.85)) + 2)
    counts, states_all = [], []
    for i, y in enumerate(s['rows']):
        cy = int(round(y + s['h'] / 2))
        _, states = detector._count_icons_in_row(
            img, max(0, cy - icon_h // 2), min(img.shape[0], cy + icon_h // 2),
            s['right'], cell_w, f'stack row {i}',
            panel_x_start=int(round(s['right'] - 6 * s['dx'])))
        counts.append(len(states))
        states_all.append(states)
    s['states'] = states_all
    return counts


def _fit_panel(cells):
    """(top, run length, bottom) of the equipment panel inside a stack, or None.

    Every ship has Deflector, Engines, Warp Core and Shield (Sec-Def on some)
    as one-cell rows directly under Fore Weapons, and at least Aft Weapons,
    Devices and a console row under them. So the panel is: one row, a run of
    one-cell rows, then 3..8 more rows (two rows above an ambiguous run). A trait grid never fits — the game
    draws the frame of every empty trait slot — and a side list fails the
    rule that the run starts on the second row.
    """
    best = None
    i = 0
    while i < len(cells):
        if cells[i] != 1:
            i += 1
            continue
        j = i
        while j < len(cells) and cells[j] == 1:
            j += 1
        run, below = j - i, len(cells) - j
        if i >= 1 and SINGLES[0] <= run <= SINGLES[1] and below >= 3:
            # A run of 4 or 5 is exactly Deflector..Shield, so Fore is the
            # row above it. A run of 3 or 6 may be one short or long because
            # something covered a row — a weapon tooltip over the Deflector
            # row made it read five cells and hid Fore one row further up —
            # so one more row is kept above, for the content to name.
            top = i - 1 if run in (4, 5) else max(0, i - 2)
            cand = (top, run, j + min(below, 8))
            if best is None or ((run in (4, 5), cand[2] - cand[0])
                                > (best[1] in (4, 5), best[2] - best[0])):
                best = cand
        i = j
    return best


# The rows a run of one-cell rows names, by the game's order. A run of 3 or 6
# is ambiguous (a misread single, or a one-weapon Aft row joining the run), so
# it names nothing.
_RUN_NAMES = {
    4: ['Deflector', 'Engines', 'Warp Core', 'Shield'],
    5: ['Deflector', 'Sec-Def', 'Engines', 'Warp Core', 'Shield'],
}
_LABEL = {r.slot: r.label for r in _ROWS.ROWS}


def _anchor(slot, s, i):
    return STD_ORDER[_LABEL[slot]], int(round(s['rows'][i] + s['h'] / 2))


def _structural_anchors(s):
    """{STD_ORDER index: cy} for the rows the panel's shape names."""
    names = _RUN_NAMES.get(s['run'])
    if names is None:
        return {}
    seq = ['Fore Weapons'] + names + ['Aft Weapons']
    return dict(_anchor(slot, s, i) for i, slot in enumerate(seq))


# ── Rows under Aft Weapons, named by what is in them ─────────────────────────
# Under Aft the game draws [Experimental], Devices, [Universal], Engineering,
# Science, Tactical, [Hangars]. The number of rows fixes how many optional
# ones are present but not which. Each possible arrangement is scored by the
# cells whose item could not sit in the row it assigns — "could sit" read
# straight from the cargo groups: a Universal console is in every console
# group, and the Universal row's group holds every console.
_TAIL = _ROWS.ROWS[_ROWS.INDEX['Aft Weapons'] + 1:]


def _arrangements(n):
    """Every on-screen sequence of n rows under Aft Weapons the table allows."""
    opt = [r.slot for r in _TAIL if r.optional]
    out = []
    for mask in range(1 << len(opt)):
        present = {o for i, o in enumerate(opt) if mask >> i & 1}
        seq = [r.slot for r in _TAIL if not r.optional or r.slot in present]
        if len(seq) == n:
            out.append(seq)
    return out


def _row_items(img, s, row_i, matcher, candidates):
    """Item names the matcher reads in a row's filled cells, each cell cut
    exactly as `_detect_via_pixel_analysis` would emit it."""
    cell_w = max(20, int(round(s['dx'])))
    icon_w = max(20, cell_w - 2)
    icon_h = max(20, int(round(s['pitch'] * 0.85)) + 2)
    cy = int(round(s['rows'][row_i] + s['h'] / 2))
    names = []
    for j, state in enumerate(s['states'][row_i]):
        if state != 'active':
            continue
        bx = int(round(s['right'] - (j + 1) * s['dx'])) + 1
        by = max(0, cy - icon_h // 2)
        crop = img[by:by + icon_h, max(0, bx):bx + icon_w]
        if crop.size == 0:
            continue
        name, conf, _thumb, _sess = matcher.match(crop, candidate_names=candidates)
        if name and not name.startswith('__') and conf >= MIN_CONF:
            names.append(name)
    return names


def _content_anchors(img, s, matcher, eq_cache):
    """{STD_ORDER index: cy} for the rows under Aft Weapons it can name.

    The stack can run past the panel's bottom, so trailing rows in which no
    item was read may be something else: arrangements with and without them
    are all scored. The best score may leave at most MAX_MISFIT of the read
    cells out of place — a sole arrangement is not right merely because it is
    the only one. Where several share the best score, a row is named only if
    every one of them names it the same; the rest stay unnamed, and the log
    says why.
    """
    first = 1 + s['run'] + 1                       # Fore, the run, Aft, then these
    rows = list(range(first, len(s['rows'])))
    if not rows:
        return {}
    allowed = {r.slot: set(eq_cache.get(r.key, {})) for r in _TAIL}
    candidates = set().union(*allowed.values())
    items = [_row_items(img, s, i, matcher, candidates) for i in rows]
    read = sum(len(x) for x in items)
    trailing_empty = 0
    for names in reversed(items):
        if names:
            break
        trailing_empty += 1
    scored = []
    for drop in range(trailing_empty + 1):
        n = len(rows) - drop
        for seq in _arrangements(n):
            misfit = sum(1 for names, slot in zip(items[:n], seq)
                         for x in names if x not in allowed[slot])
            scored.append((misfit, seq))
    if not scored or read == 0:
        log.info(f'eq_stack: {len(rows)} row(s) under Aft Weapons left unnamed — '
                 f'{"no item read in them" if read == 0 else "no arrangement of that many rows exists"}')
        return {}
    best = min(v for v, _ in scored)
    if best > max(1, MAX_MISFIT * read):
        log.info(f'eq_stack: {len(rows)} row(s) under Aft Weapons left unnamed — the best '
                 f'arrangement leaves {best} of {read} read cells in a row they cannot sit in')
        return {}
    tied = [seq for v, seq in scored if v == best]
    out, unnamed = {}, 0
    for k, i in enumerate(rows):
        names = {seq[k] if k < len(seq) else None for seq in tied}
        if len(names) == 1 and None not in names:
            idx, cy = _anchor(names.pop(), s, i)
            out[idx] = cy
        else:
            unnamed += 1
    if unnamed:
        log.info(f'eq_stack: {unnamed} row(s) under Aft Weapons left unnamed — '
                 f'{len(tied)} arrangements explain the panel equally well and disagree there')
    return out


def detect_eq_stack(img, detector, matcher=None, eq_cache=None) -> EQGeometry | None:
    """The equipment panel's grid from its cells alone, or None.

    *detector* is the `LayoutDetector` asking — its `_cell_exists` and
    `_count_icons_in_row` are what judge a cell. With *matcher* and
    *eq_cache* (the cargo equipment groups) the rows under Aft Weapons are
    named by their content; without them only the rows the panel's shape
    names are. Rows left unnamed are handled downstream exactly like rows
    whose OCR label was missed.
    """
    if img is None or img.size == 0:
        return None
    ccs = _detect_icon_ccs(img)
    best = None
    for g in _groups(ccs):
        if len(g) < 3:
            continue
        col = sorted(g, key=lambda c: c[1])
        h = median(c[3] for c in col)
        gaps = [b[1] - a[1] for a, b in zip(col, col[1:]) if 1.0 * h <= b[1] - a[1] <= 1.4 * h]
        if not gaps:
            continue
        pitch = median(gaps)
        for chain in _seed_chains(col, pitch):
            s = _stack_from_chain(img, detector, ccs, chain, pitch)
            if s is None:
                continue
            cells = _cells_per_row(img, detector, s)
            fit = _fit_panel(cells)
            log.debug(f'eq_stack: candidate right={s["right"]} pitch={s["pitch"]:.1f} '
                      f'cells={cells} → {"panel" if fit else "not a panel"}')
            if fit is None:
                continue
            top, run, bottom = fit
            s.update(rows=s['rows'][top:bottom], cells=cells[top:bottom], run=run,
                     states=s['states'][top:bottom])
            key = (run in (4, 5), bottom - top)
            if best is None or key > best[0]:
                best = (key, s)
    if best is None:
        log.info('eq_stack: no equipment panel found without labels')
        return None
    s = best[1]
    anchors = _structural_anchors(s)
    if not anchors:
        log.info(f'eq_stack: a run of {s["run"]} one-cell rows is ambiguous — '
                 f'no row named from the panel\'s shape')
    elif matcher is not None and eq_cache:
        anchors.update(_content_anchors(img, s, matcher, eq_cache))
    log.info(f'eq_stack: panel without labels — {len(s["rows"])} rows, cells {s["cells"]}, '
             f'right={s["right"]} dx={s["dx"]:.2f} pitch={s["pitch"]:.1f}, '
             f'{len(anchors)} row(s) named')
    return EQGeometry(
        panel_x_start=int(round(s['right'] - 6 * s['dx'])),
        panel_right=s['right'],
        final_dx=float(s['dx']),
        row_pitch=int(round(s['pitch'])),
        row_cys=[int(round(y + s['h'] / 2)) for y in s['rows']],
        mode='stack',
        eq_label_cys=anchors,
    )
