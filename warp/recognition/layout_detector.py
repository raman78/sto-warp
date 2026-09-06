# warp/recognition/layout_detector.py
#
# Detect equipment slot bounding boxes in STO Status-tab screenshots.
# Now with Dynamic Layout Learning — learns from user-confirmed data.
#
# Detection strategy:
#   1. Learned Layouts: Match current screen against known confirmed patterns (anchors.json)
#   2. Pixel analysis: detect dark separators + right edge automatically
#   3. OCR labels (fallback): if analysis fails, fall back to label positions
#   4. Default Anchors (last resort): use calibrated relative positions

from __future__ import annotations

import json
import logging
import os
from difflib import get_close_matches
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
try:
    from warp.debug import log as _slog
except Exception:
    _slog = log

from warp import userdata as _userdata
from warp.recognition import boff_marker as _boff_marker
from warp.recognition import trait_grid as _trait_grid
from warp.recognition.eq_geometry import detect_eq_geometry, EQGeometry, STD_ORDER
from warp.recognition.ground_eq_geometry import (
    detect_ground_eq_geometry,
    project_cells as _project_ground_cells,
    GroundEQGeometry,
    SLOT_KIT_MODULES as _G_KM,
    SLOT_GROUND_DEVICES as _G_DEV,
    SLOT_WEAPONS as _G_WEAPONS,
)

# STD_ORDER (eq_geometry) uses 'Shields' (plural); production uses 'Shield'.
# All other names match 1:1.
_STD_IDX_TO_PROD_SLOT: dict[int, str] = {
    idx: ('Shield' if name == 'Shields' else name)
    for name, idx in STD_ORDER.items()
}


def drop_boxes_on_text(result: dict, ocr_tokens: list[dict] | None,
                       img: np.ndarray, label_width_ratio: float = 2.0) -> dict:
    """A slot never sits on a label. Move it onto its icon, or drop it.

    The game writes a section's name on a band across its panel, and the
    panel's slots live between one band and the next. A projected slot that
    lands on a band is therefore wrong twice over: there is no icon under it,
    and the band is exactly the evidence that the section ended above it.

    Measured on `image-4391ccd9d2683d4e.png`. `Personal Space Traits` is 11 on
    this character, so eleven boxes were projected — but the screenshot shows
    ten, and the eleventh landed level with the `Starship Traits` heading. The
    two boxes of the second `Starship Traits` row landed on the
    `U.S.S. FURY Traits` divider while the real icons sat one row lower. All
    three were then read as blank cells and **auto-confirmed as `__inactive__`
    at confidence 1.00** — teaching the models that a heading is an empty slot.

    Shift before dropping. A row on a divider usually means the icons are just
    below it, so the box is moved and kept if that lands it somewhere clean; a
    box with nowhere to go is dropped. Losing a slot costs the user one box
    drawn by hand, keeping a wrong one costs a poisoned label.

    Where it moves *to* is read off the screenshot, not guessed. Clearing the
    writing is not the same as landing on the artwork: a divider does not
    push the next row down by a fixed amount, and putting the box one pixel
    under the text left it 5 px above the icon and 9 px short of its bottom —
    close enough to match at 0.64 instead of cleanly. So the box is snapped
    onto the icon-shaped blob below the band in its own column, which
    `trait_grid` already finds for its own purposes. The projected size is
    kept and only the position moves, so nothing downstream sees a box of an
    unexpected shape.

    Measured on the same screenshot: the real second `Starship Traits` row
    starts at y=425 and is 40 px tall, and the blob detector returns it as
    `(658, 425, 30, 40)` and `(691, 425, 31, 40)`. Without a blob to snap to
    the box still moves clear of the writing, which is the old behaviour.

    **Only bands, never marks.** A token counts as a label only if it is at
    least `label_width_ratio` times the slot's width. The game prints `Mk XV`
    and `LOC` *inside* icons, and treating those as bands would delete the very
    slots they sit on. Measured on that screenshot: the four headings run
    74-132 px against a 27 px slot, and every in-icon mark 28-46 px, so a 2x
    cut separates them with room on both sides.
    """
    if not ocr_tokens or not result:
        return result

    def _same_slot(a: str, b: str) -> bool:
        """`Shield`/`Shields`, `Hangar`/`Hangars` — the alias table and the
        production slot names disagree on the plural of two rows."""
        return a.lower().rstrip('s') == b.lower().rstrip('s')

    def _heading_kind(text: str, own_slot: str, tok: dict, box) -> str:
        """`own` / `other` / `plain` — the three are treated differently.

        Two things decide it, and neither is enough alone: **which section
        the text names**, and **where it sits relative to the box**.

        **`other`: another section's heading, so the box is past the end of
        its own section.** Drop it. This is the only destructive answer and it
        asks for the most: the text has to resolve to a section that is not
        this box's, *and* be the multi-word form the game prints over a block
        — `Personal Space Traits`, `Space Reputation`. One word is never
        enough, because the equipment column labels its rows with the short
        form and `Weapons` alone resolves to `Aft Weapons`: a single-word test
        here would delete the fore weapons row of every space screenshot.

        **`own`: the row's own label, beside its icons.** The space equipment
        panel writes `Fore Weapons`, `Impulse`, `Warp` in a column to the
        *left* of the row they name, so every equipment row shares a band with
        its own label and none of them is a boundary. (The ground and trait
        panels put the label *above* the block instead; the two layouts are
        described in `docs/EQ_DETECTION.md`.) Keep the box.

        Position is required here too. A heading can end up beside a box
        rather than over it — the phantom eleventh `Personal Space Traits`
        box ends 3 px before `Starship Traits` begins — which is why the
        section it names is checked first.

        **`plain`: everything else** — the `U.S.S. FURY Traits` divider inside
        the starship traits block, an item name, a stat line, and any label
        text over a box that did not earn a drop. The icons are usually just
        below, so the row is moved rather than lost.

        Both tests read the same alias table the OCR-header strategy reads.
        Doing this by hand is what broke it once already: the game labels the
        engines row `Impulse` and the shield row `Shields`, and wraps
        `Engineering Consoles` onto two lines, so nothing that compares the
        text to a slot *name* survives contact with the real panel.
        """
        x, _y, w, _h = box[:4]
        low = (text or '').strip().rstrip(':').lower()
        named = match_slot_label(low)
        if (named and not _same_slot(named, own_slot)
                and len(low.split()) >= 2):
            return 'other'
        beside = tok['x1'] < x or tok['x0'] > x + w
        if beside and (named or is_slot_label_text(low)):
            return 'own'
        return 'plain'

    # The columns each section occupies — its leftmost cell's left edge to its
    # rightmost cell's right edge. This is the panel the section is drawn in,
    # and it is what decides whether a piece of text has anything to do with
    # that section (see `_bands_for`).
    spans = {s: (min(b[0] for b in bs), max(b[0] + b[2] for b in bs))
             for s, bs in result.items() if bs}

    def _bands_for(box) -> list[tuple[int, int]]:
        """Text lying across this section's columns, top to bottom.

        A band is local, not full width. A screenshot holds several panels
        side by side, and `Personal Space Traits` over the trait column says
        nothing about the equipment column 200 px to its left — treating every
        wide token as a full-width band deleted 27 good boxes on the first
        attempt at this.

        Locality is measured against the **section**, not the single box. A
        heading belongs to the panel whose columns it is printed over, so that
        is the test; asking instead whether text is near *a box* needs a
        tolerance, and any tolerance wide enough for a heading that misses its
        own leftmost cell by 3 px is also wide enough to reach into the panel
        next door. Measured over 145 space screenshots with confirmed boxes,
        a padding of twice the slot width cost 85 of them: every loss was a
        row's rightmost cell, struck out by the trait panel's heading or by a
        tooltip 24-150 px to its right.
        """
        w = box[2]
        need = w * label_width_ratio
        lo, hi = spans.get(slot, (box[0], box[0] + w))
        out = []
        for t in ocr_tokens:
            if t.get('w', 0) < need:
                continue
            if t['x1'] < lo or t['x0'] > hi:
                continue
            out.append((t['y0'], t['y1'],
                        _heading_kind(t.get('text', ''), slot, t, box)))
        return sorted(out)

    def _band_at(cy: int, bands):
        for band in bands:
            if band[0] <= cy <= band[1]:
                return band
        return None

    _icon_blobs: list = []

    def _blobs():
        """Icon-shaped blobs, found once per call and only if something moves.

        `trait_grid._detect_icon_ccs` is the detector that already knows what
        an icon looks like on this UI — a threshold, connected components, and
        the height and aspect-ratio cuts calibrated for the trait panels. It
        is called, never copied: a second copy of those cuts would drift from
        the first and the drift would read as a finding.
        """
        if not _icon_blobs:
            try:
                _icon_blobs.extend(_trait_grid._detect_icon_ccs(img) or [])
            except Exception as e:                        # noqa: BLE001
                _slog.warning(f'LayoutDetector: icon blobs unavailable, '
                              f'moved boxes land under the label instead: {e}')
            _icon_blobs.append(None)                      # "already looked"
        return [b for b in _icon_blobs if b is not None]

    def _snap_below(box, band_bottom: int) -> int | None:
        """Top edge that puts *box* on the nearest icon below *band_bottom*.

        The blob has to overlap the box's column and be roughly its size, so a
        stray component cannot drag a row across the panel. Within two row
        heights of the writing, because a divider separates a row from the
        next one, not from the far end of the section.
        """
        x, _y, w, h = box[:4]
        best = None
        for bx, by, bw, bh in _blobs():
            if by < band_bottom or by - band_bottom > 2 * h:
                continue
            if bx + bw <= x or bx >= x + w:
                continue
            if not (0.6 * h <= bh <= 1.6 * h):
                continue
            if best is None or by < best[1]:
                best = (bx, by, bw, bh)
        if best is None:
            return None
        return max(band_bottom + 1, best[1] + (best[3] - h) // 2)

    img_h = int(img.shape[0])

    # ── Pass 1: decide, per box ───────────────────────────────────────────
    # 'keep', 'drop', or a move with the top edge the blobs argue for and the
    # one to use if they argue for nothing.
    plans: dict[str, list] = {}
    row_snaps: dict[tuple, list[int]] = {}
    for slot, boxes in result.items():
        row = []
        for box in boxes:
            y, h = box[1], box[3]
            bands = _bands_for(box)
            hit = _band_at(y + h // 2, bands)
            if hit is None or hit[2] == 'own':
                # No writing under it, or the row's own label level with it.
                row.append(('keep', box, bands, 0))
                continue
            if hit[2] == 'other':
                # Another section's heading — this box is past the end of its
                # own section, and moving it would only put it on the next
                # section's slots.
                row.append(('drop', box, bands, 0))
                continue
            snap = _snap_below(box, hit[1])
            if snap is not None:
                row_snaps.setdefault((slot, y), []).append(snap)
            row.append(('move', box, bands, hit[1] + 1))
        plans[slot] = row

    # ── Pass 2: a row moves as a row ──────────────────────────────────────
    # An unlit cell has no blob to snap to — `__inactive__` slots are drawn
    # dark on purpose — so on the screenshot this came from one box of the
    # pair landed on its icon and the other stayed 8 px high, splitting a row
    # that the game draws on one line. The cells that did find their icon
    # answer for the ones that could not.
    out: dict = {}
    moved = dropped = 0
    for slot, row in plans.items():
        kept = []
        for what, box, bands, fallback_y in row:
            if what == 'keep':
                kept.append(box)
                continue
            if what == 'drop':
                dropped += 1
                continue
            x, y, w, h = box[:4]
            snaps = row_snaps.get((slot, y)) or []
            shifted_y = (sorted(snaps)[len(snaps) // 2] if snaps
                         else fallback_y)
            if (shifted_y + h <= img_h
                    and _band_at(shifted_y + h // 2, bands) is None):
                kept.append((x, shifted_y, w, h) + tuple(box[4:]))
                moved += 1
            else:
                dropped += 1
        if kept:
            out[slot] = kept
        elif slot in result:
            # An emptied slot is still reported, so a caller can tell "this
            # section has no boxes" from "this section was never detected".
            out[slot] = []
    if moved or dropped:
        _slog.info(f'LayoutDetector: {moved} slot box(es) moved clear of a '
                   f'label, {dropped} dropped — a slot never sits on one')
    return out


def merge_trait_boxes(result: dict, trait_grid_res: dict,
                      in_boff_panel) -> dict:
    """Take the trait grid's boxes for a section — unless that loses slots.

    `trait_grid` is the structure-driven detector, measured at 91.5% slot IoU
    against the OCR-header baseline, so where it finds a whole section its
    positions win outright.

    What it must not do is shrink one. It used to replace unconditionally, and
    the counter reporting the change could come out negative. Measured on
    `image-4391ccd9d2683d4e.png`: the grid split one Starship Traits section
    across two row groups and dropped the second as a duplicate — a section
    really does appear once per screen — so the merge took the survivor.
    `Starship Traits` fell from the 7 the ship's profile says it has to 2,
    `Space Reputation` 5 to 3, `Personal Space Traits` 11 to 10. Eight boxes
    vanished, and the slots behind them were never drawn: nothing marked them
    for review, so the user had to notice the gap and draw them by hand.

    The rule is the one the equipment rows already follow. The profile says how
    many slots exist; a detector that sees fewer does not get to delete the
    rest. An extra box that turns out empty is confirmed away in a keystroke, a
    missing one is manual work nobody is prompted to do.

    `result` is modified in place and returned, as the caller expects.
    """
    if not trait_grid_res:
        return result
    added = 0
    dropped = 0
    kept: list[str] = []
    for slot, bxs in trait_grid_res.items():
        clean = [b for b in bxs if not in_boff_panel(b)]
        dropped += len(bxs) - len(clean)
        if not clean:
            continue
        have = len(result.get(slot, []))
        if len(clean) < have:
            kept.append(f'{slot} {len(clean)}<{have}')
            continue
        added += len(clean) - have
        result[slot] = clean
    if dropped:
        _slog.info(f'LayoutDetector: trait_grid dropped {dropped} '
                   f'bboxes overlapping BOFF marker panel')
    if kept:
        # Never set aside quietly: a count that keeps appearing here is how
        # anyone learns the grid is splitting a section in two.
        _slog.info(
            f'LayoutDetector: trait_grid found fewer boxes than the '
            f'profile-sized rows for {kept} — keeping the rows, so no slot '
            f'goes undrawn')
    _slog.info(f'LayoutDetector: trait_grid merged → '
               f'{list(trait_grid_res.keys())} (+{added} bboxes)')
    return result


def fill_unanchored_rows(row_cys: list[int], cy_to_slot: dict[int, str],
                         expected: list[str]) -> dict[int, str]:
    """Name the rows OCR could not read, from the rows it could.

    `expected` is the sequence of slots this panel should show, top to bottom —
    for space that is the canonical panel order filtered by the ship's profile,
    so reading the ship and its tier already answers which rows exist. OCR
    labels anchor whichever rows they can; every run of unanchored rows between
    two anchors is then filled from the slots that lie between those anchors in
    `expected`.

    **Only when the count is unambiguous.** A run of two unanchored rows is
    filled only if exactly two expected slots sit between its neighbours. If
    the numbers disagree, the rows stay unnamed — an unnamed row costs the user
    one manual box, a wrongly named one silently writes an item into the wrong
    slot.

    This replaces indexing a flat list by row number, which broke whenever the
    list's order did not match the panel's. Measured on
    `image-4e7c6849dd28da67.png`, where the Devices and Universal Consoles
    labels are covered: the list placed `Hangars` at position 6 — because
    `Hangars` was inserted right after `Aft Weapons`, while the game draws it
    last — so row 6 was left empty, row 7 took `Devices`, every row below
    shifted down one, and `Universal Consoles` was never placed at all.

    Returns a new mapping; the input is not modified. Anchored rows always win.
    """
    out = dict(cy_to_slot)
    if not row_cys or not expected:
        return out

    idx_of = {s: i for i, s in enumerate(expected)}
    anchors = [(i, cy_to_slot[cy]) for i, cy in enumerate(row_cys)
               if cy in cy_to_slot and cy_to_slot[cy] in idx_of]
    if not anchors:
        return out

    def _assign(rows: list[int], slots: list[str]) -> None:
        if len(rows) != len(slots) or not slots:
            return
        for r, s in zip(rows, slots):
            out[row_cys[r]] = s

    # Between consecutive anchors.
    for (ra, sa), (rb, sb) in zip(anchors, anchors[1:]):
        gap = [r for r in range(ra + 1, rb) if row_cys[r] not in cy_to_slot]
        if not gap or idx_of[sb] <= idx_of[sa]:
            continue
        _assign(gap, expected[idx_of[sa] + 1:idx_of[sb]])

    # Above the first anchor and below the last.
    first_r, first_s = anchors[0]
    _assign([r for r in range(first_r) if row_cys[r] not in cy_to_slot],
            expected[:idx_of[first_s]])
    last_r, last_s = anchors[-1]
    _assign([r for r in range(last_r + 1, len(row_cys))
             if row_cys[r] not in cy_to_slot],
            expected[idx_of[last_s] + 1:])
    return out

from warp import config
# Calibration file is stored in the XDG training-data dir.
CALIBRATION_FILENAME      = 'anchors.json'
CANONICAL_LAYOUT_FILENAME = 'canonical_layout.json'
# Minimum brightness score for canonical layout to be accepted
_CANONICAL_MIN_SCORE    = 0.35

# Slot order for space builds
# Slot names must match warp_importer.py SPACE_SLOT_ORDER exactly
SPACE_SLOT_ORDER_STANDARD = [
    'Fore Weapons', 'Deflector', 'Engines', 'Warp Core', 'Shield',
    'Aft Weapons', 'Devices', 'Universal Consoles', 'Engineering Consoles',
    'Science Consoles', 'Tactical Consoles',
]
SPACE_SLOT_ORDER_CARRIER = SPACE_SLOT_ORDER_STANDARD + ['Hangars']

GROUND_SLOT_ORDER = [
    'Kit Modules', 'Kit', 'Body Armor', 'EV Suit', 'Personal Shield', 'Weapons',
    'Ground Devices',
]

SLOT_DEFAULT_COUNTS = {
    'Fore Weapons': 5, 'Deflector': 1, 'Engines': 1, 'Warp Core': 1, 'Shield': 1,
    'Aft Weapons': 4, 'Devices': 4, 'Universal Consoles': 2, 'Engineering Consoles': 4,
    'Science Consoles': 2, 'Tactical Consoles': 4, 'Hangar': 1,
    # Ground slots
    'Body Armor': 1, 'EV Suit': 1, 'Personal Shield': 1, 'Weapons': 2,
    'Kit': 1, 'Kit Modules': 6, 'Ground Devices': 3,
}

SLOT_LABEL_ALIASES = {
    'fore weapons': 'Fore Weapons', 'fore': 'Fore Weapons', 'deflector': 'Deflector',
    'impulse': 'Engines', 'engines': 'Engines', 'warp core': 'Warp Core',
    'warp': 'Warp Core', 'shields': 'Shield', 'shield': 'Shield',
    'aft weapons': 'Aft Weapons', 'aft': 'Aft Weapons', 'devices': 'Devices',
    'universal consoles': 'Universal Consoles', 'universal': 'Universal Consoles',
    'engineering consoles': 'Engineering Consoles', 'engineering': 'Engineering Consoles',
    'science consoles': 'Science Consoles', 'science': 'Science Consoles',
    'tactical consoles': 'Tactical Consoles', 'tactical': 'Tactical Consoles',
    'hangar': 'Hangar', 'hangars': 'Hangar',
    # Traits (used by full scan OCR)
    'personal space traits': 'Personal Space Traits',
    'personal traits':       'Personal Space Traits',
    'starship traits':       'Starship Traits',
    'reputation':            'Space Reputation',
    'space reputation':      'Space Reputation',
    'active space rep':      'Active Space Rep',
    'active reputation':     'Active Space Rep',
    'personal ground traits':'Personal Ground Traits',
    'ground reputation':     'Ground Reputation',
    'active ground rep':     'Active Ground Rep',
    # Boffs
    'stations':              'Boff Tactical',
    'space stations':        'Boff Tactical',
    'standard away team':    'Boff Tactical',
}

# Extend SLOT_LABEL_ALIASES with localized synonyms from
# warp/data/ui_translations.csv so the layout detector accepts OCR text from
# non-English STO clients the same way it accepts English.
def _extend_slot_aliases() -> None:
    from warp.recognition.ui_translations import normalize_map
    # space_slot: 'bug waffen' (de) → 'Fore Weapons' (canonical English),
    # then the existing English alias already maps to the same internal slot.
    for tr, canon_en in normalize_map('space_slot').items():
        # canon_en is one of 'Fore Weapons', 'Deflector', 'Shield', etc. —
        # which is exactly the right-hand side of the English aliases above.
        # We only need to add the localized OCR text as a new key.
        SLOT_LABEL_ALIASES.setdefault(tr, canon_en)
    # Same for ground slots.
    for tr, canon_en in normalize_map('ground_slot').items():
        SLOT_LABEL_ALIASES.setdefault(tr, canon_en)
    # Headers — wire the BOFF station/header German variants to the same
    # 'Boff Tactical' anchor used by the English keys above.
    from warp.recognition.ui_translations import synonyms
    for en_phrase in ('stations', 'space stations', 'standard away team'):
        for syn in synonyms('screen_header', en_phrase):
            SLOT_LABEL_ALIASES.setdefault(syn, 'Boff Tactical')


_extend_slot_aliases()


def match_slot_label(text_lower: str) -> str | None:
    """The slot an on-screen label names, or None.

    The one place that answers "is this text a slot label?". It has to be
    shared, because what the game prints is routinely *not* the slot name:
    the engines row is labelled `Impulse`, the warp core row `Warp`, the
    shield row `Shields`, and a label too wide for its column is wrapped, so
    `Engineering Consoles` arrives as `Engineering` over `Consoles`. All of
    that lives in `SLOT_LABEL_ALIASES`, together with the localized UI.

    Anything that compares OCR text to a slot name directly gets those wrong
    — measured on `image-73df6ced341f3b5f.png`, where a similarity test
    against the full name scored `'engineering'` at 0.71, below the cutoff,
    and a console row was moved 24 px off its grid as a result.
    """
    if text_lower in SLOT_LABEL_ALIASES:
        return SLOT_LABEL_ALIASES[text_lower]
    matches = get_close_matches(text_lower, list(SLOT_LABEL_ALIASES.keys()),
                                n=1, cutoff=config.LABEL_FUZZY_CUTOFF)
    return SLOT_LABEL_ALIASES.get(matches[0]) if matches else None


# Every word that appears in a label the game prints. Derived from the alias
# table, so it cannot drift from it.
_LABEL_WORDS = frozenset(
    w for key in SLOT_LABEL_ALIASES for w in key.replace('-', ' ').split()
    if len(w) >= 3)


def is_slot_label_text(text_lower: str) -> bool:
    """True when the text reads as a slot label, or one wrapped line of one.

    A weaker question than `match_slot_label`, and a different one: *which*
    slot a label names is unanswerable for `Consoles` — four slots end in it —
    but *whether* it is label text is not in doubt. That is all a caller needs
    when it is deciding whether a box has landed on writing.

    Every word has to be a word the game uses in a label, so the
    `U.S.S. FURY Traits` divider fails on `fuay` while `Consoles` passes.
    """
    words = [w for w in text_lower.replace('-', ' ').replace('.', ' ').split()
             if len(w) >= 3]
    if not words:
        return False
    return all(
        w in _LABEL_WORDS
        or get_close_matches(w, _LABEL_WORDS, n=1,
                             cutoff=config.LABEL_FUZZY_CUTOFF)
        for w in words)


# ── Full-scan constants ───────────────────────────────────────────────────────
_SCAN_CONF_MIN  = 0.45   # min ML confidence to keep a sliding-window detection
_SCAN_NMS_IOU   = 0.50   # IoU threshold for greedy NMS
_SCAN_ROW_GAP   = 0.50   # fraction of icon_est: max Y gap within a row

# Sanity caps for FullScan output. FullScan is a fallback for cases where the
# dedicated panel detectors (boff_marker, trait_grid, equipment row scan)
# failed. When it produces output well beyond any realistic STO panel
# capacity, that's a strong sign the input is not a recognisable game panel
# (e.g. a cropped UI fragment, a non-game image, or a panel the dedicated
# detectors should have handled). Better to bail than to ship hundreds of
# spurious crops to the icon matcher.
#
# Per-slot cap: 25 covers the largest realistic slot (BOFFS max=20 per
# profession category) with margin. Personal Traits cap at 10, Devices at 6.
# Per-build totals: derived from SLOT_ORDER `max` sums + boff_marker
# geometric cap (6 seats × 4 abilities = 24 for BOFF panels).
_FULLSCAN_MAX_PER_SLOT = 25
_FULLSCAN_MAX_TOTAL: dict[str, int] = {
    'SPACE':         44,
    'SPACE_EQ':      44,
    'GROUND':        15,
    'GROUND_EQ':     15,
    'SPACE_TRAITS':  27,
    'GROUND_TRAITS': 20,
    'TRAITS':        47,  # mixed = space (27) + ground (20)
    'BOFFS':         24,
    'SPACE_BOFFS':   24,
    'GROUND_BOFFS':  24,
    'SPEC':           2,
    'SPECIALIZATIONS': 2,
    'SPACE_MIXED':   97,
    'GROUND_MIXED':  61,
}
_FULLSCAN_DEFAULT_TOTAL = 100

# Equipment item type → slot name (avoids circular import from warp_importer)
_EQ_TYPE_TO_SLOT: dict[str, str] = {
    'Ship Fore Weapon': 'Fore Weapons', 'Ship Weapon': 'Fore Weapons',
    'Experimental Weapon': 'Experimental',
    'Ship Aft Weapon': 'Aft Weapons',
    'Ship Deflector Dish': 'Deflector', 'Ship Secondary Deflector': 'Sec-Def',
    'Impulse Engine': 'Engines',
    'Warp Engine': 'Warp Core', 'Singularity Engine': 'Warp Core',
    'Ship Shields': 'Shield',
    'Ship Device': 'Devices',
    'Ship Engineering Console': 'Engineering Consoles',
    'Ship Science Console': 'Science Consoles',
    'Ship Tactical Console': 'Tactical Consoles',
    'Universal Console': 'Universal Consoles',
    'Hangar Bay': 'Hangars',
    'Ground Weapon': 'Weapons',
    'Body Armor': 'Body Armor', 'EV Suit': 'EV Suit',
    'Personal Shield': 'Personal Shield',
    'Kit': 'Kit', 'Kit Module': 'Kit Modules',
    'Ground Device': 'Ground Devices',
}

# Valid equipment types per slot (mirrors warp_importer.SLOT_VALID_TYPES)
_SCAN_SLOT_VALID_TYPES: dict[str, frozenset] = {
    'Fore Weapons':           frozenset({'Ship Fore Weapon', 'Ship Weapon', 'Experimental Weapon'}),
    'Aft Weapons':            frozenset({'Ship Aft Weapon',  'Ship Weapon', 'Experimental Weapon'}),
    'Experimental':           frozenset({'Experimental Weapon'}),
    'Deflector':              frozenset({'Ship Deflector Dish', 'Ship Secondary Deflector'}),
    'Sec-Def':                frozenset({'Ship Secondary Deflector'}),
    'Engines':                frozenset({'Impulse Engine'}),
    'Warp Core':              frozenset({'Warp Engine', 'Singularity Engine'}),
    'Shield':                 frozenset({'Ship Shields'}),
    'Devices':                frozenset({'Ship Device'}),
    'Engineering Consoles':   frozenset({'Ship Engineering Console', 'Universal Console'}),
    'Science Consoles':       frozenset({'Ship Science Console',     'Universal Console'}),
    'Tactical Consoles':      frozenset({'Ship Tactical Console',    'Universal Console'}),
    'Universal Consoles':     frozenset({'Universal Console', 'Ship Tactical Console',
                                         'Ship Engineering Console', 'Ship Science Console'}),
    'Hangars':                frozenset({'Hangar Bay'}),
    'Weapons':                frozenset({'Ground Weapon'}),
    'Kit Modules':            frozenset({'Kit Module'}),
    'Kit':                    frozenset({'Kit'}),
    'Body Armor':             frozenset({'Body Armor'}),
    'EV Suit':                frozenset({'EV Suit'}),
    'Personal Shield':        frozenset({'Personal Shield'}),
    'Ground Devices':         frozenset({'Ground Device'}),
}

# Trait category key → marker string used in type scoring
_TRAIT_KEY_TO_MARKER: dict[tuple, str] = {
    ('space',  'personal'):   '__trait_space_personal',
    ('space',  'starship'):   '__trait_space_starship',
    ('space',  'rep'):        '__trait_space_rep',
    ('space',  'active_rep'): '__trait_space_active_rep',
    ('ground', 'personal'):   '__trait_ground_personal',
    ('ground', 'rep'):        '__trait_ground_rep',
    ('ground', 'active_rep'): '__trait_ground_active_rep',
}

# Slot name → trait marker (reverse of above + starship)
_TRAIT_SLOT_MARKER: dict[str, str] = {
    'Personal Space Traits':  '__trait_space_personal',
    'Starship Traits':        '__trait_space_starship',
    'Space Reputation':       '__trait_space_rep',
    'Active Space Rep':       '__trait_space_active_rep',
    'Personal Ground Traits': '__trait_ground_personal',
    'Ground Reputation':      '__trait_ground_rep',
    'Active Ground Rep':      '__trait_ground_active_rep',
}

_SPACE_TRAIT_SLOTS = frozenset({
    'Personal Space Traits', 'Starship Traits',
    'Space Reputation', 'Active Space Rep',
})
_GROUND_TRAIT_SLOTS = frozenset({
    'Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep',
})

_BOFF_SLOT_NAMES = frozenset({
    'Boff Tactical', 'Boff Engineering', 'Boff Science',
    'Boff Intelligence', 'Boff Command', 'Boff Pilot', 'Boff Miracle Worker', 'Boff Temporal',
})

_FULL_SCAN_SLOT_NAMES = (
    list(SPACE_SLOT_ORDER_STANDARD)
    + ['Hangars', 'Experimental', 'Sec-Def']
    + list(GROUND_SLOT_ORDER)
    + ['Personal Space Traits', 'Starship Traits', 'Space Reputation', 'Active Space Rep',
       'Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep']
    + sorted(_BOFF_SLOT_NAMES)
)


def _iou_1d(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _nms_boxes(dets: list, iou_thr: float = _SCAN_NMS_IOU) -> list:
    """Greedy NMS. dets = [(x,y,w,h,name,conf,...), ...]. Returns filtered list."""
    kept = []
    for det in sorted(dets, key=lambda d: -d[5]):
        x, y, w, h = det[:4]
        dominated = any(
            _iou_1d(x, x+w, k[0], k[0]+k[2]) * _iou_1d(y, y+h, k[1], k[1]+k[3]) > iou_thr
            for k in kept
        )
        if not dominated:
            kept.append(det)
    return kept


def _cluster_rows_by_y(dets: list, icon_est: int) -> list[list]:
    """Group detections into rows by Y proximity (gap < icon_est * _SCAN_ROW_GAP)."""
    gap = icon_est * _SCAN_ROW_GAP
    rows: list[list] = []
    for det in sorted(dets, key=lambda d: d[1]):
        for row in rows:
            if abs(det[1] - row[0][1]) <= gap:
                row.append(det)
                break
        else:
            rows.append([det])
    return rows


def _get_item_type(item_name: str, eq_cache: dict, trait_cache: dict,
                   starship_traits: dict, boff_cache: dict) -> str:
    """Return type marker for item_name from any of the caches."""
    for cat_items in (eq_cache or {}).values():
        entry = cat_items.get(item_name)
        if entry is not None:
            return entry.get('type', '') if isinstance(entry, dict) else ''
    for (env, cat), marker in _TRAIT_KEY_TO_MARKER.items():
        if env == 'space' and cat == 'starship':
            if item_name in (starship_traits or {}):
                return marker
        else:
            if item_name in (trait_cache or {}).get(env, {}).get(cat, {}):
                return marker
    for env in ('space', 'ground'):
        for prof_name, rank_lists in (boff_cache or {}).get(env, {}).items():
            if not isinstance(rank_lists, list):
                continue
            for rank_dict in rank_lists:
                if isinstance(rank_dict, dict) and item_name in rank_dict:
                    return f'__boff_{prof_name.lower()}'
    return ''


def _score_row_for_slot(row_types: list[str], slot_name: str,
                        ocr_labels: dict, row_cx: float, row_cy: float,
                        icon_est: int) -> float:
    """Score how well a row of icon types fits a slot (0–1)."""
    if not row_types:
        return 0.0

    if slot_name in _TRAIT_SLOT_MARKER:
        needed = _TRAIT_SLOT_MARKER[slot_name]
        type_score = sum(1 for t in row_types if t == needed) / len(row_types)
    elif slot_name in _BOFF_SLOT_NAMES:
        # 'Boff Tactical' → 'tactical', 'Boff Miracle Worker' → 'miracle worker'
        prof_key = slot_name.split(' ', 1)[1].lower()
        exact = sum(1 for t in row_types if t == f'__boff_{prof_key}')
        generic = sum(1 for t in row_types if t.startswith('__boff') and t != f'__boff_{prof_key}')
        type_score = (exact + 0.2 * generic) / len(row_types)
    else:
        valid = _SCAN_SLOT_VALID_TYPES.get(slot_name)
        if not valid:
            return 0.0
        type_score = sum(1 for t in row_types if t in valid) / len(row_types)

    if type_score < 0.1:
        return 0.0

    ocr_score = 0.0
    if slot_name in ocr_labels:
        ocx, ocy = ocr_labels[slot_name]
        dist = ((ocx - row_cx) ** 2 + (ocy - row_cy) ** 2) ** 0.5
        if dist <= icon_est * 3:
            ocr_score = 1.0
        elif dist <= icon_est * 7:
            ocr_score = 0.5

    return 0.65 * type_score + 0.35 * ocr_score


class LayoutDetector:
    """
    Detect icon bounding boxes for all slots in an STO screenshot.
    Learns new layouts automatically from confirmed annotations.
    """

    def __init__(self):
        self._ocr = None
        # Tokens for the screenshot `detect()` is working on, set there. None
        # outside a detect() call, in which case the space panel reads for
        # itself — the standalone behaviour the dev probes rely on.
        self._ocr_tokens: list[dict] | None = None
        self._calibration = self._load_calibration()
        self._community_anchors: list | None = None  # instance cache for community_anchors.json (P11)
        # Per-detect()-call cached EQ geometry result (keyed by id(img)).
        # Shared across multiple callers within one detect() call so they
        # reuse a single OCR run. CLEARED at the top of detect() — id(img)
        # is not stable across calls (Python reuses ids after GC), so
        # persisting entries across detect() calls causes false cache hits
        # and non-deterministic results when LayoutDetector instances are
        # reused for batch processing.
        # Keyed by pixel content, not by `id()`, so a second `detect()` on the
        # same screenshot reuses the geometry instead of recomputing it. The
        # importer calls `detect()` twice whenever pixel counts refine the ship
        # profile, and the panel geometry does not depend on the profile.
        self._eq_geom_cache: dict[str, EQGeometry | None] = {}
        self._ground_eq_geom_cache: dict[str, GroundEQGeometry | None] = {}
        self._ocr_labels_cache: dict[str, dict] = {}
        self._img_key_memo: tuple = (None, '')
        # slot → icons actually counted in that row by pixel analysis, for
        # the last detect() call. A row's bboxes come from the ship profile,
        # not from this count, so it is otherwise discarded — but it is the
        # only measurement of what the screenshot really contains, and
        # warp_importer uses it to recover the T6-X/X2 bonus when the tier
        # badge is not on screen. Reset per detect() call.
        self.last_row_pixel_counts: dict[str, int] = {}

    def _img_key(self, img: np.ndarray) -> str:
        """A key for per-image caches that survives repeated `detect()` calls.

        `id()` will not do: CPython reuses ids after a garbage collection, so
        a stale entry could answer for a different image. Hashing the pixels
        cannot collide that way. It costs ~6 ms on a 14 MB screenshot, against
        the ~6.4 s of the geometry pass it lets us skip, and the result is
        memoised per array object so a single `detect()` hashes once.
        """
        ident = (id(img), img.shape, img.strides)
        if self._img_key_memo[0] == ident:
            return self._img_key_memo[1]
        import hashlib
        digest = hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()
        self._img_key_memo = (ident, digest)
        return digest

    def _get_eq_geometry(self, img: np.ndarray) -> EQGeometry | None:
        """Cached wrapper around detect_eq_geometry. Returns None when OCR
        yields no usable EQ labels (e.g. BOFF-only or trait-only screen)."""
        key = self._img_key(img)
        if key in self._eq_geom_cache:
            return self._eq_geom_cache[key]
        try:
            geom = detect_eq_geometry(
                img, ocr_tokens=getattr(self, '_ocr_tokens', None))
        except Exception as e:
            _slog.warning(f'LayoutDetector: detect_eq_geometry crashed: {e}')
            geom = None
        self._eq_geom_cache[key] = geom
        if geom is not None:
            _slog.info(
                f'LayoutDetector: eq_geometry mode={geom.mode} '
                f'p_start={geom.panel_x_start} p_right={geom.panel_right} '
                f'dx={geom.final_dx:.1f} pitch={geom.row_pitch} '
                f'rows={len(geom.row_cys)}')
        return geom

    def _get_ground_eq_geometry(self, img: np.ndarray,
                               ocr_tokens: list[dict] | None = None,
                               ) -> GroundEQGeometry | None:
        """Cached wrapper around detect_ground_eq_geometry. Returns None when
        OCR yields fewer than 2 left-column EQ labels."""
        key = self._img_key(img)
        if key in self._ground_eq_geom_cache:
            return self._ground_eq_geom_cache[key]
        try:
            geom = detect_ground_eq_geometry(img, ocr_tokens=ocr_tokens)
        except Exception as e:
            _slog.warning(f'LayoutDetector: detect_ground_eq_geometry crashed: {e}')
            geom = None
        self._ground_eq_geom_cache[key] = geom
        if geom is not None:
            _slog.info(
                f'LayoutDetector: ground_eq_geometry mode={geom.mode} '
                f'col_left={geom.col_left_x} col_right={geom.col_right_x} '
                f'pitch={geom.row_pitch} cell={geom.cell_w:.1f}x{geom.cell_h:.1f} '
                f'slots={list(geom.slot_label_cys.keys())}')
        return geom

    def _detect_via_ground_geometry(
        self, img: np.ndarray, profile: dict,
        ocr_tokens: list[dict] | None = None,
    ) -> dict[str, list[tuple[int, int, int, int]]] | None:
        """Ground EQ detection via OCR-anchored geometry.

        Projects cells from `ground_eq_geometry.project_cells`, then trims
        Kit Modules / Ground Devices counts against the ship profile
        (profile is the floor for KM; cap Devices to profile when projected
        rows exceed). Returns None when geometry detection failed.
        """
        geom = self._get_ground_eq_geometry(img, ocr_tokens=ocr_tokens)
        if geom is None:
            return None
        h, w = img.shape[:2]
        projected = _project_ground_cells(geom, w, h)
        if not projected:
            return None

        # Kit Modules: profile floors the count. project_cells already caps at
        # KM_MAX_CELLS=7 and at col_left_x boundary. Trim down to profile
        # count when projection over-counts.
        km_profile = profile.get(_G_KM, SLOT_DEFAULT_COUNTS.get(_G_KM, 6))
        if _G_KM in projected and km_profile > 0:
            projected[_G_KM] = projected[_G_KM][:max(km_profile, 1)]

        # Ground Devices: project_cells emits up to 3 rows × 2 cols = 6 cells.
        # Do NOT trim to profile — the trainer needs to see every grid-aligned
        # position so empty/missed slots are reviewable (matcher will tag
        # blank cells as __empty__). Profile lower than 6 is just informational.

        # Drop slots with empty bbox lists (defensive — _project should not
        # emit them, but a future change might).
        return {k: v for k, v in projected.items() if v}

    def detect(self, img: np.ndarray, build_type: str, ship_profile: dict | None = None,
               icon_matcher=None, app_cache=None,
               ocr_tokens: list[dict] | None = None,
               ) -> dict[str, list[tuple[int, int, int, int]]]:
        """Detect every slot's boxes, then take them off the labels.

        A thin wrapper so the rule holds for whichever of the strategies below
        answered — there are twenty-two ways out of `_detect_raw`, and a guard
        applied at one of them is a guard that does not exist. See
        `drop_boxes_on_text`.
        """
        result = self._detect_raw(img, build_type, ship_profile,
                                  icon_matcher, app_cache, ocr_tokens)
        return drop_boxes_on_text(result, ocr_tokens, img)

    def _detect_raw(self, img: np.ndarray, build_type: str, ship_profile: dict | None = None,
                    icon_matcher=None, app_cache=None,
                    ocr_tokens: list[dict] | None = None,
                    ) -> dict[str, list[tuple[int, int, int, int]]]:
        # The geometry caches deliberately survive this call — they are keyed
        # on pixel content (`_img_key`), so they cannot answer for a different
        # image, and the importer re-enters `detect()` on the same screenshot
        # once the pixel counts refine the ship profile. Bounded to a handful
        # of screenshots so a folder run does not accumulate them.
        for _cache in (self._eq_geom_cache, self._ground_eq_geom_cache,
                       self._ocr_labels_cache):
            while len(_cache) > 3:
                _cache.pop(next(iter(_cache)))
        # Held for this call so `_get_eq_geometry` can hand them to the space
        # panel without threading a parameter through its five call sites. The
        # ground panel already receives them directly; this is what makes both
        # read the same tokens instead of the space one opening the screenshot
        # a second time.
        self._ocr_tokens = ocr_tokens
        self.last_row_pixel_counts = {}
        if build_type in ('TRAITS', 'SPACE_TRAITS', 'GROUND_TRAITS'):
            # Strategy 0: structure-driven trait grid detector with ML probe.
            # Multi-panel grid lock + multi-chain row extraction + per-group
            # ML classification (no canonical-order assumption). Prototype
            # measured 91.5% slot IoU≥30 on 59 GT screens vs OCR-header
            # baseline. Falls back to OCR-header strategy on failure.
            if icon_matcher is not None and app_cache is not None:
                grid = _trait_grid.detect_traits(img, icon_matcher, app_cache,
                                                 build_type=build_type)
                if grid and sum(len(v) for v in grid.values()) >= 5:
                    _slog.info(
                        f'LayoutDetector: Strategy 0 (trait_grid) → '
                        f'{len(grid)} sections, '
                        f'{sum(len(v) for v in grid.values())} bboxes')
                    return grid
            return self._detect_traits(img, build_type)
        if build_type in ('BOFFS', 'SPACE_BOFFS', 'GROUND_BOFFS'):
            # Strategy 0: marker-panel detector (HSV badges + RANSAC grid +
            # bible-driven slot projection). 100% panel anchor on 36 GT
            # screens; 96.0% slot IoU≥.70.
            marker = self._detect_boffs_via_markers(img)
            if marker:
                return marker
            if icon_matcher is not None and app_cache is not None:
                full = self._detect_via_full_scan(img, build_type, icon_matcher, app_cache)
                if full and len(full) >= 2:
                    return full
            # No structural fallback. The legacy `_detect_boffs` brightness/
            # V-profile path produced false positives on ~140 STO_screens
            # that had no real BOFF panel — it was masking gaps in marker
            # detection. With Strategy 0 now covering all 247 real-panel
            # cases, returning {} is the honest answer.
            return {}
        if build_type == 'SPEC':
            return {}

        profile = ship_profile or {}
        if build_type == 'GROUND':
            slot_order = GROUND_SLOT_ORDER
        else:
            slot_order = (SPACE_SLOT_ORDER_CARRIER if profile.get('Hangars', 0) > 0 else SPACE_SLOT_ORDER_STANDARD)

        # GROUND Strategy 1: OCR-anchored ground EQ geometry. Single source of
        # truth for the 7-slot ground panel (Kit Modules + 2-col block +
        # Weapons stack + Devices grid). 94.4% recall, 91.2% precision,
        # mean IoU 0.814 on 12 GT-annotated screens.
        if build_type == 'GROUND':
            ground = self._detect_via_ground_geometry(img, profile,
                                                        ocr_tokens=ocr_tokens)
            if ground and len(ground) >= 3:
                _slog.info(
                    f'LayoutDetector: GROUND Strategy 1 (ground_eq_geometry) → '
                    f'{len(ground)} slot groups, '
                    f'{sum(len(v) for v in ground.values())} bboxes')
                for slot, boxes in ground.items():
                    for b in boxes:
                        _slog.info(f'  [{slot}] bbox={b}')
                return ground

        # MIXED detection chain: learned → OCR-anchored → full_scan → fallback
        if build_type in ('SPACE_MIXED', 'GROUND_MIXED'):
            marker_boffs = self._detect_boffs_via_markers(img)

            # Strategy 0 trait grid: structure-driven trait detector. Run once
            # for MIXED screens; merged into whichever equipment chain wins.
            trait_grid_res: dict | None = None
            if icon_matcher is not None and app_cache is not None:
                tg = _trait_grid.detect_traits(img, icon_matcher, app_cache,
                                               build_type=build_type)
                if tg and sum(len(v) for v in tg.values()) >= 5:
                    trait_grid_res = tg

            # BOFF panel guard: trait_grid locks consistent-spacing icon rows,
            # which BOFF abilities also produce — same icon size, same dx.
            # If the marker panel anchored a BOFF region, any trait_grid bbox
            # whose center falls inside it is a false positive and must drop.
            boff_panel_box: tuple[int, int, int, int] | None = None
            if marker_boffs:
                _all = [b for bxs in marker_boffs.values() for b in bxs]
                if _all:
                    # Inflate left by ~1 icon_w (seat-letter / portrait CCs
                    # sitting just left of the markers) and down by ~1 icon_h
                    # (ability-icon row immediately below each slot row, which
                    # trait_grid otherwise grabs as a phantom panel).
                    _ws = sorted(b[2] for b in _all)
                    _hs = sorted(b[3] for b in _all)
                    _iw = int(_ws[len(_ws) // 2])
                    _ih = int(_hs[len(_hs) // 2])
                    boff_panel_box = (
                        min(b[0] for b in _all) - _iw,
                        min(b[1] for b in _all),
                        max(b[0] + b[2] for b in _all),
                        max(b[1] + b[3] for b in _all) + _ih,
                    )

            def _in_boff_panel(bbox):
                if not boff_panel_box:
                    return False
                cx = bbox[0] + bbox[2] // 2
                cy = bbox[1] + bbox[3] // 2
                x0, y0, x1, y1 = boff_panel_box
                return x0 <= cx <= x1 and y0 <= cy <= y1

            def _merge_traits(result):
                return merge_trait_boxes(result, trait_grid_res,
                                         _in_boff_panel)


            # GROUND_MIXED Strategy 1: ground EQ geometry + traits + BOFFs.
            # Ground panel uses different OCR anchors than space, so the
            # space _get_eq_geometry path below would miss the entire EQ
            # grid on ground screens.
            if build_type == 'GROUND_MIXED':
                ground_eq = self._detect_via_ground_geometry(img, profile,
                                                              ocr_tokens=ocr_tokens)
                if ground_eq and len(ground_eq) >= 3:
                    g_geom = self._get_ground_eq_geometry(img, ocr_tokens=ocr_tokens)
                    if g_geom is not None:
                        labels = self._ocr_section_labels(img)
                        # GROUND_MIXED: drop any phantom space-trait OCR hits
                        # so the trait grid cannot anchor on "Starship Traits".
                        trait_labels = {s: v for s, v in labels.items()
                                        if s in _GROUND_TRAIT_SLOTS}
                        cell_w = max(20, int(round(g_geom.cell_w)))
                        icon_h = max(20, int(round(g_geom.cell_h)))
                        ground_eq.update(
                            self._detect_traits_via_ocr(
                                img, trait_labels, cell_w, icon_h))
                    if marker_boffs:
                        ground_eq.update(marker_boffs)
                    _slog.info(
                        f'LayoutDetector: GROUND_MIXED Strategy 1 '
                        f'(ground_eq_geometry + OCR traits) → '
                        f'{len(ground_eq)} slot groups, '
                        f'{sum(len(v) for v in ground_eq.values())} bboxes')
                    return _merge_traits(ground_eq)

            # Strategy 1: EQ via geom-based pixel_analysis + traits via OCR
            # + BOFFs via marker/in_mixed. One EQ source of truth shared with
            # SPACE_EQ/GROUND_EQ (no divergent _detect_via_ocr_anchored grid).
            geom = self._get_eq_geometry(img)
            if geom is not None and geom.row_cys:
                eq_result = self._detect_via_pixel_analysis(img, slot_order, profile)
                if eq_result and len(eq_result) >= 3:
                    # Traits: OCR section labels → 5-column grid via
                    # _detect_traits_via_ocr. Cell geometry mirrors the
                    # eq_geometry values used for EQ above.
                    # GROUND_MIXED returns earlier; here build_type is
                    # SPACE_MIXED so drop any phantom ground-trait OCR hits.
                    labels = self._ocr_section_labels(img)
                    trait_labels = {s: v for s, v in labels.items()
                                    if s in _SPACE_TRAIT_SLOTS}
                    cell_w = max(20, int(round(geom.final_dx)))
                    icon_h = max(20, int(round(geom.row_pitch * 0.85)) + 2)
                    eq_result.update(
                        self._detect_traits_via_ocr(img, trait_labels, cell_w, icon_h))

                    if marker_boffs:
                        eq_result.update(marker_boffs)
                    _slog.info(f'LayoutDetector: Strategy 1 (geom/pixel + OCR traits) → '
                               f'{len(eq_result)} slot groups, '
                               f'{sum(len(v) for v in eq_result.values())} bboxes')
                    return _merge_traits(eq_result)

            # Strategy 1a fallback: legacy OCR-anchored path (geom unavailable
            # or pixel_analysis produced too few rows). Retains independent
            # cluster-based right-edge logic.
            ocr_anch = self._detect_via_ocr_anchored(img, build_type, slot_order, profile)
            if ocr_anch and len(ocr_anch) >= 3:
                if marker_boffs:
                    ocr_anch.update(marker_boffs)
                _slog.info(f'LayoutDetector: Strategy 1a (OCR-anchored fallback) → '
                           f'{len(ocr_anch)} slot groups, '
                           f'{sum(len(v) for v in ocr_anch.values())} bboxes')
                return _merge_traits(ocr_anch)

            # Strategy 1b: learned layouts — fallback only when geom failed.
            learned = self._detect_via_learned_layouts(img, build_type, slot_order, profile)
            if learned and len(learned) >= 5:
                if marker_boffs:
                    learned.update(marker_boffs)
                _slog.info(f'LayoutDetector: Strategy 1b (learned/MIXED fallback) → '
                           f'{len(learned)} slot groups')
                return _merge_traits(learned)

            if icon_matcher is not None and app_cache is not None:
                full = self._detect_via_full_scan(img, build_type, icon_matcher, app_cache)
                if full and len(full) >= 3:
                    has_boff = any(k.startswith('Boff ') for k in full)
                    if not has_boff and marker_boffs:
                        full.update(marker_boffs)
                        _slog.info(f'LayoutDetector: MIXED merged → {len(full)} slot groups total')
                    return _merge_traits(full)

            # All EQ strategies failed. BOFF/trait detection is independent of
            # EQ — user may have pasted a MIXED screen with partial content.
            # Emit whatever boffs/traits we found rather than dropping them.
            if marker_boffs or trait_grid_res:
                result: dict[str, list] = {}
                if marker_boffs:
                    result.update(marker_boffs)
                merged = _merge_traits(result)
                _slog.info(
                    f'LayoutDetector: MIXED (no EQ anchor) → '
                    f'{sum(1 for k in merged if k.startswith("Boff "))} boff seats, '
                    f'{sum(1 for k in merged if not k.startswith("Boff "))} trait groups')
                return merged
            # Fall through to standard strategies if both OCR-anchored and full scan fail

        # Strategy 1: pixel analysis backed by eq_geometry. On 38 GT screens
        # geom delivers pixel-perfect grid (max panel_right Δ=7 px, max dx
        # Δ=2 px, 100 % slot coverage). Promoted ahead of learned layouts
        # which had 32 px mean panel_right error and returned None for 4
        # screens on the same benchmark. Learned moves to Strategy 1b as a
        # safety net when geometry produces no usable result.
        geom_available = self._get_eq_geometry(img) is not None
        if geom_available:
            result = self._detect_via_pixel_analysis(img, slot_order, profile)
            if result and len(result) >= max(3, int(len(slot_order) * 0.7)):
                # Supplement missing optional slots (Hangars, Universal
                # Consoles, Sec-Def, Experimental) from learned layout.
                missing = [s for s in slot_order
                           if s not in result and profile.get(s, 0) > 0]
                if missing:
                    learned = self._detect_via_learned_layouts(
                        img, build_type, slot_order, profile)
                    if learned:
                        for slot in missing:
                            if learned.get(slot):
                                result[slot] = learned[slot]
                                _slog.info(f'LayoutDetector: Strategy 1 supplement '
                                           f'[{slot}] from learned ({len(learned[slot])} bboxes)')
                _slog.info(f'LayoutDetector: Strategy 1 (geom/pixel) → '
                           f'{len(result)} slot groups, '
                           f'{sum(len(v) for v in result.values())} bboxes')
                for slot, boxes in result.items():
                    for b in boxes:
                        _slog.info(f'  [{slot}] bbox={b}')
                return result

        # Strategy 1b: learned layouts — fallback when geometry unavailable
        # or under-covered the screen.
        learned = self._detect_via_learned_layouts(img, build_type, slot_order, profile)
        if learned:
            missing = [s for s in slot_order if s not in learned and profile.get(s, 0) > 0]
            if missing:
                pixel = self._detect_via_pixel_analysis(img, slot_order, profile)
                for slot in missing:
                    if pixel.get(slot):
                        learned[slot] = pixel[slot]
                        _slog.info(f'LayoutDetector: Strategy 1b supplement [{slot}] '
                                   f'from pixel analysis ({len(pixel[slot])} bboxes)')
            _slog.info(f'LayoutDetector: Strategy 1b (learned fallback) → '
                       f'{len(learned)} slot groups, '
                       f'{sum(len(v) for v in learned.values())} bboxes')
            for slot, boxes in learned.items():
                for b in boxes:
                    _slog.info(f'  [{slot}] bbox={b}')
            return learned

        # Strategy 2: pixel analysis without geometry (legacy brightness path).
        result = self._detect_via_pixel_analysis(img, slot_order, profile)
        if result and len(result) >= len(slot_order) * 0.7:
            _slog.info(f'LayoutDetector: Strategy 2 (pixel legacy) → '
                       f'{len(result)} slot groups, '
                       f'{sum(len(v) for v in result.values())} bboxes')
            for slot, boxes in result.items():
                for b in boxes:
                    _slog.info(f'  [{slot}] bbox={b}')
            return result

        # Strategy 2.5: Canonical layout + Y-offset scan
        # Uses aggregate learned Y positions when pixel analysis under-covers the screen
        canonical = self._detect_via_canonical_layout(img, build_type, slot_order, profile)
        if canonical and len(canonical) >= max(3, int(len(slot_order) * 0.6)):
            _slog.info(f'LayoutDetector: Strategy 2.5 (canonical) → {len(canonical)} slot groups, {sum(len(v) for v in canonical.values())} bboxes')
            for slot, boxes in canonical.items():
                for b in boxes:
                    _slog.info(f'  [{slot}] bbox={b}')
            return canonical

        # Strategy 3: OCR labels
        ocr_result = self._detect_via_ocr(img, slot_order, profile)
        if ocr_result and len(ocr_result) >= 2:
            filled = self._fill_gaps(ocr_result, slot_order, img, profile)
            _slog.info(f'LayoutDetector: Strategy 3 (OCR) → {len(filled)} slot groups, {sum(len(v) for v in filled.values())} bboxes')
            for slot, boxes in filled.items():
                for b in boxes:
                    _slog.info(f'  [{slot}] bbox={b}')
            return filled

        # Strategy 4: Anchor fallback — uses canonical learned values if available,
        # otherwise falls back to hardcoded SPACE_ANCHORS_REL
        fallback = self._detect_via_anchors(img, slot_order, profile)
        _slog.info(f'LayoutDetector: Strategy 4 (anchors) → {len(fallback)} slot groups, {sum(len(v) for v in fallback.values())} bboxes')
        for slot, boxes in fallback.items():
            for b in boxes:
                _slog.info(f'  [{slot}] bbox={b}')
        return fallback
    # ── Learning Logic ────────────────────────────────────────────────────────

    def remove_layout(self, source_file: str) -> bool:
        """Remove all learned layout entries for source_file from anchors.json."""
        if not source_file or not self._calibration or 'learned' not in self._calibration:
            return False
        before = len(self._calibration['learned'])
        self._calibration['learned'] = [
            e for e in self._calibration['learned']
            if e.get('source_file') != source_file
        ]
        removed = before - len(self._calibration['learned'])
        if removed:
            self._save_calibration()
            _slog.info(f'LayoutDetector: removed {removed} layout entries for {source_file!r}')
            return True
        return False

    # ── Canonical Layout (aggregate from anchors.json) ────────────────────────

    @classmethod
    def build_canonical_layout(cls) -> dict:
        """
        Aggregate all learned entries from anchors.json into canonical_layout.json.
        Computes median Y/W/H per slot per screen type.
        Called after learn_layout() and as a one-time bootstrap.
        Returns the canonical dict (empty on failure).
        """
        import statistics as _st

        cfile = _userdata.training_data_dir() / CALIBRATION_FILENAME
        cal_data = None
        if cfile.exists():
            try:
                cal_data = json.loads(cfile.read_text(encoding='utf-8'))
            except Exception:
                cal_data = None
        if not cal_data:
            return {}

        learned = cal_data.get('learned', [])
        if not learned:
            return {}

        from collections import defaultdict
        type_slots: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {'y': [], 'w': [], 'h': []}))

        for entry in learned:
            btype = entry.get('type')
            if not btype:
                continue
            for slot_name, geo in entry.get('slots', {}).items():
                if not isinstance(geo, dict):
                    continue
                sd = type_slots[btype][slot_name]
                if 'y_rel' in geo:
                    sd['y'].append(geo['y_rel'])
                if 'w_rel' in geo:
                    sd['w'].append(geo['w_rel'])
                if 'h_rel' in geo:
                    sd['h'].append(geo['h_rel'])

        canonical: dict = {'version': 1, 'types': {}}
        for btype, slot_data in type_slots.items():
            slots_out = {}
            for slot_name, vals in slot_data.items():
                if len(vals['y']) < 2:
                    continue
                slots_out[slot_name] = {
                    'y_rel': round(_st.median(vals['y']), 5),
                    'y_std': round(_st.stdev(vals['y']), 5),
                    'w_rel': round(_st.median(vals['w']), 5) if vals['w'] else 0.028,
                    'h_rel': round(_st.median(vals['h']), 5) if vals['h'] else 0.055,
                    'n':     len(vals['y']),
                }
            if slots_out:
                canonical['types'][btype] = {
                    'n_samples': sum(1 for e in learned if e.get('type') == btype),
                    'slots':     slots_out,
                }

        # Save next to anchors.json
        out = _userdata.training_data_dir() / CANONICAL_LAYOUT_FILENAME
        out.write_text(json.dumps(canonical, indent=2), encoding='utf-8')
        _slog.info(
            f'LayoutDetector: canonical_layout.json saved '
            f'({len(canonical["types"])} types, '
            f'{sum(len(v["slots"]) for v in canonical["types"].values())} slot entries)'
        )

        return canonical

    def _load_canonical_layout(self) -> dict | None:
        """Load canonical_layout.json. Returns None if missing/corrupt."""
        cfile = _userdata.training_data_dir() / CANONICAL_LAYOUT_FILENAME
        if cfile.exists():
            try:
                return json.loads(cfile.read_text(encoding='utf-8'))
            except Exception:
                return None
        return None

    def _detect_via_canonical_layout(
        self,
        img,
        build_type: str,
        slot_order: list,
        profile: dict,
    ) -> dict | None:
        """
        Strategy 2.5: canonical layout + vertical offset scan.

        Loads the aggregate canonical Y positions (median across all learned entries),
        then searches for the best Y offset by scoring pixel brightness at predicted
        icon rows. Generates bboxes from the best-fit offset.

        Triggered when pixel analysis produces < 70% coverage.
        """
        canonical = self._load_canonical_layout()
        if not canonical:
            return None
        type_data = canonical.get('types', {}).get(build_type)
        if not type_data or not type_data.get('slots'):
            return None

        can_slots: dict = type_data['slots']

        import cv2
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        panel_right = self._find_panel_right_edge(img)

        # High-coverage "anchor" slots used for brightness scoring
        ANCHOR_CHECK = {
            'Fore Weapons', 'Deflector', 'Engines', 'Warp Core', 'Shield', 'Aft Weapons',
            'Boff Engineering', 'Boff Science', 'Boff Tactical',
            'Personal Space Traits', 'Space Reputation', 'Starship Traits',
            'Primary Specialization', 'Secondary Specialization',
            'Kit Modules', 'Kit', 'Body Armor', 'Personal Shield',
        }

        def _score(dy: float) -> float:
            sc = ck = 0
            for slot_name, geo in can_slots.items():
                if slot_name not in ANCHOR_CHECK:
                    continue
                cy = int((geo['y_rel'] + dy) * h)
                bw = max(18, int(geo['w_rel'] * w))
                bh = max(18, int(geo['h_rel'] * h))
                y1 = max(0, cy - bh // 4)
                y2 = min(h, cy + bh // 4)
                if y1 >= y2:
                    continue
                # Check at panel right edge (STO icons are right-aligned)
                x0 = max(0, panel_right - 5 * bw)
                patch = gray[y1:y2, x0:panel_right]
                if patch.size == 0:
                    continue
                ck += 1
                if float(patch.mean()) > 40:
                    sc += 1
            return sc / max(ck, 1)

        # Clamp dy range: don't allow shifts that push any slot above the image top
        min_y_rel = min((geo['y_rel'] for geo in can_slots.values()), default=0.0)

        # Scan Y offsets -0.20 … +0.20 in 0.01 steps
        best_dy, best_score = 0.0, _score(0.0)
        for dy_i in range(-20, 21):
            if dy_i == 0:
                continue
            dy = dy_i / 100.0
            if min_y_rel + dy < 0:  # would push topmost slot above image top
                continue
            s = _score(dy)
            if s > best_score:
                best_score, best_dy = s, dy

        if best_score < _CANONICAL_MIN_SCORE:
            _slog.debug(
                f'LayoutDetector: canonical [{build_type}] score={best_score:.2f} '
                f'< {_CANONICAL_MIN_SCORE} — skipping'
            )
            return None

        _slog.info(
            f'LayoutDetector: Strategy 2.5 (canonical) [{build_type}] '
            f'dy={best_dy:+.2f} score={best_score:.2f}'
        )

        row_h_est = int(h * 0.068)
        cell_w    = max(30, int(row_h_est * 0.80))

        result = {}
        for slot_name in slot_order:
            geo = can_slots.get(slot_name)
            if geo is None:
                continue
            cy   = int((geo['y_rel'] + best_dy) * h)
            bh   = max(26, int(geo['h_rel'] * h))
            iy   = max(0, cy - bh // 2)
            n    = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            if n == 0:
                continue
            bboxes = []
            for j in range(n):
                bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, cell_w - 4, bh))
            bboxes.reverse()
            result[slot_name] = bboxes

        return result if len(result) >= 3 else None

    def learn_layout(self, screen_type: str, img_size: tuple[int, int], annotations: list[dict], source_file: str = ''):
        """
        Record a confirmed layout to anchors.json.

        Stores full relative geometry per slot:
          x0_rel   — leftmost icon X / image width
          y_rel    — icon row center Y / image height
          w_rel    — icon width / image width
          h_rel    — icon height / image height
          step_rel — X step between consecutive icons / image width
          count    — number of icons stored (for this ship)

        All values are relative so the layout scales correctly to different
        window sizes and resolutions without any estimation.
        """
        if not annotations: return
        h, w = img_size
        aspect = round(w / h, 3)

        # Group annotations by slot, keeping only confirmed ones
        from collections import defaultdict
        slot_bboxes: dict[str, list] = defaultdict(list)
        for ann in annotations:
            bbox = ann.get('bbox')
            slot = ann.get('slot')
            if not bbox or not slot:
                continue
            slot_bboxes[slot].append(bbox)

        if not slot_bboxes:
            return

        GAP_FACTOR = 2.5   # gap > 2.5× median step = new column (e.g. split Boff Tactical)
        slot_map = {}
        for slot, bboxes in slot_bboxes.items():
            # Sort left-to-right
            bboxes_s = sorted(bboxes, key=lambda b: b[0])
            bw = int(round(sum(b[2] for b in bboxes_s) / len(bboxes_s)))
            bh = int(round(sum(b[3] for b in bboxes_s) / len(bboxes_s)))
            cy = int(round(sum(b[1] + b[3] / 2 for b in bboxes_s) / len(bboxes_s)))

            if len(bboxes_s) == 1:
                step = bw + max(2, int(bw * 0.08))
                slot_map[slot] = {
                    'x0_rel':   round(bboxes_s[0][0] / w, 5),
                    'y_rel':    round(cy / h, 5),
                    'w_rel':    round(bw / w, 5),
                    'h_rel':    round(bh / h, 5),
                    'step_rel': round(step / w, 5),
                    'count':    1,
                }
                continue

            steps = [bboxes_s[i+1][0] - bboxes_s[i][0] for i in range(len(bboxes_s) - 1)]
            median_step = sorted(steps)[len(steps) // 2]
            split_indices = [i for i, s in enumerate(steps) if s > GAP_FACTOR * median_step]

            if not split_indices:
                # Single contiguous run — use flat format (backward compatible)
                step = int(round(sum(steps) / len(steps)))
                slot_map[slot] = {
                    'x0_rel':   round(bboxes_s[0][0] / w, 5),
                    'y_rel':    round(cy / h, 5),
                    'w_rel':    round(bw / w, 5),
                    'h_rel':    round(bh / h, 5),
                    'step_rel': round(step / w, 5),
                    'count':    len(bboxes_s),
                }
            else:
                # Multiple columns (e.g. same Boff profession in left+right column)
                runs = []
                prev = 0
                for si in split_indices + [len(bboxes_s) - 1]:
                    chunk = bboxes_s[prev:si + 1]
                    chunk_steps = [chunk[j+1][0] - chunk[j][0] for j in range(len(chunk) - 1)]
                    chunk_step = int(round(sum(chunk_steps) / len(chunk_steps))) if chunk_steps else (bw + max(2, int(bw * 0.08)))
                    runs.append({
                        'x0_rel':   round(chunk[0][0] / w, 5),
                        'step_rel': round(chunk_step / w, 5),
                        'count':    len(chunk),
                    })
                    prev = si + 1
                slot_map[slot] = {
                    'y_rel': round(cy / h, 5),
                    'w_rel': round(bw / w, 5),
                    'h_rel': round(bh / h, 5),
                    'runs':  runs,
                }
                _slog.info(f'LayoutDetector: learn_layout [{slot}] split into {len(runs)} runs: {[(r["count"], round(r["x0_rel"],3)) for r in runs]}')

        if not slot_map:
            return

        if not self._calibration:
            self._calibration = {}
        if 'learned' not in self._calibration:
            self._calibration['learned'] = []

        entry = {
            'type':        screen_type,
            'aspect':      aspect,
            'slots':       slot_map,
            'res':         f'{w}x{h}',
            'timestamp':   int(__import__('time').time()),
            'source_file': source_file,
        }

        # Avoid exact duplicates
        total = len(self._calibration['learned'])
        for existing in self._calibration['learned']:
            if (existing['type'] == screen_type
                    and existing['res'] == entry['res']
                    and existing['slots'] == slot_map):
                _slog.debug(f'LayoutDetector: learn_layout {screen_type} {w}x{h} — duplicate, skipping')
                return

        self._calibration['learned'].append(entry)

        # P3: LRU cap — keep at most 200 entries, evict oldest
        MAX_LEARNED = 200
        if len(self._calibration['learned']) > MAX_LEARNED:
            self._calibration['learned'] = self._calibration['learned'][-MAX_LEARNED:]
            _slog.info(f'LayoutDetector: LRU eviction — trimmed to {MAX_LEARNED} entries')

        self._save_calibration()
        total_bboxes = sum(
            v['count'] if 'count' in v else sum(r['count'] for r in v.get('runs', []))
            for v in slot_map.values()
        )
        _slog.info(
            f'LayoutDetector: saved layout [{screen_type}] {w}x{h} '
            f'({len(slot_map)} slot groups, {total_bboxes} bboxes'
            + (f', src={source_file}' if source_file else '')
            + f', total entries={len(self._calibration["learned"])})'
        )
        # Rebuild canonical layout so Strategy 2.5 benefits from new data
        try:
            LayoutDetector.build_canonical_layout()
        except Exception as _ce:
            _slog.debug(f'LayoutDetector: canonical rebuild failed: {_ce}')

    def _detect_via_learned_layouts(self, img, build_type, slot_order, profile):
        """Find the best matching learned layout by scoring pixel brightness.

        P3 improvement: instead of blindly picking the most recent layout,
        score each candidate by checking whether bright pixels (icons) exist
        at the predicted slot positions.  The layout whose predicted positions
        best match actual icon regions in the image wins.
        """
        if not self._calibration or 'learned' not in self._calibration:
            return None

        h, w = img.shape[:2]
        aspect = round(w / h, 3)

        # Filter by screen type and similar aspect ratio
        candidates = [e for e in self._calibration['learned']
                      if e['type'] == build_type and abs(e['aspect'] - aspect) < 0.05]

        if not candidates:
            # Strategy 1b: try community anchors (P11)
            community = self._load_community_anchors()
            candidates = [e for e in community
                          if e.get('type') == build_type and abs(e.get('aspect', 0) - aspect) < 0.05]
            if not candidates:
                return None

        # ── Score each candidate by pixel brightness at predicted Y rows ─────
        # Convert to grayscale once for fast brightness sampling
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        best_score = -1
        best_entry = None

        for entry in candidates:
            score = 0
            checked = 0
            for slot_name, geo in entry['slots'].items():
                if isinstance(geo, (int, float)):
                    continue  # old-format entry
                bw = max(1, int(geo['w_rel'] * w))
                bh = max(1, int(geo['h_rel'] * h))
                cy = int(geo['y_rel'] * h)
                y1 = max(0, cy - bh // 4)
                y2 = min(h, cy + bh // 4)
                # Normalise to runs — handles both flat and multi-run formats
                runs = geo.get('runs') or [{'x0_rel': geo['x0_rel'], 'step_rel': geo['step_rel'], 'count': geo.get('count', 1)}]
                for run in runs:
                    x0   = int(run['x0_rel'] * w)
                    step = max(bw, int(run['step_rel'] * w))
                    for j in range(min(run['count'], 8)):
                        ix  = x0 + j * step
                        ix2 = min(w, ix + bw)
                        if ix >= w or y1 >= y2:
                            continue
                        patch = gray[y1:y2, ix:ix2]
                        if patch.size == 0:
                            continue
                        checked += 1
                        if float(patch.mean()) > 40:  # icon region (brighter than dark BG)
                            score += 1

            # Normalise: fraction of predicted positions that have bright pixels
            norm_score = score / max(checked, 1)
            if norm_score > best_score or (norm_score == best_score and
                    entry.get('timestamp', 0) > (best_entry or {}).get('timestamp', 0)):
                best_score = norm_score
                best_entry = entry

        if best_entry is None:
            return None

        _slog.info(f'LayoutDetector: Strategy 1 (learned) — scored {len(candidates)} layouts '
                   f'for [{build_type}] aspect={aspect}, best score={best_score:.2f} '
                   f'({best_entry["res"]})')

        # ── Build result from best layout ────────────────────────────────────
        result = {}
        for slot_name in slot_order:
            geo = best_entry['slots'].get(slot_name)
            if geo is None or isinstance(geo, (int, float)):
                continue

            cy = int(geo['y_rel'] * h)
            bw = max(1, int(geo['w_rel'] * w))
            bh = max(1, int(geo['h_rel'] * h))
            iy = max(0, cy - bh // 2)

            bboxes = []
            if 'runs' in geo:
                # Multi-column layout — use stored run counts (authoritative)
                for run in geo['runs']:
                    x0   = int(run['x0_rel'] * w)
                    step = max(bw, int(run['step_rel'] * w))
                    for j in range(run['count']):
                        bboxes.append((max(0, x0 + j * step), iy, bw, bh))
            else:
                # Single-run layout — respect ship profile count
                n_icons = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, geo.get('count', 1)))
                if n_icons == 0:
                    continue
                x0   = int(geo['x0_rel'] * w)
                step = max(bw, int(geo['step_rel'] * w))
                for j in range(n_icons):
                    bboxes.append((max(0, x0 + j * step), iy, bw, bh))

            if bboxes:
                result[slot_name] = bboxes

        return result if result else None

    # ── Original Logic (truncated for brevity, but kept in final write) ────────

    def _load_community_anchors(self) -> list:
        """Load community_anchors.json (P11) from warp/models/, cached in-memory."""
        if self._community_anchors is not None:
            return self._community_anchors
        try:
            p = _userdata.models_dir() / 'community_anchors.json'
            if not p.exists():
                self._community_anchors = []
                return []
            data = json.loads(p.read_text(encoding='utf-8'))
            self._community_anchors = data.get('entries', [])
            _slog.info(f'LayoutDetector: loaded {len(self._community_anchors)} community anchor entries')
        except Exception as e:
            _slog.debug(f'LayoutDetector: community anchors unavailable: {e}')
            self._community_anchors = []
        return self._community_anchors

    @staticmethod
    def reset_community_anchors_cache() -> None:
        """Invalidate in-memory community anchors cache on all instances (called by ModelUpdater)."""
        # Walk all live LayoutDetector instances via gc — simpler than a class-level ref
        import gc
        for obj in gc.get_objects():
            if type(obj).__name__ == 'LayoutDetector' and hasattr(obj, '_community_anchors'):
                obj._community_anchors = None

    def _detect_traits(self, img, build_type):
        h, w = img.shape[:2]
        section_map = {
            'personal ground traits': 'Personal Ground Traits', 'ground reputation': 'Ground Reputation',
            'active ground rep': 'Active Ground Rep'
        } if 'GROUND' in build_type else {
            'personal space traits': 'Personal Space Traits', 'starship traits': 'Starship Traits',
            'space reputation': 'Space Reputation', 'active space rep': 'Active Space Rep'
        }
        try: ocr_out = self._get_ocr().readtext(img)
        except: return {}
        headers = []
        for (bbox, text, conf) in ocr_out:
            if conf < 0.3: continue
            text_low = text.lower().strip()
            matched = next((can for kw, can in section_map.items() if kw in text_low or text_low in kw), None)
            if matched: headers.append((matched, int((bbox[0][1] + bbox[2][1]) / 2), int(max(p[0] for p in bbox))))
        if not headers: return {}
        headers.sort(key=lambda x: x[1])
        # Trait icons are ~44–55 px absolute regardless of screen height.
        # h * 0.055 underestimates at low-res windows (gives 30 px at h=560).
        icon_est = max(44, int(h * 0.065))
        result = {}
        for i, (section, hy, xr) in enumerate(headers):
            row_y = hy + int(icon_est * 0.5)
            row_y_end = (headers[i + 1][1] - 10) if i + 1 < len(headers) else (row_y + icon_est * 4)
            strip = img[max(0, row_y): min(h, row_y_end), :]
            if strip.size == 0: continue
            # Detect individual icon rows within the section strip —
            # a section may overflow to 2+ rows (e.g. 11 personal space traits).
            row_centers = self._find_icon_rows_in_strip(strip, icon_est)
            _slog.debug(f'LayoutDetector._detect_traits: section={section!r} '
                        f'strip y={max(0,row_y)}..{min(h,row_y_end)} '
                        f'(h={strip.shape[0]}) icon_est={icon_est} '
                        f'row_centers={row_centers}')
            all_bboxes = []
            for rc in row_centers:
                r0 = max(0, rc - icon_est // 2)
                r1 = min(strip.shape[0], rc + icon_est // 2 + 1)
                row_strip = strip[r0:r1, :]
                bboxes = self._find_icon_bboxes_in_strip(row_strip, max(0, row_y) + r0, icon_est)
                _slog.debug(f'LayoutDetector._detect_traits:   rc={rc} '
                            f'row_strip h={row_strip.shape[0]} → {len(bboxes)} bboxes')
                all_bboxes.extend(bboxes)
            if all_bboxes:
                result[section] = all_bboxes
        return result

    def _find_icon_rows_in_strip(self, strip: np.ndarray, icon_est: int) -> list[int]:
        """
        Find Y-centers of icon rows within a section strip.
        Returns list of Y offsets (relative to strip top), one per row found.
        Handles multi-row sections (e.g. 11 personal space traits = 2 rows).
        """
        sh, sw = strip.shape[:2]
        if sh < icon_est // 2:
            return [sh // 2]
        import cv2
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        # Row brightness: how many bright pixels per row
        row_bright = np.sum(mask, axis=1).astype(float) / max(sw, 1)
        # Smooth with a narrow kernel to reduce per-pixel noise
        kernel = max(3, icon_est // 10)
        smoothed = np.convolve(row_bright, np.ones(kernel) / kernel, mode='same')
        # Use 90th-percentile instead of max to avoid a single bright separator line
        # inflating the threshold above actual icon rows.
        peak = float(np.percentile(smoothed, 90))
        _slog.debug(f'LayoutDetector._find_icon_rows: sh={sh} peak={peak:.1f}')
        if peak < 5:          # almost no bright pixels → empty strip
            return [sh // 2]
        threshold = peak * 0.30
        _slog.debug(f'LayoutDetector._find_icon_rows: threshold={threshold:.1f}')
        # Find bright runs (candidate icon rows), merge gaps < icon_est//3
        min_sep = max(icon_est // 2, 20)
        centers: list[int] = []
        in_bright, run_start = False, 0
        for y in range(sh):
            if smoothed[y] >= threshold and not in_bright:
                in_bright, run_start = True, y
            elif smoothed[y] < threshold and in_bright:
                in_bright = False
                center = (run_start + y) // 2
                run_len = y - run_start
                # Filter out thin text labels (< icon_est/2 tall)
                accepted = run_len >= icon_est // 2 and (not centers or center - centers[-1] >= min_sep)
                _slog.debug(f'LayoutDetector._find_icon_rows:   run y={run_start}..{y} '
                            f'len={run_len} center={center} → {"OK" if accepted else "skip"}')
                if accepted:
                    centers.append(center)
        if in_bright:
            center = (run_start + sh) // 2
            run_len = sh - run_start
            accepted = run_len >= icon_est // 2 and (not centers or center - centers[-1] >= min_sep)
            _slog.debug(f'LayoutDetector._find_icon_rows:   run y={run_start}..{sh} '
                        f'len={run_len} center={center} → {"OK(tail)" if accepted else "skip(tail)"}')
            if accepted:
                centers.append(center)
        return centers if centers else [sh // 2]

    def _find_icon_bboxes_in_strip(self, strip, y_offset, icon_size):
        import cv2
        sh, sw = strip.shape[:2]
        if sh == 0 or sw == 0: return []
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        col_bright = np.sum(mask, axis=0).astype(float) / 255
        in_icon, start, bboxes = False, 0, []
        min_w = max(20, icon_size // 2)
        # Threshold: 10% of strip height (was 20%). Dark-bordered icons can have
        # columns with only 3–5 bright pixels; 20% broke their runs into sub-min_w
        # fragments that were discarded, causing the first icon to be missed.
        for x in range(sw):
            bright = col_bright[x] > sh * 0.1
            if bright and not in_icon: in_icon, start = True, x
            elif not bright and in_icon:
                in_icon, run_w = False, x - start
                if run_w >= min_w: bboxes.append((start, y_offset + max(0, (sh - icon_size) // 2), run_w, min(icon_size, sh)))
        if in_icon:
            run_w = sw - start
            if run_w >= min_w: bboxes.append((start, y_offset + max(0, (sh - icon_size) // 2), run_w, min(icon_size, sh)))
        return bboxes

    # Boff profession → canonical slot name
    _PROF_MAP = {
        'tactical':      'Boff Tactical',
        'engineering':   'Boff Engineering',
        'science':       'Boff Science',
        'operations':    'Boff Engineering',
        'intelligence':  'Boff Intelligence',
        'command':       'Boff Command',
        'pilot':         'Boff Pilot',
        'miracle worker':'Boff Miracle Worker',
        'temporal':      'Boff Temporal',
        'medical':       'Boff Science',
    }

    def _detect_boffs_via_markers(self, img) -> dict[str, list]:
        """Strategy 0 for BOFFs: locate panel via profession-coloured seat
        markers and project ability slots from the bible.

        Returns a `{seat_id: [bbox, ...]}` dict using the same `Boff Seat
        L_<y>` / `Boff Seat R_<y>` keys as `_detect_boffs`, so downstream
        consumers (warp_importer) need no special-casing. Empty dict on
        failure — caller falls back to legacy detection chain.

        Detection-only: no annotations.json access (CORE RULE).
        """
        try:
            res = _boff_marker.detect_panel(img)
        except Exception as e:
            _slog.warning(f'LayoutDetector: marker panel detector raised — {e!r}')
            return {}
        if not res:
            return {}

        a = res['col_a']; b = res['col_b']
        slots = res['slots']
        n_a = len(a)
        # Group slots by seat_idx → emit a Boff Seat key per seat.
        per_seat: dict[int, list[tuple[int, int, int, int]]] = {}
        # seat_idx → (side, my, prof_code, spec_code | None)
        seat_meta: dict[int, tuple[str, int, str, str | None]] = {}
        for (mx, my, mw, mh, code, spec) in a:
            si = len(seat_meta)
            seat_meta[si] = ('L', int(my), code, spec)
        for (mx, my, mw, mh, code, spec) in b:
            si = len(seat_meta)
            seat_meta[si] = ('R', int(my), code, spec)

        for (seat_idx, _slot_idx, x, y, w, h, _code) in slots:
            per_seat.setdefault(seat_idx, []).append((int(x), int(y), int(w), int(h)))

        out: dict[str, list] = {}
        for seat_idx, bboxes in per_seat.items():
            side, my, code, spec = seat_meta.get(seat_idx, ('L', 0, 'U', None))
            # Embed prof code (+ optional spec) so warp_dialog Phase 2 and
            # trainer autocomplete can recover the seat profession without
            # re-classifying. Schema parsed by warp.recognition.boff_keys.
            tag = f'{code}+{spec}' if spec else code
            seat_id = f'Boff Seat {side}[{tag}]_{my}'
            out[seat_id] = bboxes

        _slog.info(
            f'LayoutDetector: Strategy 0 (marker panel) → {len(out)} seats, '
            f'{sum(len(v) for v in out.values())} slot bboxes '
            f'(panel score={res["score"]:.2f}, '
            f'cols L={n_a} R={len(b)})'
        )
        return out

    def _detect_boffs(self, img, icon_dims=None, offset=(0, 0), max_bands=3, n_cols=2):
        """Detect BOFF ability icons using structural knowledge of BOFFS screen.

        BOFFS layout: 2 columns (left: max 3 seats, right: max 2 seats).
        Each seat has up to 4 ability icons in a horizontal row.

        Args:
            img: BGR image (full BOFFS screen or sub-region of MIXED).
            icon_dims: optional (icon_w, icon_h, spacing) override — used when
                       detecting BOFFs in a MIXED sub-region where proportional
                       sizing relative to the sub-region width would be wrong.
            offset: (x_off, y_off) added to all output coordinates — used when
                    detecting in a sub-region to translate back to full-image space.
            max_bands: max row bands to keep (3 for standalone BOFFS, more for MIXED).

        Approach:
        1. Find icon row bands via vertical brightness/variance profile
        2. Column split at ~55% of image width (consistent across BOFFS screens)
        3. Template-slide a 4-icon pattern across each (row, column) cell
        4. Classify profession per seat via color analysis
        """
        import cv2
        from collections import Counter
        h, w = img.shape[:2]
        x_off, y_off = offset

        if icon_dims:
            icon_w, icon_h, spacing = icon_dims
        else:
            icon_w = max(20, round(w * 0.078))
            icon_h = max(28, round(icon_w * 1.33))
            spacing = max(24, round(w * 0.093))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Step 2: Column split (defined early — also used for band scan) ---
        # n_cols=1 → single full-width column (Space Stations panel in PicCollage
        # composites). n_cols=2 → standard BOFFS / MIXED screens with L/R seats.
        if n_cols == 1:
            col_split = w
            columns = [(0, w)]
        else:
            col_split = round(w * 0.55)
            columns = [(0, col_split), (col_split, w)]

        # --- Step 1: Find icon row bands ---
        # Scan PER COLUMN independently and merge. Rationale: in MIXED screens
        # the Pilot/last row may have icons only in one column; full-width std
        # averaging drops below threshold and misses the band. Per-column scan
        # captures rows where either side has icons.
        win = max(3, icon_h // 8)
        skip_y = round(h * 0.12)  # unused — kept for reference

        def _scan_bands_in_strip(x1: int, x2: int) -> list[tuple[int, int]]:
            bands: list[tuple[int, int]] = []
            in_b = False
            b_start = 0
            for y in range(0, h - win):
                strip = gray[y:y + win, x1:x2]
                mn = float(strip.mean())
                sd = float(strip.std())
                if 30 < mn < 100 and sd > 30:
                    if not in_b:
                        b_start = y
                        in_b = True
                else:
                    if in_b and y - b_start > icon_h * 0.5:
                        bands.append((b_start, y))
                    in_b = False
            if in_b and h - b_start > icon_h * 0.5:
                bands.append((b_start, h))
            return bands

        left_bands = _scan_bands_in_strip(0, col_split)
        right_bands = _scan_bands_in_strip(col_split, w) if col_split < w else []

        # Merge bands from both columns by Y overlap/proximity
        all_bands = sorted(left_bands + right_bands, key=lambda b: b[0])
        row_bands: list[tuple[int, int]] = []
        for y1, y2 in all_bands:
            if row_bands and y1 < row_bands[-1][1] + icon_h // 3:
                # Overlap or close — merge (expand to union)
                row_bands[-1] = (min(row_bands[-1][0], y1), max(row_bands[-1][1], y2))
            else:
                row_bands.append((y1, y2))

        if not row_bands:
            _slog.debug('LayoutDetector: _detect_boffs — no row bands found')
            return {}

        # Score each band by best 4-icon template match across both columns.
        # Score = peak_score - gap_score, where:
        #   peak_score = sum(mean*std) at 4 icon positions (x_start + k*spacing)
        #   gap_score  = sum(mean*std) at 3 gap positions BETWEEN icons
        # Real BOFF rows have strong peaks with darker gaps between icons → high diff.
        # Text labels / noise are continuous → peak ≈ gap → low diff.
        def _band_score(by1: int, by2: int) -> float:
            mid_y1 = by1 + max(0, (by2 - by1 - icon_h) // 2)
            mid_y2 = min(mid_y1 + icon_h, by2)
            best = 0.0
            gap_w = max(2, (spacing - icon_w) // 1) if spacing > icon_w else max(2, icon_w // 3)
            for cx1, cx2 in columns:
                cell = gray[mid_y1:mid_y2, cx1:cx2]
                cw = cx2 - cx1
                if cell.size == 0 or cw < spacing * 2:
                    continue
                h_std = cell.std(axis=0).astype(float)
                h_mean = cell.mean(axis=0).astype(float)
                search_end = max(1, cw - 3 * spacing - icon_w + 1)
                for xs in range(search_end):
                    peak = 0.0
                    for k in range(4):
                        x = xs + k * spacing
                        xe = min(x + icon_w, cw)
                        if x < cw:
                            peak += float(h_mean[x:xe].mean()) * float(h_std[x:xe].mean())
                    gap = 0.0
                    n_gaps = 0
                    for k in range(3):
                        gx = xs + k * spacing + icon_w
                        gxe = min(gx + gap_w, cw)
                        if gx < cw and gxe > gx:
                            gap += float(h_mean[gx:gxe].mean()) * float(h_std[gx:gxe].mean())
                            n_gaps += 1
                    if n_gaps > 0:
                        # Scale gap score to 4 icons for fair comparison
                        sc = peak - (gap / n_gaps) * 4
                    else:
                        sc = peak
                    if sc > best:
                        best = sc
            return best

        scored = [((b1, b2), _band_score(b1, b2)) for b1, b2 in row_bands]
        scored.sort(key=lambda t: t[1], reverse=True)
        # BOFFS has max 3 row positions — keep top 3 by score, restore Y order
        top = sorted([t[0] for t in scored[:3]], key=lambda b: b[0])
        row_bands = top

        _slog.debug(
            f'LayoutDetector: _detect_boffs — {len(row_bands)} row bands (top-3 by score): '
            + ', '.join(f'y={y1}-{y2}' for y1, y2 in row_bands)
            + ' | all scored: '
            + ', '.join(f'y={b[0][0]}-{b[0][1]}:{b[1]:.0f}' for b in scored)
        )

        # --- Step 3: Template-slide 4 icons per cell ---
        result: dict[str, list] = {}

        for band_y1, band_y2 in row_bands:
            # 3a. Find horizontal alignment (best_x) per cell
            col_best_x = []
            for col_x1, col_x2 in columns:
                cell_w = col_x2 - col_x1
                if cell_w < spacing * 2:
                    col_best_x.append(None)
                    continue

                # Use vertical center of band for the horizontal profile
                mid_y1 = band_y1 + max(0, (band_y2 - band_y1 - icon_h) // 2)
                mid_y2 = min(mid_y1 + icon_h, band_y2)
                cell = gray[mid_y1:mid_y2, col_x1:col_x2]
                if cell.size == 0:
                    col_best_x.append(None)
                    continue

                # Score using horizontal std profile (icon artwork has high variance;
                # profession indicators and empty areas have low variance).
                h_std = cell.std(axis=0).astype(float)
                h_mean = cell.mean(axis=0).astype(float)

                search_end = max(1, cell_w - 3 * spacing - icon_w + 1)
                # Leading gap bonus: real BOFF rows start right after a deep
                # dark gap (between the left-edge profession/rank badge and the
                # first icon). Use MIN of a narrow window just before x_start
                # to capture the deepest dark point (e.g. 1-2 px cliff between
                # badge and icon); robust when the badge extends close to the
                # icon edge.
                lead_w = max(3, (spacing - icon_w) + 2)
                all_scores: list[tuple[float, int]] = []
                for x_start in range(search_end):
                    score = 0.0
                    for k in range(4):
                        x = x_start + k * spacing
                        x_end = min(x + icon_w, cell_w)
                        if x < cell_w:
                            m = float(h_mean[x:x_end].mean())
                            s = float(h_std[x:x_end].mean())
                            score += m * s
                    if x_start - lead_w >= 0:
                        m_arr = h_mean[x_start - lead_w:x_start]
                        s_arr = h_std[x_start - lead_w:x_start]
                        if m_arr.size > 0:
                            lead_min = float((m_arr * s_arr).min())
                            if lead_min < 500:
                                score += (500 - lead_min) * 8
                    all_scores.append((score, x_start))

                if not all_scores:
                    col_best_x.append(None)
                    continue

                all_scores.sort(reverse=True)
                best_score, best_x = all_scores[0]

                if best_score < 500:
                    col_best_x.append(None)
                else:
                    col_best_x.append(best_x)

            if not any(bx is not None for bx in col_best_x):
                continue

            # 3b. Find UNIFIED vertical alignment (icon_y) for the entire row
            # For BOFFs, band_y1 is the highly precise top edge of the icon row.
            # Using try_y search on the whole strip gets pulled downwards by bright text labels.
            icon_y = band_y1 + 2

            # 3c. Build bboxes for valid columns and classify professions
            for c_idx, (col_x1, col_x2) in enumerate(columns):
                best_x = col_best_x[c_idx]
                if best_x is None:
                    continue

                bboxes = []
                for k in range(4):
                    ix = col_x1 + best_x + k * spacing
                    crop_g = gray[icon_y:icon_y + icon_h, ix:ix + icon_w]
                    std = float(crop_g.std()) if crop_g.size > 0 else 0
                    state = 'active' if std > 30 else ('inactive' if std > 8 else 'empty')
                    bboxes.append((ix + x_off, icon_y + y_off, icon_w, icon_h, state))

                # Classify profession by majority vote — only on active crops.
                profs: list[str] = []
                for ix, iy, iw, ih, state in bboxes:
                    if state != 'active':
                        continue
                    crop = img[iy - y_off:iy - y_off + ih, ix - x_off:ix - x_off + iw]
                    if crop.size > 0:
                        prof = self._classify_boff_profession(crop)
                        if prof:
                            profs.append(prof)

                if not profs:
                    continue

                # Instead of mapping to a fixed profession by majority vote,
                # we return the geometrical seat identifier.
                # c_idx == 0 is Left column, c_idx == 1 is Right column.
                side = 'L' if c_idx == 0 else 'R'
                seat_id = f"Boff Seat {side}_{icon_y}"
                result[seat_id] = bboxes

        _slog.debug(
            f'LayoutDetector: _detect_boffs — {len(result)} sections, '
            f'{sum(len(v) for v in result.values())} bboxes'
        )

        # Grid regularity check: real BOFF panels have consistent row spacing
        # and aligned column starts. Reject irregular grids (e.g. trait rows
        # or UI elements that passed the band scan but aren't a real grid).
        if len(result) >= 2:
            ys = sorted({bxs[0][1] for bxs in result.values()})
            if len(ys) >= 2:
                pitches = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
                p_min, p_max = min(pitches), max(pitches)
                if p_min > 0 and p_max / p_min > 2.0:
                    _slog.debug(
                        f'LayoutDetector: _detect_boffs — rejected: '
                        f'irregular row spacing {pitches}')
                    return {}

            # Check column alignment: left-column x_start values should be
            # within ~1 icon_w of each other across rows.
            left_xs = [bxs[0][0] for key, bxs in result.items()
                       if key.startswith('Boff Seat L')]
            if len(left_xs) >= 2:
                x_spread = max(left_xs) - min(left_xs)
                if x_spread > icon_w * 1.5:
                    _slog.debug(
                        f'LayoutDetector: _detect_boffs — rejected: '
                        f'column misalignment x_spread={x_spread} '
                        f'(max={icon_w * 1.5:.0f})')
                    return {}

        return result

    def _detect_boffs_in_mixed(self, img) -> dict[str, list]:
        """Detect BOFF ability icons within a MIXED screen.

        BOFF icons in MIXED screens are much smaller relative to the full image
        than in standalone BOFFS screens, so proportional sizing fails.
        Strategy: estimate absolute icon dimensions from full image width,
        crop to the lower portion (BOFF section is always below y=30%),
        then try left 40% and right 50% sub-regions. Keep whichever has more.
        """
        h, w = img.shape[:2]

        # BOFF icon dimensions scale with overall UI, not sub-region size.
        # Empirical: icon_w ≈ w * 0.021, icon_h ≈ h * 0.047, spacing ≈ w * 0.023
        icon_w = max(20, round(w * 0.021))
        icon_h = max(28, round(h * 0.047))
        spacing = max(24, round(w * 0.023))
        dims = (icon_w, icon_h, spacing)

        # BOFF section in MIXED is usually in the lower 70% (below the ship
        # image/header). Bottom scans use this as a top cap. A separate
        # top-right scan covers PicCollage-style composites where a Space
        # Stations panel can sit above this cut, directly to the right of
        # the EQ column.
        y_start = int(h * 0.30)
        top_skip = int(h * 0.02)  # skip the row of UI tabs at very top

        # Sub-region covers the BOFF panel (not full left/right half).
        # Narrower crop prevents band scan from averaging BOFF icons with adjacent
        # non-BOFF content (ship stats, equipment columns) which dilutes std below
        # the threshold and also generates false bands from the other content.
        # Empirical: BOFF panels span ~30% of image width; use 0.34 for safety margin.
        panel_w = int(w * 0.34)
        right_x = w - panel_w

        # Top-right scan: a Space Stations panel in a collage sits in
        # y∈[~0.02h, ~0.30h]. The left half of the image at this y range
        # contains the ship card + EQ icons (high false-positive risk),
        # so we only add the right column here.
        #
        # Space Stations geometry differs from standard MIXED BOFF rows:
        # icons are larger relative to the panel (~80×105 in a ~1000-px-tall
        # panel) and use the canonical 3+2 seat layout (3 left, 2 right) with
        # 4 abilities per seat at intra-seat spacing ≈ icon_w * 1.14. Use
        # panel-relative dims; n_cols stays at the default 2.
        # Crop wider than the normal panel_w so both columns of seats fit.
        tr_panel_w = int(w * 0.42)
        tr_x = w - tr_panel_w
        tr_h_px = y_start - top_skip
        tr_icon_h = max(28, round(tr_h_px * 0.104))
        tr_icon_w = max(20, round(tr_icon_h * 0.76))
        tr_spacing = max(24, round(tr_icon_w * 1.14))
        tr_dims = (tr_icon_w, tr_icon_h, tr_spacing)

        left_img      = img[y_start:, :panel_w]
        right_img     = img[y_start:, right_x:]
        top_right_img = img[top_skip:y_start, tr_x:]

        left_result      = self._detect_boffs(left_img,  icon_dims=dims, offset=(0,       y_start))
        right_result     = self._detect_boffs(right_img, icon_dims=dims, offset=(right_x, y_start))
        top_right_result = self._detect_boffs(top_right_img, icon_dims=tr_dims,
                                              offset=(tr_x, top_skip))

        left_count       = sum(len(v) for v in left_result.values())
        right_count      = sum(len(v) for v in right_result.values())
        top_right_count  = sum(len(v) for v in top_right_result.values())
        left_groups      = len(left_result)
        right_groups     = len(right_result)
        top_right_groups = len(top_right_result)

        # Tiebreak by (slot_groups, item_count) — more professions = more likely
        # real BOFF panel. Raw count alone misfires when one side (e.g. traits
        # panel) is periodic but all classifies as one profession (e.g. 16 items
        # → "Boff Science" via color default). Real BOFF panels have 3-5
        # distinct professions across seats.
        # BOFF panel physical maximum is 5 rows × 4 cols = 20 bboxes.
        # Candidates with more than 20 bboxes are periodic non-BOFF content
        # (traits / reputation / equipment column) and must be rejected so the
        # tiebreak cannot pick them over a real BOFF panel on the other side.
        MAX_BOFF_BBOXES = 20
        candidates = []
        if 4 <= left_count <= MAX_BOFF_BBOXES:
            candidates.append(('left', left_groups, left_count, left_result))
        if 4 <= right_count <= MAX_BOFF_BBOXES:
            candidates.append(('right', right_groups, right_count, right_result))
        if 4 <= top_right_count <= MAX_BOFF_BBOXES:
            candidates.append(('top_right', top_right_groups, top_right_count, top_right_result))

        if candidates:
            candidates.sort(key=lambda c: (c[1], c[2]), reverse=True)
            side, groups, count, result = candidates[0]
            _slog.info(
                f'LayoutDetector: BOFF-in-MIXED ({side}) → {groups} seats, {count} bboxes '
                f'(left={left_count}/{left_groups}g, '
                f'right={right_count}/{right_groups}g, '
                f'top_right={top_right_count}/{top_right_groups}g)')
            return result

        _slog.debug(
            f'LayoutDetector: BOFF-in-MIXED — no BOFF seats found '
            f'(left={left_count}, right={right_count}, top_right={top_right_count})')
        return {}

    def _fill_boff_gaps(self, bboxes_abs: list, img, icon_est: int,
                        x_min: int = 0, max_slots: int = 4) -> list:
        """
        Given absolute-coordinate bboxes of active BOFF icons in one seat row,
        fill in empty/inactive positions at expected grid intervals.

        Returns list of (x, y, w, h, state) 5-tuples where state is
        'active', 'empty', or 'inactive'.
        """
        if not bboxes_abs:
            return []

        sorted_bx = sorted(bboxes_abs, key=lambda b: b[0])
        xs = [b[0] for b in sorted_bx]

        # Step estimate: minimum positive X-gap, or icon_est for a single icon
        if len(xs) >= 2:
            gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            pos_gaps = [g for g in gaps if g > 0]
            step = min(pos_gaps) if pos_gaps else icon_est
        else:
            step = icon_est

        x0, y0, w0, h0 = sorted_bx[0]
        result: list = []
        consumed: set = set()

        for slot_i in range(max_slots):
            x_exp = x0 + slot_i * step
            if x_exp + w0 > img.shape[1]:
                break

            # Find active bbox closest to expected position (within 45% of step)
            match_i = None
            for i, (bx, _, _, _) in enumerate(sorted_bx):
                if i not in consumed and abs(bx - x_exp) < step * 0.45:
                    match_i = i
                    break

            if match_i is not None:
                result.append((*sorted_bx[match_i], 'active'))
                consumed.add(match_i)
            else:
                y1 = max(0, y0)
                y2 = min(img.shape[0], y0 + h0)
                x1 = max(0, int(x_exp))
                x2 = min(img.shape[1], x1 + w0)
                crop = img[y1:y2, x1:x2]
                state = self._classify_cell(crop) if crop.size > 0 else 'empty'
                if state == 'active':
                    state = 'empty'  # no matched bbox here — treat as empty
                result.append((int(x_exp), y0, w0, h0, state))

        virtual_n = sum(1 for r in result if r[4] != 'active')
        if virtual_n:
            _slog.debug(f'LayoutDetector: _fill_boff_gaps — {virtual_n} virtual positions added '
                        f'({len(result) - virtual_n} active)')
        return result

    @staticmethod
    def _classify_boff_profession(crop_bgr) -> str | None:
        """
        Classify Boff profession from accent glow color in the icon.

        All STO Boff icons share a dark navy-blue background (H 85-120).
        Profession glow color is identified as an ACCENT on top of that background:

          Tactical       — red accent      H  0-15 / 165-180, bright (V≥80)
          Command        — dark-red accent  H  0-15 / 165-180, dim   (V<80)
          Engineering    — amber accent     H 15-30, dominant over blue
          Temporal       — amber + strong mid-blue (H 105-115) alongside amber
          Intelligence   — purple accent    H 115-145
          Miracle Worker — green accent     H 48-72
          Pilot          — cyan accent      H 78-88  (slightly below the bg range)
          Science        — no accent (pure background blue) → default

        Returns the lowercase profession key (matches _PROF_MAP) or None.
        """
        import cv2

        # Drop the outermost edge first. The profession glow is in the icon's
        # own frame, but the outermost pixels of a slot bbox are UI chrome —
        # the slot border, which is tinted by the officer's name-plate colour
        # and has nothing to do with the ability. On kvort_elp.png that border
        # is mint green: three Science abilities each showed ~30% of their
        # sampled ring in the H48-72 "green" band against a 66% blue majority,
        # so all three classified as Miracle Worker, the seat voted 3/3 for it,
        # and every ability in that Universal seat was then looked up in the
        # wrong candidate list.
        #
        # Measured over 1073 confirmed BOFF crops from 79 screenshots, by
        # calling this function rather than a reimplementation of it:
        #
        #   overall      84.9% -> 93.9%      Tactical     96.3% -> 97.6%
        #   Science      72.7% -> 92.2%      Temporal     80.8% -> 75.6%
        #   Pilot        33.3% -> 91.7%      Command      84.8% -> 81.2%
        #   Engineering  91.1% -> 98.3%
        #
        # Temporal and Command lose a few crops — both are identified by an
        # amber or red accent that the inset trims along with the chrome —
        # and that was accepted for the overall gain. Revisit if Temporal
        # seats start reading wrong.
        #
        # A fraction rather than a pixel count because the chrome scales with
        # the icon: crops run from 19 px to 68 px on the short side, and a
        # fixed 4 px inset would eat 42% of the smallest. The 14% found by
        # sweep matches a by-eye reading of the screenshot, where the border
        # glow covers about 5 px of a 35 px icon.
        ih, iw = crop_bgr.shape[:2]
        inset = max(1, int(min(ih, iw) * 0.14))
        if ih - 2 * inset >= 8 and iw - 2 * inset >= 8:
            crop_bgr = crop_bgr[inset:ih - inset, inset:iw - inset]
            ih, iw = crop_bgr.shape[:2]

        # Sample the border ring of what is left. Measured on the same 1073
        # crops by editing this function and re-running it, not by
        # reimplementing it: ring 93.9%, whole crop 94.4%, centre alone
        # 90.6%. So the tint is spread across the icon — the centre by itself
        # is the weakest of the three — but the edge region does carry signal,
        # and the ring is within five crops of the best option. The inset
        # above is what actually mattered; this choice is close to a wash and
        # is left as it was.
        b = max(3, int(min(ih, iw) * 0.22))
        top    = crop_bgr[:b, :].reshape(-1, 3)
        bottom = crop_bgr[-b:, :].reshape(-1, 3)
        left   = crop_bgr[b:-b, :b].reshape(-1, 3)
        right  = crop_bgr[b:-b, -b:].reshape(-1, 3)
        border = np.concatenate([top, bottom, left, right])

        hsv = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        sat_mask = (hsv[:, 1] > 80) & (hsv[:, 2] > 60)
        if sat_mask.sum() < 8:
            return None

        hues = hsv[sat_mask, 0]   # 0-180
        vals = hsv[sat_mask, 2]
        mean_v = float(vals.mean())

        # 36-bin hue histogram (5° per bin)
        hist, _ = np.histogram(hues, bins=36, range=(0, 180))

        # Helper: count pixels whose hue falls in [lo, hi] degrees (OpenCV 0-180)
        def _h(lo, hi):
            return int(hist[lo // 5: hi // 5 + 1].sum())

        # Hue bands (OpenCV H: 0-180 = half of 360°)
        red_lo    = _h(0,    9)    # pure red H0-9 (real 0-18°); H10+ is orange/amber
        red_hi    = _h(160, 175)   # dark red / maroon (high-H side of wrap-around)
        red_total = red_lo + red_hi
        amber     = _h(10,  30)    # amber/gold = Engineering & Temporal (H10+ = orange-amber)
        mid_blue  = _h(105, 120)   # Temporal's distinctive mid-blue (H105-120, NOT H90-100 bg)
        green     = _h(48,  72)    # green = Miracle Worker
        purple    = _h(115, 145)   # purple/violet = Intelligence
        bg_blue   = _h(85, 120)    # common background navy (shared by all)

        total = int(hist.sum()) or 1

        # ── Command / Tactical (red accent) ───────────────────────────────────
        # Command peaks at H 160-175 (dark maroon), Tactical at H 0-9 (pure red)
        if red_total / total >= 0.05:
            return 'command' if red_hi > red_lo else 'tactical'

        # ── Intelligence (purple accent — unique to this profession) ──────────
        if purple / total >= 0.07 or purple >= 40:
            return 'intelligence'

        # ── Miracle Worker (green accent) ─────────────────────────────────────
        if green / total >= 0.10 or green >= 25:
            return 'miracle worker'

        # ── Engineering vs Temporal (both amber, Temporal also has strong mid-blue)
        if amber / total >= 0.12 or amber >= 40:
            # Temporal: amber is prominent AND significant mid-blue (H105-120) also present
            # Engineering has mid_blue ≈ 0; Temporal has mid_blue = 30-50% of amber.
            # mid_blue >= 60 alone is also sufficient — covers temporal crops with very
            # strong amber where the ratio rule would otherwise reject them.
            if (mid_blue >= 40 and mid_blue >= amber * 0.28) or mid_blue >= 60:
                return 'temporal'
            return 'engineering'

        # ── Pilot vs Science (both pure blue; Pilot peaks at H95, Science at H100+)
        # hist bin 19 = H95-99, bin 20 = H100-104
        if int(hist[19]) > int(hist[20]) and int(hist[19]) >= 30:
            return 'pilot'

        # ── Science (default: icon is dominated by background blue, no accent)
        return 'science'

    @staticmethod
    def _classify_cell(crop_bgr) -> str:
        """
        Classify a single slot cell crop as 'active', 'empty', or 'inactive'.

        Uses the inner 60% of the crop to avoid border contamination.

        active   — has a visible icon (bright content)
        inactive — locked / unavailable slot. The game draws this several
                   ways, and all of them share one property: something is
                   painted on an otherwise dark cell.
                     BOFFS:     navy-blue fill with an X across it
                     EQ/Traits: near-black carrying a 'LOCK' word, a padlock
                                symbol, or a level requirement such as
                                'lvl 65'
        empty    — slot exists but nothing is slotted: an unpainted cell,
                   either near-black or the navy panel showing through, with
                   a thin border and nothing else

        So the three separate on how much the cell varies, not on how bright
        or how blue it is — an unpainted cell is flat whatever its colour,
        and every inactive marking, X or word or padlock, adds variance.
        """
        import cv2
        if crop_bgr is None or crop_bgr.size == 0:
            return 'active'  # unknown → treat as active (safe fallback)
        ih, iw = crop_bgr.shape[:2]
        mx = max(1, int(iw * 0.20))
        my = max(1, int(ih * 0.20))
        inner = crop_bgr[my:ih - my, mx:iw - mx]
        if inner.size == 0:
            inner = crop_bgr
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        mean_v = hsv[:, 2].mean()
        std_v  = hsv[:, 2].std()
        mean_s = hsv[:, 1].mean()
        mean_h = hsv[:, 0].mean()

        # Blue-saturated navy window. Must be checked BEFORE the brightness
        # gate: at typical screenshot resolutions these cells have mean_v ~70,
        # well above the 45 threshold for generic active. `mean_s > 100`
        # separates the saturated navy from dim-but-desaturated icons (e.g.
        # Hold Together).
        #
        # Inside it, brightness variance tells the three apart, because what
        # varies is how much pattern the cell carries. Measured 2026-09-04
        # against 985 confirmed virtual crops and a 1200-crop sample of real
        # icons, all classified by this function:
        #
        #   flat navy fill  std_v < 5    the panel showing through an EMPTY
        #                                slot — no X drawn on it at all
        #   navy + X        std_v 5-33   INACTIVE; the X is the variance
        #   real icon       std_v > 33   ACTIVE (minimum seen: 33.4)
        #
        # The upper gate was 20, which left 28 inactive cells reading as
        # active — the damaging direction, since a locked cell then gets an
        # item name. Raising it to 30 catches 662 of 663 and still clears the
        # dimmest real icon by 3.4, so it is a widened margin, not a tighter
        # squeeze between the classes.
        if mean_s > 100 and 95 < mean_h < 130:
            if std_v < 5:
                return 'empty'
            if std_v < 30:
                return 'inactive'
        if mean_v > 45:
            return 'active'
        # Darker BOFF inactive (lower-res screenshots / dimmer monitors)
        if mean_s > 40 and 95 < mean_h < 130:
            return 'inactive'
        # The dark, non-navy markings: a 'LOCK' word, a padlock symbol, or a
        # level requirement. Whichever it is, it is drawn on a near-black cell
        # and its pixels raise brightness variance above a flat one.
        #
        # Measured 2026-09-04 over the same crops, restricted to dark
        # non-navy cells (124 inactive, 150 empty): inactive sits at a median
        # std_v of 15.1 and a 10th percentile of 11.7, empty at a median of
        # 1.7 and a 90th percentile of 4.3. The gate was 10, which missed 4
        # of the faintest markings; 8 catches 122 of 124 for the same two
        # false positives, and 6 buys nothing further, so 8 is the widest
        # margin that costs nothing.
        if std_v > 8:
            return 'inactive'
        return 'empty'

    def _count_icons_in_row(self, img, y_top, y_bot, panel_right, cell_w,
                            slot_name: str = '',
                            panel_x_start: int | None = None) -> tuple[int, list[str]]:
        """
        Count active icons in a row, scanning right-to-left.

        Returns (count, cell_states) where cell_states is a list of
        'active' | 'empty' | 'inactive' for each scanned cell position
        (index 0 = rightmost cell).

        Empty and inactive cells are skipped in the count but do NOT
        stop the scan — only two consecutive background cells stop it.
        A background cell is any dark cell that lies outside the known
        slot grid (distinguished from empty/inactive by context: once
        we exit the grid there is no more slot structure).

        When `panel_x_start` is given the scan is hard-capped to the
        6-cell matrix width — we stop as soon as the next sample would
        cross the panel's left edge. This avoids classifying off-panel
        content (ship image, BOFF tray) as inactive/empty grid cells.
        """
        import cv2
        row_h = y_bot - y_top
        y1 = max(0, y_top + row_h // 4)
        y2 = min(img.shape[0], y_bot - row_h // 4)
        count = 0
        # Matrix is 6 cells wide when panel_x_start is known; otherwise
        # the legacy buffered scan (8) is retained for backwards compat.
        max_icons = 6 if panel_x_start is not None else 8
        consecutive_bg = 0   # counts cells that look like plain background (not a slot)
        cell_states: list[str] = []
        for j in range(max_icons):
            x2 = panel_right - j * cell_w
            x1 = max(0, x2 - int(cell_w * 0.85))
            if x1 >= x2 or x1 < 0:
                break
            if panel_x_start is not None and x1 < panel_x_start - 2:
                break
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                break
            state = self._classify_cell(crop)
            cell_states.append(state)
            if state == 'active':
                count += 1
                consecutive_bg = 0
            else:
                # empty/inactive — still a grid cell, don't increment bg counter
                # but check if it truly looks like featureless background:
                # background has even lower brightness and zero border structure
                avg = float(crop.mean())
                if avg < 8:          # essentially pure black — outside grid
                    consecutive_bg += 1
                    if consecutive_bg >= 2:
                        break
                # else: empty/inactive slot within grid — keep scanning
        if slot_name and any(s in ('empty', 'inactive') for s in cell_states):
            # cell_states is indexed 0 = rightmost; report 1-based positions
            # counted from the LEFT for human readability (pos 1 = leftmost slot).
            n_cells = len(cell_states)
            non_active = [(n_cells - i, s) for i, s in enumerate(cell_states)
                          if s != 'active']
            non_active.sort()
            _slog.info(
                f'LayoutDetector: [{slot_name}] {count} active + '
                + ', '.join(f'{s} at pos {p}' for p, s in non_active)
            )
        return max(1, count), cell_states

    def _detect_via_pixel_analysis(self, img, slot_order, profile):
        """Equipment layout detection.

        Primary path uses OCR-anchored EQ geometry (eq_geometry.detect_eq_geometry):
        OCR labels supply row anchors, single-slot icon right-edges supply
        panel_right, and dx is computed as (panel_right - panel_x_start) / 6.
        Cell width = final_dx, icon dims derived from row_pitch. Row identity
        is taken from geom.eq_label_cys (OCR-anchored STD_ORDER mapping) when
        present; slot_order[i] is a positional fallback for unmapped rows.

        Fallback (no EQ labels found, OCR failure, etc.) is the legacy
        brightness/row-separator scan retained as _detect_via_pixel_analysis_legacy.
        """
        h, w = img.shape[:2]
        geom = self._get_eq_geometry(img)
        if geom is None or not geom.row_cys:
            return self._detect_via_pixel_analysis_legacy(img, slot_order, profile)

        panel_x_start = geom.panel_x_start
        panel_right   = geom.panel_right
        cell_w  = max(20, int(round(geom.final_dx)))
        icon_w  = max(20, cell_w - 2)
        icon_h  = max(20, int(round(geom.row_pitch * 0.85)) + 2)

        # OCR-anchored slot identity per row. Carriers / non-standard ships
        # may skip rows (e.g. T6 carrier has no Universal Consoles but does
        # have Hangars) — positional slot_order[i] mislabels rows below the
        # skip point. eq_label_cys maps cy → STD_ORDER index from OCR'd
        # canonical labels; use it as authoritative when present.
        cy_to_slot: dict[int, str] = {
            cy: _STD_IDX_TO_PROD_SLOT[std_idx]
            for std_idx, cy in geom.eq_label_cys.items()
            if std_idx in _STD_IDX_TO_PROD_SLOT
        }

        # Positional fallback: extend slot_order with optional slots present
        # in profile (Sec-Def after Deflector, Experimental/Hangars after
        # Aft Weapons). Mirrors _detect_via_ocr_anchored extended_order logic
        # so ships with Secondary Deflector or Experimental Weapons don't
        # shift rows when OCR misses the label for those rows.
        #
        # ShipDB profile is the source of truth for slot presence: when the
        # profile is known, drop base slots the ship has 0 of (e.g. ships
        # without Universal Consoles) so the positional sequence does not
        # reserve a row for them and shove the real rows below into the
        # wrong label. Unknown profile → keep full slot_order (no regression).
        profile_known = bool(profile)
        extended_order: list[str] = []
        for s in slot_order:
            if profile_known and profile.get(s, -1) == 0:
                continue
            if s in extended_order:
                continue
            extended_order.append(s)
            if s == 'Deflector' and profile.get('Sec-Def', 0) > 0 and 'Sec-Def' not in extended_order:
                extended_order.append('Sec-Def')
            # Experimental only. `Hangars` used to be inserted here too, and
            # the game draws it LAST, not after Aft Weapons — which put it at
            # position 6 of the sequence and shifted every row below it down by
            # one. It is already last in SPACE_SLOT_ORDER_CARRIER, so it needs
            # no insertion; the `s in extended_order` guard above keeps it from
            # being appended twice if a caller's order lists it early.
            if s == 'Aft Weapons' and profile.get('Experimental', 0) > 0 \
                    and 'Experimental' not in extended_order:
                extended_order.append('Experimental')

        # Rows OCR could not read are named from the rows it could, using the
        # sequence the ship's profile says this panel holds — see
        # `fill_unanchored_rows`. Anchored rows are untouched, and a run is
        # only filled when its length matches the slots that belong in it.
        filled = fill_unanchored_rows(geom.row_cys, cy_to_slot, extended_order)
        for cy, name in filled.items():
            if cy not in cy_to_slot:
                _slog.info(
                    f'LayoutDetector: row cy={cy} had no OCR label — '
                    f'{name!r} from the profile sequence between the '
                    f'anchored rows around it')
        _unnamed = [cy for cy in geom.row_cys if cy not in filled]
        if _unnamed:
            # Said out loud rather than skipped quietly: an unnamed row is a
            # slot the user will have to draw by hand, and the count is how
            # anyone learns the sequence and the anchors disagree.
            _slog.info(
                f'LayoutDetector: {len(_unnamed)} row(s) left unnamed at '
                f'cy={_unnamed} — the profile sequence does not fit the gaps '
                f'between the OCR-anchored rows')

        result: dict = {}
        for i, cy in enumerate(geom.row_cys):
            slot_name = filled.get(cy)
            if slot_name is None:
                continue
            y_top = max(0, cy - icon_h // 2)
            y_bot = min(h, cy + icon_h // 2)
            pixel_count, _ = self._count_icons_in_row(
                img, y_top, y_bot, panel_right, cell_w, slot_name,
                panel_x_start=panel_x_start)
            # Record before the profile decides what to emit: a row the
            # profile counts as 0 is skipped below, and its measurement is
            # exactly the evidence that the profile is missing a bonus.
            self.last_row_pixel_counts[slot_name] = max(
                self.last_row_pixel_counts.get(slot_name, 0), pixel_count)
            profile_count = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            # ShipDB profile already includes tier bonuses (T6-X +1 Universal,
            # T6-X2 +1 Device, etc.) via warp_importer. Trust profile_count;
            # pixel_count is logged only for sanity-check / regression diag.
            # Older code added +1 tolerance, which double-counted tier bonuses
            # and dragged the decorative left-side sash into the grid as a
            # phantom __empty__ slot on short rows (e.g. 3-slot Eng Cons).
            n_icons = profile_count
            if n_icons == 0:
                continue
            # Project cells right→left from panel_right. The grid itself is
            # the constraint: any candidate whose left edge falls before
            # panel_x_start lies outside the 6-cell matrix and is discarded.
            bboxes = []
            for j in range(n_icons):
                # Float-domain positioning per cell — avoids 1-2 px overshoot
                # past panel_x_start when round(final_dx) is just above the
                # true float (e.g. final_dx=36.67 → cell_w=37 → 6×37=222 vs
                # true panel width 220).
                bx = int(round(panel_right - (j + 1) * geom.final_dx)) + 1
                if bx < panel_x_start:
                    break
                # Clamp bbox to image bounds. Row 0 with cy<icon_h/2
                # (e.g. Fore Weapons at cy=28 with icon_h=58) otherwise
                # emits y=-1, which downstream slicers treat as "last row"
                # and produce empty crops.
                _by = cy - icon_h // 2
                _bh = icon_h
                if _by < 0:
                    _bh += _by
                    _by = 0
                if _by + _bh > h:
                    _bh = h - _by
                if _bh <= 0:
                    continue
                bboxes.append((bx, _by, icon_w, _bh))
            if not bboxes:
                continue
            _slog.info(
                f'LayoutDetector: row {i} [{slot_name}] '
                f'pixel_count={pixel_count} profile={profile_count} → '
                f'requested {n_icons}, kept {len(bboxes)} within grid')
            bboxes.reverse()
            result[slot_name] = bboxes
        return result

    def _detect_via_pixel_analysis_legacy(self, img, slot_order, profile):
        """Brightness/row-separator fallback used when OCR-anchored geometry
        cannot lock onto EQ labels (no usable text, BOFF-only screen, etc.)."""
        h, w = img.shape[:2]
        panel_right = self._find_panel_right_edge_brightness(img)
        if panel_right < w * 0.3: return {}
        row_seps = self._find_row_separators(img, max(0, panel_right - int(w * 0.25)), panel_right)
        if len(row_seps) < 3: return {}
        row_bounds = [(row_seps[i], row_seps[i+1]) for i in range(len(row_seps)-1) if row_seps[i+1]-row_seps[i] >= 30]
        if not row_bounds: return {}
        row_h_avg = sum(b-a for a, b in row_bounds) / len(row_bounds)
        cell_w, icon_w, icon_h = max(30, int(row_h_avg * 0.80)), max(26, int(row_h_avg * 0.80)-4), max(26, int(row_h_avg * 0.78))
        result = {}
        for i, (y_top, y_bot) in enumerate(row_bounds):
            if i >= len(slot_order): break
            slot_name = slot_order[i]
            pixel_count, _ = self._count_icons_in_row(img, y_top, y_bot, panel_right, cell_w, slot_name)
            profile_count = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            n_icons = profile_count
            if n_icons == 0: continue
            _slog.info(f'LayoutDetector: row {i} [{slot_name}] pixel_count={pixel_count} profile={profile_count} → using {n_icons} (legacy)')
            iy, bboxes = (y_top + y_bot) // 2 - icon_h // 2, []
            for j in range(n_icons): bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, icon_w, icon_h))
            bboxes.reverse()
            result[slot_name] = bboxes
        return result

    def _find_panel_right_edge(self, img: np.ndarray) -> int:
        """Right edge of the equipment matrix in pixels.

        Primary: OCR-anchored EQ geometry detector — uses single-slot icon
        right-edges (Deflector/Engines/Warp Core/Shields) for pixel-accurate
        anchoring. Falls back to a brightness-histogram scan when no EQ labels
        are detected (e.g. BOFF-only or trait-only screens reaching deep
        fallback paths)."""
        geom = self._get_eq_geometry(img)
        if geom is not None:
            return geom.panel_right
        return self._find_panel_right_edge_brightness(img)

    def _find_panel_right_edge_brightness(self, img: np.ndarray) -> int:
        """Legacy brightness-histogram fallback. Walks columns right→left and
        returns the first x whose 10 horizontal bands are ≥7/10 bright."""
        h, w = img.shape[:2]
        y_bands = [(int(h * 0.03 + i * int(h * 0.87 / 10)), int(h * 0.03 + (i + 1) * int(h * 0.87 / 10))) for i in range(10)]
        for x in range(w - 2, max(w // 5, 50), -1):
            if sum(1 for (y1, y2) in y_bands if any(sum(int(c) for c in img[y, x]) / 3 > 50 for y in range(y1, y2, 4))) >= 7: return x
        return int(w * 0.90)

    def _find_row_separators(self, img, x_start, x_end):
        h, w = img.shape[0], img.shape[1]
        x_step = max(1, (x_end - x_start) // 25)
        row_avgs = [sum(sum(int(c) for c in img[y, x]) / 3 for x in range(x_start, x_end, x_step) if x < w) / max(1, (x_end-x_start)//x_step) for y in range(h)]
        smoothed = [sum(row_avgs[max(0, y-2):min(h, y+3)]) / 5 for y in range(h)]
        dark_thr = min(30.0, max(smoothed[10:h-10] if h > 20 else [100.0]) * 0.25)
        dark_runs, in_dark, ds = [], False, 0
        for y, avg in enumerate(smoothed):
            if avg < dark_thr and not in_dark: in_dark, ds = True, y
            elif avg >= dark_thr and in_dark:
                in_dark = False
                if y - ds >= 2: dark_runs.append((ds, y))
        if in_dark: dark_runs.append((ds, h - 1))
        merged = []
        for s, e in dark_runs:
            if merged and s - merged[-1][1] < 4: merged[-1] = (merged[-1][0], e)
            else: merged.append([s, e])
        seps = sorted([int((s + e) / 2) for s, e in merged])
        if not seps or seps[0] > 15: seps = [0] + seps
        if not seps or seps[-1] < h - 40: seps = seps + [h]
        return sorted(seps)

    # ── Full scan (MIXED screens) ────────────────────────────────────────────────

    def _ocr_section_labels(self, img) -> dict[str, tuple[float, float]]:
        """Run full-image OCR, return {slot_name: (center_x, center_y)} for each found label.

        Cached on pixel content for the same reason as the panel geometry: the
        importer re-enters `detect()` once the pixel counts refine the ship
        profile, and label positions do not depend on the profile. Measured at
        6.1-6.9 s per call on a 2563x1841 screenshot, which was most of what
        the second pass still cost.

        STO stacks multi-word labels vertically in narrow sidebars (e.g. "Fore" and
        "Weapons" on separate lines), so EasyOCR returns them as two fragments. A
        single word like "Weapons" fuzzy-matches to 'Aft Weapons' / 'Fore Weapons'
        and picks the first alphabetically → phantom 'Aft Weapons' detections.
        Merge vertically-stacked fragments (same cx, small cy gap) before matching.
        """
        key = self._img_key(img)
        if key in self._ocr_labels_cache:
            return self._ocr_labels_cache[key]

        # The shared tokens when `detect()` supplied them — this was the last
        # place that opened the screenshot for itself, and it did so in BGR,
        # handing a reader that expects RGB the red and blue channels
        # swapped. Reading for itself is kept for callers outside detect().
        tokens = getattr(self, '_ocr_tokens', None)
        if tokens is None:
            import cv2
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tokens = [
                    {'text': t.strip(),
                     'conf': float(c),
                     'cx': float(np.mean([p[0] for p in b])),
                     'cy': float(np.mean([p[1] for p in b]))}
                    for b, t, c in self._get_ocr().readtext(rgb)
                    if t.strip()
                ]
            except Exception:
                return {}

        # Extract raw candidates with positions
        raw: list[tuple[float, float, str]] = []
        for tok in tokens:
            text, conf = tok['text'], tok['conf']
            if conf < config.OCR_CONF_THRESHOLD:
                continue
            cx = float(tok['cx'])
            cy = float(tok['cy'])
            raw.append((cx, cy, text.strip()))

        # Merge stacked fragments: same column (cx within 25px), close below (cy gap 5..30).
        # Sort top-to-bottom so parent labels (e.g. "Fore") are processed before their
        # stacked children (e.g. "Weapons") and can absorb them.
        raw.sort(key=lambda r: (r[1], r[0]))
        used = [False] * len(raw)
        merged: list[tuple[float, float, str]] = []
        for i, (cx_i, cy_i, t_i) in enumerate(raw):
            if used[i]:
                continue
            group = [(cx_i, cy_i, t_i)]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(raw)):
                    if used[j]:
                        continue
                    cx_j, cy_j, t_j = raw[j]
                    # Join if cx near any group member and cy directly below it
                    for (cx_g, cy_g, _) in group:
                        if abs(cx_j - cx_g) <= 25 and 0 < cy_j - cy_g <= 30:
                            group.append((cx_j, cy_j, t_j))
                            used[j] = True
                            changed = True
                            break
            group.sort(key=lambda g: g[1])
            joined = ' '.join(g[2] for g in group)
            avg_cx = sum(g[0] for g in group) / len(group)
            avg_cy = sum(g[1] for g in group) / len(group)
            merged.append((avg_cx, avg_cy, joined))

        # Match joined texts to slot aliases. Track the longest text per slot:
        # a single word like "Starship" aliases to "Starship Traits" and would
        # overwrite the real "Starship Traits" hit if it came later in the
        # iteration (e.g. fragment from "Starship Mastery" text near the BOFF
        # panel). Prefer the longest matching text — more specific = correct label.
        labels: dict[str, tuple[float, float]] = {}
        best_text_len: dict[str, int] = {}
        for cx, cy, text in merged:
            slot = self._match_label(text.lower())
            if not slot:
                continue
            tlen = len(text)
            if tlen > best_text_len.get(slot, -1):
                labels[slot] = (cx, cy)
                best_text_len[slot] = tlen
        self._ocr_labels_cache[key] = labels
        return labels

    def _detect_via_full_scan(self, img: np.ndarray, build_type: str,
                               icon_matcher, app_cache) -> dict[str, list[tuple]]:
        """Full-image detection for MIXED screens: OCR labels + dense icon scan + fusion.

        Phase 1 — OCR: find slot label positions across the full image.
        Phase 2 — Dense scan: sliding window + EfficientNet classify every patch.
        Phase 3 — NMS + row clustering + per-row slot scoring.
        Phase 4 — Output dict {slot_name: [(x,y,w,h), ...]} like other strategies.
        """
        from collections import Counter
        h, w = img.shape[:2]
        icon_est = max(32, int(h * 0.060))
        stride   = max(8, icon_est // 2)

        eq_cache        = getattr(app_cache, 'equipment',      {}) or {}
        trait_cache     = getattr(app_cache, 'traits',         {}) or {}
        starship_traits = getattr(app_cache, 'starship_traits',{}) or {}
        boff_cache      = getattr(app_cache, 'boff_abilities', {}) or {}

        # Phase 1: OCR section labels
        ocr_labels = self._ocr_section_labels(img)
        _slog.info(f'LayoutDetector FullScan: OCR labels → {list(ocr_labels.keys())}')

        # Phase 2: Sliding window
        raw_dets = []
        for y in range(0, h - icon_est + 1, stride):
            for x in range(0, w - icon_est + 1, stride):
                patch = img[y:y+icon_est, x:x+icon_est]
                if float(patch.std()) < 12.0:  # skip uniform background
                    continue
                name, conf = icon_matcher.classify_patch(patch)
                if conf >= _SCAN_CONF_MIN and name:
                    raw_dets.append((x, y, icon_est, icon_est, name, conf))

        if not raw_dets:
            _slog.info('LayoutDetector FullScan: no detections — scan failed')
            return {}

        dets = _nms_boxes(raw_dets)
        _slog.info(f'LayoutDetector FullScan: {len(raw_dets)} raw → {len(dets)} after NMS')

        # Enrich detections with item type (computed once, reused across scoring).
        # For unrecognised patches in BOFF screens, try color classification as fallback.
        _is_boff_screen = build_type in ('BOFFS', 'SPACE_BOFFS', 'GROUND_BOFFS')
        enriched = []
        for det in dets:
            itype = _get_item_type(det[4], eq_cache, trait_cache, starship_traits, boff_cache)
            if not itype and _is_boff_screen:
                x, y, bw, bh = det[:4]
                crop = img[y:y+bh, x:x+bw]
                if crop.size > 0:
                    prof = self._classify_boff_profession(crop)
                    if prof:
                        itype = f'__boff_{prof}'
            enriched.append(det + (itype,))
        # each entry: (x, y, w, h, name, conf, item_type)

        # Phase 3: Cluster into rows
        rows = _cluster_rows_by_y(enriched, icon_est)
        _slog.info(f'LayoutDetector FullScan: {len(rows)} rows')

        # Phase 4: Score each row against candidate slot names
        result: dict[str, list] = {}
        used_slots: set[str] = set()

        # Only consider slot names relevant to this build type
        boff_slots = sorted(_BOFF_SLOT_NAMES)
        if 'SPACE' in build_type or build_type in ('BOFFS',):
            candidates = (
                list(SPACE_SLOT_ORDER_STANDARD) + ['Hangars', 'Experimental', 'Sec-Def']
                + ['Personal Space Traits', 'Starship Traits', 'Space Reputation', 'Active Space Rep']
                + boff_slots
            )
        else:
            candidates = (
                list(GROUND_SLOT_ORDER)
                + ['Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep']
                + boff_slots
            )

        for row in sorted(rows, key=lambda r: min(it[1] for it in r)):
            row_types = [it[6] for it in row]
            row_cx    = sum(it[0] + it[2] / 2 for it in row) / len(row)
            row_cy    = sum(it[1] + it[3] / 2 for it in row) / len(row)

            best_slot, best_score = '', 0.0
            for slot_name in candidates:
                score = _score_row_for_slot(row_types, slot_name, ocr_labels,
                                            row_cx, row_cy, icon_est)
                if score > best_score:
                    best_score = score
                    best_slot  = slot_name

            if not best_slot or best_score < 0.30:
                continue

            # Multi-row slots (traits, boffs) — allow multiple rows per slot
            is_multi = best_slot in _TRAIT_SLOT_MARKER or best_slot in _BOFF_SLOT_NAMES
            if best_slot in used_slots and not is_multi:
                continue  # single-row slot already filled
            used_slots.add(best_slot)

            bboxes = [(it[0], it[1], it[2], it[3]) for it in sorted(row, key=lambda it: it[0])]
            result.setdefault(best_slot, []).extend(bboxes)

        # Sanity guard — drop slot groups exceeding the realistic per-slot cap,
        # then bail entirely if the (post-drop) total still exceeds the
        # per-build-type cap. Either signal means FullScan latched onto noise
        # rather than real icons.
        dropped = [(s, len(b)) for s, b in result.items()
                   if len(b) > _FULLSCAN_MAX_PER_SLOT]
        if dropped:
            for s, _ in dropped:
                del result[s]
            _slog.warning(
                f'LayoutDetector FullScan: dropped slot-group(s) over '
                f'{_FULLSCAN_MAX_PER_SLOT}-bbox cap: {dropped} '
                f'(likely false positives — dedicated detector should have '
                f'handled this panel)')
        total_cap = _FULLSCAN_MAX_TOTAL.get(build_type, _FULLSCAN_DEFAULT_TOTAL)
        total = sum(len(v) for v in result.values())
        if total > total_cap:
            _slog.warning(
                f'LayoutDetector FullScan: {total} bboxes exceeds {build_type} '
                f'cap {total_cap} — discarding all FullScan output')
            return {}

        _slog.info(f'LayoutDetector FullScan: {len(result)} slots → {list(result.keys())}')
        return result

    # ── OCR-anchored MIXED detection ─────────────────────────────────────────────

    def _find_row_right_edge(self, img: np.ndarray, y_top: int, y_bot: int,
                              x_min: int, x_max: int, cell_w: int) -> int | None:
        """
        Scan y-band right-to-left, return rightmost x where a contiguous bright
        column exists (=right edge of icon column).
        """
        h, w = img.shape[:2]
        y_top = max(0, y_top)
        y_bot = min(h, y_bot)
        x_max = min(w - 1, x_max)
        x_min = max(0, x_min)
        if y_bot - y_top < 4 or x_max - x_min < cell_w:
            return None
        # Sample brightness per column, then find rightmost bright run of ≥ cell_w/3
        probe_step = max(1, (y_bot - y_top) // 6)
        col_brightness = []
        for x in range(x_min, x_max):
            if x >= w:
                break
            strip = img[y_top:y_bot:probe_step, x]
            col_brightness.append(float(np.mean(strip)))
        bright_thr = 45.0
        min_run = max(6, cell_w // 3)
        # Find rightmost column that is the end of a bright run
        run_len, last_end = 0, None
        for i, b in enumerate(col_brightness):
            if b >= bright_thr:
                run_len += 1
                if run_len >= min_run:
                    last_end = x_min + i + 1  # exclusive end
            else:
                run_len = 0
        return last_end

    def _detect_via_ocr_anchored(self, img: np.ndarray, build_type: str,
                                  slot_order: list, profile: dict) -> dict:
        """
        OCR-anchored layout detection for MIXED screens.

        Uses OCR-found label positions as row anchors (y = label cy), then does
        per-row pixel analysis to find the local right edge of the icon column
        and counts icons. This avoids the failure mode of _find_panel_right_edge
        (global scan) finding the wrong panel (e.g. traits or BOFFs) on MIXED.

        Returns {slot_name: [bbox,...]} or {} if insufficient anchors.
        """
        import statistics

        h, w = img.shape[:2]
        labels = self._ocr_section_labels(img)
        if not labels:
            _slog.info('LayoutDetector OCRAnchored: no OCR labels')
            return {}

        eq_slots = set(slot_order)
        trait_slots = set(_TRAIT_SLOT_MARKER.keys())

        # Separate EQ labels from trait labels (traits used only to bound x_max)
        eq_labels = {s: (cx, cy) for s, (cx, cy) in labels.items() if s in eq_slots}
        trait_labels = {s: (cx, cy) for s, (cx, cy) in labels.items() if s in trait_slots}

        if len(eq_labels) < 3:
            _slog.info(f'LayoutDetector OCRAnchored: only {len(eq_labels)} EQ labels, skip')
            return {}

        # Outlier filter: keep only labels whose cx is near the cluster median
        cxs = [cx for (cx, _) in eq_labels.values()]
        median_cx = statistics.median(cxs)
        eq_labels = {s: (cx, cy) for s, (cx, cy) in eq_labels.items()
                     if abs(cx - median_cx) <= 100}
        if len(eq_labels) < 3:
            _slog.info('LayoutDetector OCRAnchored: <3 labels after cx clustering')
            return {}

        # Drop labels that violate slot_order vs cy (OCR misreads like
        # "Aft Weapons" at cy=56 when Fore Weapons is at cy=39 on the same screen).
        # Two checks: monotonic slot_order index AND minimum cy gap.
        order_idx = {s: i for i, s in enumerate(slot_order)}
        sorted_by_cy = sorted(eq_labels.items(), key=lambda kv: kv[1][1])
        # Provisional row_h for gap check (median of all gaps above 20)
        raw_cys = [cy for (_, (_, cy)) in sorted_by_cy]
        raw_gaps = [raw_cys[i+1] - raw_cys[i] for i in range(len(raw_cys) - 1)
                    if raw_cys[i+1] - raw_cys[i] > 20]
        prov_row_h = statistics.median(raw_gaps) if raw_gaps else h * 0.06
        min_gap = prov_row_h * 0.6
        kept: dict[str, tuple[float, float]] = {}
        last_idx = -1
        last_cy = -1e9
        for s, (cx, cy) in sorted_by_cy:
            idx = order_idx.get(s, 99)
            if idx <= last_idx:
                _slog.info(f'LayoutDetector OCRAnchored: drop {s} (out-of-order cy={cy:.0f})')
                continue
            if cy - last_cy < min_gap:
                _slog.info(f'LayoutDetector OCRAnchored: drop {s} (cy_gap={cy-last_cy:.0f} < {min_gap:.0f})')
                continue
            kept[s] = (cx, cy)
            last_idx = idx
            last_cy = cy
        eq_labels = kept
        if len(eq_labels) < 3:
            _slog.info('LayoutDetector OCRAnchored: <3 labels after order check')
            return {}

        # Estimate row_h from median cy gap between consecutive labels
        cys = sorted(cy for (_, cy) in eq_labels.values())
        gaps = [cys[i+1] - cys[i] for i in range(len(cys) - 1) if cys[i+1] - cys[i] > 20]
        row_h = int(statistics.median(gaps)) if gaps else int(h * 0.06)
        icon_h = max(26, int(row_h * 0.92))
        cell_w = max(30, int(row_h * 0.72))

        # x search bounds:
        # - x_min = median label cx + offset (skip label text)
        # - x_max = min trait label cx - buffer (trait label is ~centered above its column,
        #          so column starts ~100px to the LEFT of its label → larger buffer)
        x_min_search = int(median_cx + 40)
        # Trait column half-width scales with image width (~8% of w);
        # fixed 100px buffer fails on 1920+ screens where traits span wider.
        trait_buffer = max(100, int(w * 0.08))
        if trait_labels:
            # Cluster trait labels by cx too (filter OCR misreads like
            # "Starship Traits" at cx=513 when real traits are at cx=924)
            t_cxs = [cx for (cx, _) in trait_labels.values()]
            t_median = statistics.median(t_cxs)
            t_clean = [cx for cx in t_cxs if abs(cx - t_median) <= 100]
            if t_clean:
                x_max_search = int(min(t_clean) - trait_buffer)
            else:
                x_max_search = w - 1
        else:
            x_max_search = w - 1

        if x_max_search - x_min_search < cell_w * 2:
            _slog.info('LayoutDetector OCRAnchored: search window too narrow')
            return {}

        _slog.info(f'LayoutDetector OCRAnchored: {len(eq_labels)} EQ labels, '
                   f'row_h={row_h}, cell_w={cell_w}, icon_h={icon_h}, '
                   f'x_search=[{x_min_search},{x_max_search}]')

        # Interpolate cy for slots missing from OCR but present in profile.
        # STO equipment rows are sequential in slot_order at consistent row_h spacing,
        # so a gap in OCR can be filled from neighboring found labels.
        # Insert optional slots (Sec-Def/Hangars/Experimental) at their canonical
        # STO UI positions so linear interpolation counts the correct number of
        # rows between anchors (see docs/sto_slots_rules.md):
        #   - Secondary Deflector: after Deflector, before Engines (Science Vessels)
        #   - Hangars: after Aft Weapons (Carriers)
        #   - Experimental Weapon: after Aft Weapons (Escorts/Destroyers/etc.)
        extended_order: list[str] = []
        for s in slot_order:
            extended_order.append(s)
            if s == 'Deflector' and profile.get('Sec-Def', 0) > 0 and 'Sec-Def' not in extended_order:
                extended_order.append('Sec-Def')
            if s == 'Aft Weapons':
                for opt in ('Hangars', 'Experimental'):
                    if profile.get(opt, 0) > 0 and opt not in extended_order:
                        extended_order.append(opt)
        active_slots = [s for s in extended_order
                        if profile.get(s, SLOT_DEFAULT_COUNTS.get(s, 1)) > 0]
        labeled_idx = [i for i, s in enumerate(active_slots) if s in eq_labels]
        for i, slot in enumerate(active_slots):
            if slot in eq_labels:
                continue
            prev = max((j for j in labeled_idx if j < i), default=-1)
            nxt  = min((j for j in labeled_idx if j > i), default=-1)
            if prev >= 0 and nxt >= 0:
                cy_p = eq_labels[active_slots[prev]][1]
                cy_n = eq_labels[active_slots[nxt]][1]
                cy_int = cy_p + (i - prev) * (cy_n - cy_p) / (nxt - prev)
            elif prev >= 0:
                cy_int = eq_labels[active_slots[prev]][1] + (i - prev) * row_h
            elif nxt >= 0:
                cy_int = eq_labels[active_slots[nxt]][1] - (nxt - i) * row_h
            else:
                continue
            if 0 < cy_int < h:
                eq_labels[slot] = (median_cx, cy_int)
                _slog.info(f'LayoutDetector OCRAnchored: interpolated {slot} cy={cy_int:.0f}')

        # Pass 1: per-row right_edge candidates
        row_info: list[tuple] = []  # (slot_name, cx, cy, y_top, y_bot, right_edge)
        for slot_name, (cx, cy) in eq_labels.items():
            n_default = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            if n_default == 0:
                continue
            y_top = int(cy - icon_h // 2)
            y_bot = int(cy + icon_h // 2)
            right_edge = self._find_row_right_edge(img, y_top, y_bot,
                                                    x_min_search, x_max_search, cell_w)
            row_info.append((slot_name, cx, cy, y_top, y_bot, right_edge))

        # Determine global equipment-column right edge via clustering:
        # group right_edges within ±20px. On some screens the BOFF/ability panel to the
        # right of equipment has MORE rows hitting its edge than the equipment column
        # (e.g. image.png: equipment=3 rows, BOFF panel=6 rows). So pick the LEFTMOST
        # cluster with sufficient support (≥ 25% of rows OR ≥ 2 members).
        edges = sorted([r[5] for r in row_info if r[5] is not None])
        global_right: int | None = None
        if edges:
            clusters: list[list[int]] = []
            for e in edges:
                if clusters and e - clusters[-1][-1] <= 20:
                    clusters[-1].append(e)
                else:
                    clusters.append([e])
            n_rows = len(row_info)
            min_support = max(2, int(n_rows * 0.25))
            # Left-to-right: pick first cluster with enough support
            qualifying = [c for c in clusters if len(c) >= min_support]
            if qualifying:
                best = qualifying[0]  # leftmost qualifying (clusters built L→R)
            else:
                # No cluster meets support → fall back to largest, ties to leftmost
                best = max(clusters, key=lambda c: (len(c), -c[0]))
            global_right = max(best)  # rightmost within winning cluster
            _slog.info(f'LayoutDetector OCRAnchored: right_edge clusters={[len(c) for c in clusters]} '
                       f'(support≥{min_support}) → global_right={global_right}')

        if global_right is None:
            _slog.info('LayoutDetector OCRAnchored: no right_edge candidates')
            return {}

        # Pass 2: generate bboxes using the global right edge for every row.
        # Borrow panel_x_start from the EQ geometry detector (cached) so the
        # diagnostic count scan stops at the 6-cell matrix's left edge instead
        # of bleeding into adjacent panels.
        geom = self._get_eq_geometry(img)
        diag_panel_x_start = geom.panel_x_start if geom is not None else None
        result: dict[str, list] = {}
        for slot_name, cx, cy, y_top, y_bot, _row_re in row_info:
            n_default = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            if n_default == 0:
                continue
            # Cap strictly at profile count. _count_icons_in_row can overshoot
            # because it scans past equipment into adjacent BOFF/character columns
            # (no x_min_search bound). warp_importer truncates layout[slot][:max_count]
            # which drops the RIGHTMOST (correct) slot in STO's right-aligned UI.
            # T6-X tier bonuses are already applied to profile upstream.
            n_icons = n_default
            if n_default > 1:
                self._count_icons_in_row(img, y_top, y_bot, global_right, cell_w,
                                          slot_name,
                                          panel_x_start=diag_panel_x_start)  # log only

            bboxes = []
            for j in range(n_icons):
                bx = max(0, global_right - (j + 1) * cell_w + 2)
                bboxes.append((bx, y_top, cell_w - 4, icon_h))
            bboxes.reverse()
            result[slot_name] = bboxes
            _slog.info(f'  [{slot_name}] cy={cy:.0f} n_icons={n_icons} (profile={n_default})')

        # Merge trait detection on top of EQ (traits use separate geometry — 5-column grid)
        trait_result = self._detect_traits_via_ocr(img, trait_labels, cell_w, icon_h)
        result.update(trait_result)

        return result

    def _detect_traits_via_ocr(self, img: np.ndarray,
                                trait_labels: dict,
                                cell_w: int,
                                icon_h: int) -> dict:
        """
        Trait/reputation detection anchored on OCR trait labels.

        STO trait panels use a 5-column grid BELOW each label:
          Personal Space Traits  — up to 11 icons (5+5+1)
          Starship Traits        — up to 7  icons (5+2)
          Space Reputation       — 5 icons (single row)
          Active Space Rep       — 5 icons (single row)

        Empirical offsets (median across 14 annotated MIXED screens):
          cx(1st icon) ≈ cx(label) − 2.35 × cell_w  (~−85 at cell_w=36)
          cy(1st icon) ≈ cy(label) + 0.80 × icon_h  (~+40 at icon_h=49)
          col_step     ≈ 1.17 × cell_w              (~42 at cell_w=36)
          row_step     ≈ 1.10 × icon_h              (~54 at icon_h=49)
        """
        import statistics

        if not trait_labels:
            return {}

        h, w = img.shape[:2]

        # Filter OCR outliers: some screens have phantom trait labels far from
        # the real trait column (e.g. Nautilus Starship Traits at cx=513 when
        # real column is at cx=924). Cluster by cx, keep median ±2.5×cell_w.
        cxs = [cx for (cx, _) in trait_labels.values()]
        if len(cxs) >= 2:
            median_cx = statistics.median(cxs)
            max_dev = max(cell_w * 2.5, 100)
            clean = {s: (cx, cy) for s, (cx, cy) in trait_labels.items()
                     if abs(cx - median_cx) <= max_dev}
            if clean:
                trait_labels = clean

        # Space and ground trait panels are mutually exclusive per screen.
        # OCR sometimes produces phantom labels from the other group — pick
        # whichever group has more labels and drop the other.
        space_slots = {'Personal Space Traits', 'Starship Traits',
                       'Space Reputation', 'Active Space Rep'}
        ground_slots = {'Personal Ground Traits', 'Ground Reputation',
                        'Active Ground Rep'}
        n_space = sum(1 for s in trait_labels if s in space_slots)
        n_ground = sum(1 for s in trait_labels if s in ground_slots)
        if n_space >= n_ground:
            group_slots = space_slots
            group_order = ['Personal Space Traits', 'Starship Traits',
                           'Space Reputation', 'Active Space Rep']
        else:
            group_slots = ground_slots
            group_order = ['Personal Ground Traits', 'Ground Reputation',
                           'Active Ground Rep']
        trait_labels = {s: v for s, v in trait_labels.items() if s in group_slots}

        # Enforce cy order within the selected group. Drop any label whose cy
        # breaks the ordering (catches OCR misreads that placed a label in
        # the wrong vertical region).
        order_idx = {s: i for i, s in enumerate(group_order)}
        sorted_by_cy = sorted(trait_labels.items(), key=lambda kv: kv[1][1])
        kept: dict[str, tuple[float, float]] = {}
        last_idx = -1
        for s, (cx, cy) in sorted_by_cy:
            idx = order_idx.get(s, 99)
            if idx <= last_idx:
                _slog.info(f'LayoutDetector TraitDetect: drop {s} (out-of-order cy={cy:.0f})')
                continue
            kept[s] = (cx, cy)
            last_idx = idx

        if not kept:
            return {}

        # Extrapolate a missing Space trait label from its two siblings.
        # OCR occasionally drops one label (cx outlier or order-violation). When
        # the other two anchors are reliable, we can place the missing one via
        # linear spacing. Active Rep is rarely extrapolated (appears on few screens).
        if group_slots is space_slots:
            anchors = {
                'Personal Space Traits': kept.get('Personal Space Traits'),
                'Starship Traits':       kept.get('Starship Traits'),
                'Space Reputation':      kept.get('Space Reputation'),
            }
            p, s, r = anchors['Personal Space Traits'], anchors['Starship Traits'], anchors['Space Reputation']
            # Median cx across valid anchors — all trait columns share the same label cx.
            anchor_cxs = [v[0] for v in anchors.values() if v is not None]
            anchor_cx = statistics.median(anchor_cxs) if anchor_cxs else None

            if anchor_cx is not None:
                # Case 1: Starship missing → midpoint of Personal/Rep
                # (empirically Starship.cy lies 0.44-0.52 between Personal.cy and Rep.cy).
                if s is None and p and r:
                    est_cy = (p[1] + r[1]) * 0.48
                    kept['Starship Traits'] = (anchor_cx, est_cy)
                    _slog.info(f'LayoutDetector TraitDetect: extrapolate Starship cy={est_cy:.0f} '
                               f'from Personal={p[1]:.0f}/Rep={r[1]:.0f}')
                # Case 2: Personal missing → reflect below Starship
                # (Personal.cy ≈ Starship.cy - (Rep.cy - Starship.cy); symmetry around S)
                if p is None and s and r:
                    delta = r[1] - s[1]
                    est_cy = s[1] - delta
                    if est_cy > 0:
                        kept['Personal Space Traits'] = (anchor_cx, est_cy)
                        _slog.info(f'LayoutDetector TraitDetect: extrapolate Personal cy={est_cy:.0f} '
                                   f'from Starship={s[1]:.0f}/Rep={r[1]:.0f}')
                # Case 3: Rep missing → reflect above Starship
                if r is None and p and s:
                    delta = s[1] - p[1]
                    est_cy = s[1] + delta * 1.05  # Rep is slightly wider gap
                    if est_cy < h:
                        # Sanity gate: if the predicted Rep icon row sits over a
                        # near-black band, the screen is cropped above the Rep
                        # area (no Rep UI present) — skip the extrapolation
                        # instead of emitting 5 phantom slots that always resolve
                        # to `__empty__` downstream.
                        import cv2 as _cv2
                        y_off = int(icon_h * 0.88)
                        x_off = int(cell_w * 2.27)
                        col_step = max(int(cell_w * 1.135), cell_w + 2)
                        icon_cy = int(est_cy + y_off)
                        by0 = max(0, icon_cy - icon_h // 2)
                        by1 = min(h, icon_cy + icon_h // 2)
                        bx0 = max(0, int(anchor_cx - x_off - cell_w // 2))
                        bx1 = min(w, int(anchor_cx - x_off + 4 * col_step + cell_w // 2))
                        if by1 > by0 and bx1 > bx0:
                            band = img[by0:by1, bx0:bx1]
                            gray = _cv2.cvtColor(band, _cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
                            band_mean = float(gray.mean())
                        else:
                            band_mean = 0.0
                        if band_mean < 15.0:
                            _slog.info(f'LayoutDetector TraitDetect: skip Rep extrapolation — '
                                       f'predicted band too dark (mean={band_mean:.1f}); '
                                       f'screen likely lacks Rep UI')
                        else:
                            kept['Space Reputation'] = (anchor_cx, est_cy)
                            _slog.info(f'LayoutDetector TraitDetect: extrapolate Rep cy={est_cy:.0f} '
                                       f'from Personal={p[1]:.0f}/Starship={s[1]:.0f} '
                                       f'(band mean={band_mean:.1f})')

        # Counts — use game maximums; downstream truncates per profile/tier
        counts = {
            'Personal Space Traits':  11,
            'Starship Traits':         7,  # covers T6-X2; less = truncated downstream
            'Space Reputation':        5,
            'Active Space Rep':        5,
            'Personal Ground Traits': 11,
            'Ground Reputation':       5,
            'Active Ground Rep':       5,
        }

        # Geometry: bbox SIZE matches EQ icons per STO game rule — trait
        # and EQ icons are rendered at identical pixel dimensions on both
        # space and ground screens. Single calibration applies to both
        # groups now that ground_eq_geometry returns correct cell dims
        # (previously ground cell_w was doubled by a Body-label OCR bug;
        # ground ratios then drifted to compensate. Fixed 2026-05-16 —
        # multi-candidate scoring in ground_eq_geometry restores correct
        # cell_w, and space-style ratios work uniformly).
        is_ground = group_slots is ground_slots
        x_off    = -int(cell_w * 2.27)
        y_off    =  int(icon_h * 0.88)
        col_step = max(int(cell_w * 1.135), cell_w + 2)
        row_step = max(int(icon_h * 1.21), icon_h + 2)
        bbox_w   = max(1, cell_w - 2)
        bbox_h   = icon_h
        N_COLS = 5

        result: dict[str, list] = {}
        for slot, (lcx, lcy) in kept.items():
            n = counts.get(slot, 5)
            top_cx = int(lcx + x_off)
            top_cy = int(lcy + y_off)
            bboxes = []
            for i in range(n):
                col = i % N_COLS
                row = i // N_COLS
                bx = top_cx - bbox_w // 2 + col * col_step
                by = top_cy - bbox_h // 2 + row * row_step
                if 0 <= bx <= w - bbox_w and 0 <= by <= h - bbox_h:
                    bboxes.append((bx, by, bbox_w, bbox_h))
            if bboxes:
                result[slot] = bboxes
                _slog.info(f'  [{slot}] n={len(bboxes)} anchor=({lcx:.0f},{lcy:.0f}) '
                           f'top=({top_cx},{top_cy}) bbox={bbox_w}x{bbox_h} '
                           f'step=({col_step},{row_step}) group={"ground" if is_ground else "space"}')
        return result

    def _detect_via_ocr(self, img, slot_order, profile):
        try: results = self._get_ocr().readtext(img)
        except: return {}
        h, w = img.shape[:2]
        panel_right, row_h_est = self._find_panel_right_edge(img), int(h * 0.068)
        cell_w, icon_h = max(30, int(row_h_est * 0.80)), max(26, int(row_h_est * 0.78))
        found = {}
        for (bbox_pts, text, conf) in results:
            if conf < config.OCR_CONF_THRESHOLD: continue
            can = self._match_label(text.strip().lower())
            if not can or can not in slot_order: continue
            n_icons = profile.get(can, SLOT_DEFAULT_COUNTS.get(can, 1))
            if n_icons == 0: continue
            iy, bboxes = int(np.mean([pt[1] for pt in bbox_pts])) - icon_h // 2, []
            for j in range(n_icons): bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, cell_w - 4, icon_h))
            bboxes.reverse(); found[can] = bboxes
        return found

    def _match_label(self, text_lower: str) -> str | None:
        return match_slot_label(text_lower)

    SPACE_ANCHORS_REL: dict[str, tuple[float, int]] = {
        'Fore Weapons': (0.036, 5), 'Deflector': (0.107, 1), 'Engines': (0.178, 1), 'Warp Core': (0.249, 1), 'Shield': (0.325, 1),
        'Aft Weapons': (0.401, 4), 'Devices': (0.475, 4), 'Universal Consoles': (0.547, 2), 'Engineering Consoles': (0.620, 4),
        'Science Consoles': (0.695, 2), 'Tactical Consoles': (0.768, 4), 'Hangar': (0.840, 1),
    }

    def _detect_via_anchors(self, img, slot_order, profile):
        h, w = img.shape[:2]
        panel_right, row_h_est = self._find_panel_right_edge(img), int(h * 0.072)
        cell_w, icon_h = max(30, int(row_h_est * 0.80)), max(26, int(row_h_est * 0.78))

        # Load canonical learned Y values; fall back to hardcoded SPACE_ANCHORS_REL
        canonical = self._load_canonical_layout()
        can_slots = {}
        if canonical:
            # Use build_type='SPACE' as best general fallback for equipment screens
            can_slots = canonical.get('types', {}).get('SPACE', {}).get('slots', {})

        cal = (self._calibration or {}).get('SPACE', {})
        result = {}
        for slot_name in slot_order:
            # Priority: canonical learned > hardcoded
            if slot_name in can_slots:
                y_rel    = can_slots[slot_name]['y_rel']
                n_default = SLOT_DEFAULT_COUNTS.get(slot_name, 1)
            else:
                anchor = cal.get(slot_name, self.SPACE_ANCHORS_REL.get(slot_name))
                if anchor is None:
                    continue
                y_rel, n_default = anchor if isinstance(anchor, tuple) else (anchor, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            n_icons = profile.get(slot_name, n_default)
            if n_icons == 0:
                continue
            iy = int(h * y_rel) - icon_h // 2
            bboxes = []
            for j in range(n_icons):
                bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, cell_w - 4, icon_h))
            bboxes.reverse()
            result[slot_name] = bboxes
        return result

    def _fill_gaps(self, found, slot_order, img, profile):
        h, w = img.shape[:2]
        panel_right, row_h_est = self._find_panel_right_edge(img), int(h * 0.068)
        cell_w, icon_h = max(30, int(row_h_est * 0.80)), max(26, int(row_h_est * 0.78))
        result, order_map = dict(found), {name: i for i, name in enumerate(slot_order)}
        anchored = sorted([(order_map[name], bboxes[0][1] + bboxes[0][3] // 2) for name, bboxes in found.items() if bboxes and name in order_map])
        if len(anchored) < 2: return result
        for slot_name in slot_order:
            if slot_name in result: continue
            idx = order_map.get(slot_name)
            n_icons = profile.get(slot_name, SLOT_DEFAULT_COUNTS.get(slot_name, 1))
            if idx is None or n_icons == 0: continue
            before, after = [i for i in anchored if i[0] < idx], [i for i in anchored if i[0] > idx]
            if before and after: cy = int(before[-1][1] + (idx - before[-1][0]) / max(1, after[0][0] - before[-1][0]) * (after[0][1] - before[-1][1]))
            elif before: cy = int(before[-1][1] + (idx - before[-1][0]) * ((before[-1][1] - before[-2][1]) / max(1, before[-1][0] - before[-2][0]) if len(before) >= 2 else row_h_est))
            elif after: cy = int(after[0][1] - (after[0][0] - idx) * ((after[1][1] - after[0][1]) / max(1, after[1][0] - after[0][0]) if len(after) >= 2 else row_h_est))
            else: continue
            iy, bboxes = cy - icon_h // 2, []
            for j in range(n_icons): bboxes.append((max(0, panel_right - (j + 1) * cell_w + 2), iy, cell_w - 4, icon_h))
            bboxes.reverse(); result[slot_name] = bboxes
        return result

    def _get_ocr(self):
        # One reader per process, not one per component — see
        # `text_extractor.shared_reader`.
        if self._ocr is None:
            from warp.recognition.text_extractor import shared_reader
            self._ocr = shared_reader()
        return self._ocr

    def _load_calibration(self) -> dict | None:
        cfile = _userdata.training_data_dir() / CALIBRATION_FILENAME
        if cfile.exists():
            try:
                return json.loads(cfile.read_text())
            except Exception:
                return None
        return None

    def _save_calibration(self):
        cfile = _userdata.training_data_dir() / CALIBRATION_FILENAME
        cfile.write_text(json.dumps(self._calibration, indent=2))
