"""A slot box never sits on a label.

`drop_boxes_on_text` is the last thing `LayoutDetector.detect` does. A
projected box that lands on writing is wrong twice over: there is no icon
under it, and on a screenshot it was read as a blank cell and auto-confirmed
as `__inactive__` at confidence 1.00 — teaching the models that a heading is
an empty slot.

The hard part is that the same text means opposite things in two panels. The
space equipment column writes a row's label *beside* its icons, so sharing a
band with a label is normal there; the trait panel writes a section heading
*above* its block, so a box under one has run past the end of its section.

Offline: tokens are dicts, boxes are tuples, no screenshot and no OCR.
"""
from __future__ import annotations

import numpy as np
import pytest

from warp.recognition.layout_detector import (drop_boxes_on_text,
                                              is_slot_label_text,
                                              match_slot_label)


def tok(text, x0, y0, x1, y1):
    return {'text': text, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'w': x1 - x0, 'h': y1 - y0, 'low': text.lower()}


IMG_H = 536


def blank(h=IMG_H, w=960):
    """A screenshot with nothing on it — no icon blob to snap to."""
    return np.zeros((h, w, 3), np.uint8)


def with_icon(x, y, w, h, img=None):
    """Paint a lit rectangle where the game would draw an icon."""
    img = blank() if img is None else img
    img[y:y + h, x:x + w] = 200
    return img


# ── What counts as label text ─────────────────────────────────────────────

def test_a_wrapped_line_is_label_text():
    """`Engineering Consoles` does not fit the column, so the game breaks it
    and the reader returns each line on its own."""
    assert is_slot_label_text('engineering')
    assert is_slot_label_text('consoles')


def test_the_ship_name_divider_is_not_label_text():
    """`U.S.S. FURY Traits` — read as `FuAY Traits` — sits inside the starship
    trait block and names no section."""
    assert not is_slot_label_text('fuay traits')


def test_the_alias_table_answers_for_labels_that_are_not_slot_names():
    """The engines row is labelled `Impulse` and the shield row `Shields`.
    Anything comparing OCR text to the slot name gets both wrong."""
    assert match_slot_label('impulse') == 'Engines'
    assert match_slot_label('shields') == 'Shield'
    assert match_slot_label('engineering') == 'Engineering Consoles'


# ── Space equipment: the label shares the row's line ──────────────────────

def test_a_space_row_keeps_its_wrapped_own_label():
    """Measured on `image-73df6ced341f3b5f.png`: `Engineering` bands the row
    it names, and the row was moved 24 px below its grid."""
    boxes = [(731, 369, 31, 41), (765, 369, 31, 41)]
    out = drop_boxes_on_text({'Engineering Consoles': boxes},
                             [tok('Engineering', 612, 372, 695, 392)], blank())
    assert out['Engineering Consoles'] == boxes


def test_a_space_row_keeps_a_label_that_is_not_its_name():
    boxes = [(865, 95, 31, 41)]
    out = drop_boxes_on_text({'Engines': boxes},
                             [tok('Impulse', 600, 98, 693, 114)], blank())
    assert out['Engines'] == boxes


def test_one_word_never_drops_a_row():
    """`Weapons` resolves to `Aft Weapons`, and is also the second line of the
    fore weapons label. Dropping on a one-word match would delete the fore
    weapons row of every space screenshot."""
    boxes = [(765, 2, 31, 41)]
    out = drop_boxes_on_text({'Fore Weapons': boxes},
                             [tok('Weapons', 631, 23, 695, 39)], blank())
    assert out['Fore Weapons'] == boxes


# ── Traits: the heading is a section boundary ─────────────────────────────
#
# Sections are given their real rows here, because that is what the guard
# measures against: a heading ends a section when it is printed over that
# section's columns. A one-box section spans one column and almost nothing
# lies over it.

def _row(y, n=5, x0=661, w=27, h=36, pitch=32):
    return [(x0 + i * pitch, y, w, h) for i in range(n)]


PST = _row(230) + _row(273) + [(661, 316, 27, 36)]   # 11th box, one too many


def test_a_box_on_another_sections_heading_goes():
    """The eleventh `Personal Space Traits` box landed level with the
    `Starship Traits` heading; the screenshot shows ten."""
    out = drop_boxes_on_text({'Personal Space Traits': list(PST)},
                             [tok('Starship Traits', 691, 329, 785, 345)],
                             blank())
    assert out['Personal Space Traits'] == PST[:10]


def test_a_heading_beside_the_box_still_drops_it():
    """It is not enough to ask where the text sits: that box ends 3 px before
    the heading begins, so 'beside' alone would have kept it."""
    heading = tok('Starship Traits', 691, 329, 785, 345)
    box = PST[-1]
    assert box[0] + box[2] < heading['x0']          # no overlap at all
    out = drop_boxes_on_text({'Personal Space Traits': list(PST)},
                             [heading], blank())
    assert box not in out['Personal Space Traits']


def test_an_emptied_section_is_still_reported():
    """So a caller can tell 'this section has no boxes' from 'this section was
    never detected'."""
    out = drop_boxes_on_text({'Personal Space Traits': [(661, 316, 27, 36),
                                                       (789, 316, 27, 36)]},
                             [tok('Starship Traits', 691, 329, 785, 345)],
                             blank())
    assert out['Personal Space Traits'] == []


# ── A divider moves the row instead of losing it ──────────────────────────

def test_a_row_on_a_divider_is_moved_below_it():
    boxes = _row(350) + [(692, 393, 27, 36)]
    out = drop_boxes_on_text({'Starship Traits': boxes},
                             [tok('FuAY Traits', 723, 405, 797, 419)], blank())
    x, y, w, h = out['Starship Traits'][-1]
    assert (x, w, h) == (692, 27, 36)
    assert y > 405


def test_a_row_with_nowhere_to_go_is_dropped():
    boxes = _row(450) + [(692, 500, 27, 36)]
    out = drop_boxes_on_text({'Starship Traits': boxes},
                             [tok('FuAY Traits', 723, 505, 797, 530)],
                             blank(540))
    assert out['Starship Traits'] == boxes[:5]


# ── Only bands, never marks ───────────────────────────────────────────────

def test_text_printed_inside_an_icon_is_not_a_band():
    """The game prints `Mk XV` and `LOC` on the artwork. Measured on
    `image-4391ccd9d2683d4e.png`: headings run 74-132 px against a 27 px
    slot, in-icon marks 28-46 px."""
    boxes = [(791, 350, 27, 36)]
    out = drop_boxes_on_text({'Starship Traits': boxes},
                             [tok('LoC', 791, 363, 821, 375)], blank())
    assert out['Starship Traits'] == boxes


def test_a_band_is_local_to_its_column():
    """A screenshot holds several panels side by side. `Personal Space Traits`
    over the trait column says nothing about the equipment column 200 px to
    its left — treating every wide token as full width deleted 27 good boxes
    on the first attempt at this."""
    boxes = [(361, 209, 31, 41)]
    out = drop_boxes_on_text({'Aft Weapons': boxes},
                             [tok('Personal Space Traits', 673, 209, 805, 225)],
                             blank())
    assert out['Aft Weapons'] == boxes


# ── Where a moved box lands ───────────────────────────────────────────────

def test_a_moved_box_lands_on_the_icon_not_under_the_writing():
    """Clearing the text is not the same as landing on the artwork. On
    `image-4391ccd9d2683d4e.png` the divider ends at y=419 and the real icons
    start at y=425, so a box parked one pixel under the writing sat 5 px high
    and 9 px short of the icon's bottom, matching at 0.64."""
    img = with_icon(692, 425, 30, 40)
    out = drop_boxes_on_text(
        {'Starship Traits': _row(350) + [(692, 393, 27, 36)]},
        [tok('FuAY Traits', 723, 405, 797, 419)], img)
    _x, y, _w, _h = out['Starship Traits'][-1]
    assert 423 <= y <= 429


def test_a_row_moves_as_a_row():
    """An unlit cell has no icon to snap to — `__inactive__` slots are drawn
    dark on purpose. Measured on `image-817e2e37c01aed8c.png`: one box of the
    pair found its icon and the other stayed 8 px high, splitting a row the
    game draws on one line."""
    img = with_icon(692, 425, 30, 40)          # only the left cell is lit
    out = drop_boxes_on_text(
        {'Starship Traits': _row(350) + [(692, 393, 27, 36),
                                         (724, 393, 27, 36)]},
        [tok('FuAY Traits', 700, 405, 797, 419)], img)
    tops = {b[1] for b in out['Starship Traits'][5:]}
    assert len(tops) == 1
    assert 423 <= tops.pop() <= 429


def test_a_blank_screenshot_still_moves_the_box_clear():
    """With no blob to snap to the old behaviour stands: below the writing."""
    out = drop_boxes_on_text(
        {'Starship Traits': _row(350) + [(692, 393, 27, 36)]},
        [tok('FuAY Traits', 723, 405, 797, 419)], blank())
    _x, y, _w, _h = out['Starship Traits'][-1]
    assert y == 420


# ── Nothing to do ─────────────────────────────────────────────────────────

@pytest.mark.parametrize('tokens', [None, []])
def test_no_tokens_leaves_the_layout_alone(tokens):
    res = {'Devices': [(698, 277, 31, 41)]}
    assert drop_boxes_on_text(res, tokens, blank()) is res


# ── Two slots are never the same icon ─────────────────────────────────────

def test_a_box_is_dropped_rather_than_moved_onto_another_section():
    """The heading that ends a section is normally what stops a projected row
    from running on — but the reader has to get it right. On
    `image-8ee54291302414af.png` `Starship Traits` came back as
    `'Sterehlp Trelte'`, which names no slot, so the band read as an ordinary
    divider and the eleventh personal-trait box was moved onto the first
    starship trait. Geometry says what the text could not."""
    img = with_icon(831, 221, 42, 56)
    res = {
        'Personal Space Traits': _row(49, x0=831, w=41, h=52, pitch=48)
                                 + _row(111, x0=831, w=41, h=52, pitch=48)
                                 + [(831, 173, 41, 52)],
        'Starship Traits': _row(221, x0=831, w=42, h=56, pitch=48),
    }
    out = drop_boxes_on_text(
        res, [tok('Sterehlp Trelte', 880, 190, 1016, 214)], img)
    assert len(out['Personal Space Traits']) == 10
    assert out['Starship Traits'] == res['Starship Traits']


def test_a_move_into_free_space_is_still_allowed():
    """Same shape, but nothing owns the destination."""
    img = with_icon(831, 221, 42, 56)
    res = {'Personal Space Traits': _row(49, x0=831, w=41, h=52, pitch=48)
                                    + [(831, 173, 41, 52)]}
    out = drop_boxes_on_text(
        res, [tok('Sterehlp Trelte', 880, 190, 1016, 214)], img)
    assert len(out['Personal Space Traits']) == 6
    assert out['Personal Space Traits'][-1][1] > 173
