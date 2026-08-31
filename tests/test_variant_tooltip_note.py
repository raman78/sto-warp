"""The tooltip says when the picture is the 23rd-century art.

Cargo has one row for such an item and the wiki has two pictures, so the
variant is not a different item — and its tag must never reach the name, which
goes to the build writer and on to SETS, where only the cargo name exists.

It is still worth saying: a player who wants the exact weapon in the
screenshot needs to know they are looking for the 23c version of it. Without
the note, a blue icon under a name whose usual art is red reads as a
misdetection.

Run standalone:
    python -m pytest tests/test_variant_tooltip_note.py -v
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip('PySide6')

NAME = 'Phaser Dual Heavy Cannons'
VARIANT = 'Phaser Dual Heavy Cannons (23c)'


def _text(html: str) -> str:
    return re.sub(r'<[^>]+>', ' ', html)


def _tooltip(**kw):
    from warp.gui import slot_tooltip_html

    return slot_tooltip_html('Fore Weapons', NAME, 0.60, **kw)


def test_the_variant_tag_is_shown():
    assert 'art: 23c' in _text(_tooltip(variant=VARIANT))


def test_nothing_is_added_for_an_item_with_one_picture():
    assert 'art:' not in _text(_tooltip())


def test_the_note_never_alters_the_item_name():
    """The name is what reaches SETS; the tag is display only."""
    body = _text(_tooltip(variant=VARIANT))

    assert NAME in body
    assert f'{NAME} (23c)' not in body


def test_only_the_distinguishing_part_is_shown():
    """The rest of the filename is the name again."""
    from warp.gui import _variant_note

    assert _variant_note(NAME, VARIANT) == (
        '<span style="color:#888">art: 23c</span>')


def test_a_variant_equal_to_the_name_says_nothing():
    from warp.gui import _variant_note

    assert _variant_note(NAME, NAME) == ''


def test_a_missing_name_is_harmless():
    from warp.gui import _variant_note

    assert _variant_note('', VARIANT) == ''
