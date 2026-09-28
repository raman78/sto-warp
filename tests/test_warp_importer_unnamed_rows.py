"""Equipment rows whose type is a guess or Unknown are said to be.

`LayoutDetector.last_row_guesses` names the rows the detector could only
guess, or not type at all. The importer turns it into a line in
`ImportResult.errors`, which WARP CORE shows under the review list, and
stamps each item with `row_guess` so the review list can mark its group.

Offline: no image.
"""
from __future__ import annotations

import warp.warp_importer as wi


def test_nothing_to_say_when_every_row_was_named():
    assert wi._row_guess_message({}) == ''
    assert wi._row_guess_message(None) == ''


def test_guessed_and_unknown_rows_are_listed():
    msg = wi._row_guess_message({'Devices': 'guess', 'Universal Consoles': 'guess',
                                 'Unknown': 'unknown'})
    assert '2 guessed (Devices, Universal Consoles)' in msg
    assert '1 Unknown' in msg
    assert 'Mark Done' in msg


def test_an_item_carries_how_sure_its_row_is():
    item = wi.RecognisedItem(slot='Unknown', slot_index=0, name='', confidence=0.0,
                             row_guess='unknown')
    assert item.row_guess == 'unknown'
    assert wi.RecognisedItem(slot='Devices', slot_index=0, name='', confidence=0.0).row_guess == ''
