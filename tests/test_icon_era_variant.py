"""Era-variant icon art is folded onto the item it depicts.

STO draws some gear differently in 23rd-century content, and the wiki files
that second picture under its own name — `Impulse Engines (23c)` beside
`Impulse Engines` — while the item keeps one name and one cargo row. The icon
index is keyed by filename, so without folding the variant enters it under a
name no cargo row carries and every candidate filter downstream drops it.

The tag is not a reliable variant marker on its own: `Modified Phaser Pistol
(23c.)` is a real item name. Cargo decides, which is also what makes the rule
survive the wiki gaining or renaming 23c items later.
"""
from __future__ import annotations

from warp.recognition.icon_matcher import _base_item_name

# A slice of the real picture, verified against the live wiki + cargo cache
# on 2026-08-22: 35 `(23c)` files, none of them a cargo name; one `(23c.)`
# file whose full name is a cargo name.
CARGO = {
    'Impulse Engines',
    'Phaser Cannon',
    'Shield Array',
    'Modified Phaser Pistol (23c.)',
}


# ── Variant art folds onto its item ───────────────────────────────────

def test_a_variant_folds_onto_the_base_item():
    assert _base_item_name('Impulse Engines (23c)', CARGO) == 'Impulse Engines'


def test_folding_works_for_every_item_that_has_variant_art():
    for base in ('Phaser Cannon', 'Shield Array'):
        assert _base_item_name(f'{base} (23c)', CARGO) == base


def test_a_plain_icon_name_is_untouched():
    assert _base_item_name('Impulse Engines', CARGO) == 'Impulse Engines'


# ── …but a tagged *item* name is left alone ───────────────────────────

def test_a_tag_that_is_part_of_the_item_name_survives():
    # Folding this one would trade a working item for a name cargo does not
    # have — the exact regression the cargo check exists to prevent.
    assert (_base_item_name('Modified Phaser Pistol (23c.)', CARGO)
            == 'Modified Phaser Pistol (23c.)')


def test_a_future_dotless_tag_that_becomes_an_item_name_survives():
    # The wiki spells the marker both ways; if a `(23c)` name ever turns into
    # a real item, being an item wins over looking like a variant.
    cargo = CARGO | {'Some Relic (23c)'}
    assert _base_item_name('Some Relic (23c)', cargo) == 'Some Relic (23c)'


# ── Nothing to fold onto ──────────────────────────────────────────────

def test_a_variant_of_an_unknown_item_is_left_as_is():
    # `Matter Anti-Matter Warp Core (23c)` is in this state today: variant art
    # for an item with no cargo row. Left alone, it simply never wins.
    assert (_base_item_name('Matter Anti-Matter Warp Core (23c)', CARGO)
            == 'Matter Anti-Matter Warp Core (23c)')


def test_a_new_23c_item_starts_folding_once_cargo_carries_it():
    # The rule is data-driven on purpose: the wiki adding the missing item is
    # all it takes, with no code change here.
    name = 'Matter Anti-Matter Warp Core (23c)'
    grown = CARGO | {'Matter Anti-Matter Warp Core'}
    assert _base_item_name(name, grown) == 'Matter Anti-Matter Warp Core'


def test_without_cargo_names_nothing_is_folded():
    # Pre-existing behaviour when the cargo cache is missing — degrade to the
    # old filename-is-the-name rule rather than guessing.
    assert (_base_item_name('Impulse Engines (23c)', set())
            == 'Impulse Engines (23c)')


# ── Shapes that must not be mistaken for a tag ────────────────────────

def test_an_unrelated_parenthesis_is_not_a_tag():
    cargo = {'Console - Universal - Something (Rare)'}
    name = 'Console - Universal - Something (Rare)'
    assert _base_item_name(name, cargo) == name


def test_a_bare_tag_with_no_item_in_front_is_left_alone():
    assert _base_item_name('(23c)', CARGO) == '(23c)'
