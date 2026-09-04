"""Qualified icon art is folded onto the item it depicts.

STO draws some gear differently in 23rd-century content, and the wiki files
that second picture under its own name — `Impulse Engines (23c)` beside
`Impulse Engines` — while the item keeps one name and one cargo row. The icon
index is keyed by filename, so without folding the variant enters it under a
name no cargo row carries and every candidate filter downstream drops it.

The tag is not a reliable variant marker on its own: `Modified Phaser Pistol
(23c.)` is a real item name. Cargo decides, which is also what makes the rule
survive the wiki gaining or renaming 23c items later.

The era tag was only the first of these. The wiki also qualifies art by
environment, colour, faction and reputation, and cargo carries none of those
in the item name — measured 2026-09-04 over the 4406-icon library, 12 such
files entered the gallery under a label cargo has never heard of, two of them
reaching confirmed training data. Same rule, wider tag.
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


# ── Qualifiers other than the era tag ─────────────────────────────────
#
# Measured on the real library (2026-09-04): of 155 icons carrying a
# parenthesised tag, 109 are cargo names outright and 46 fold onto a base.
# None falls outside those two cases, which is what makes the wider rule safe.

ENV_CARGO = {
    'Adaptive Defense',
    'Fire on my Mark',
    'Liberated Borg Kingdom Nanoprobes',
    'Sniper',
}


def test_an_environment_qualifier_folds_onto_the_base():
    """Cargo stores the space and ground traits as two rows under one `name`,
    so one name is all this program can emit for either."""
    assert _base_item_name('Adaptive Defense (ground)', ENV_CARGO) == 'Adaptive Defense'
    assert _base_item_name('Adaptive Defense (space)', ENV_CARGO) == 'Adaptive Defense'


def test_the_qualifier_is_matched_whatever_its_case():
    """The library spells it both ways — `(ground)` and `(Ground)`."""
    assert _base_item_name('Fire on my Mark (Ground)', ENV_CARGO) == 'Fire on my Mark'


def test_a_reputation_qualifier_folds():
    assert (_base_item_name('Slippery Target (Lukari Reputation)',
                            {'Slippery Target'}) == 'Slippery Target')


def test_a_starship_qualifier_folds():
    assert _base_item_name('Sniper (starship)', ENV_CARGO) == 'Sniper'


def test_a_colour_qualifier_folds():
    assert _base_item_name('Some Kit (Red)', {'Some Kit'}) == 'Some Kit'


def test_a_qualified_name_cargo_carries_is_still_left_alone():
    """The widening must not start folding names whose parenthesis is part of
    the item — 109 of the library's 155 tagged icons are in this state."""
    cargo = {'Modified Phaser Pistol (23c.)', 'Modified Phaser Pistol'}

    assert (_base_item_name('Modified Phaser Pistol (23c.)', cargo)
            == 'Modified Phaser Pistol (23c.)')


def test_a_qualifier_whose_base_is_unknown_is_left_alone():
    """Widening the tag must not widen what counts as a fold target."""
    assert (_base_item_name('Unheard Of (space)', ENV_CARGO)
            == 'Unheard Of (space)')


def test_only_a_trailing_qualifier_counts():
    """A parenthesis mid-name is not a tag."""
    assert (_base_item_name('Console (Mk XII) Booster', ENV_CARGO)
            == 'Console (Mk XII) Booster')
