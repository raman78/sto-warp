"""Quick verification that German UI translations flow into the recognition
pipeline's keyword tables.

This is a pure unit-style smoke test — no images, no EasyOCR. It exercises:
  - ui_translations CSV loads cleanly
  - eq_geometry classifier tables (SINGLE_LINE_KW / FIRST_LINE_KW /
    SECOND_LINE_KW) include German tokens
  - text_extractor screen-type classifier sees German space-EQ labels and
    BOFF/screen headers
  - layout_detector SLOT_LABEL_ALIASES accepts German row labels
  - ground_eq_geometry OCR_KEYWORD_TO_SLOT (untouched here, English-only ground)

Run:
    python -m tests.diag_german_ui
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(label: str, ok: bool, detail: str = '') -> bool:
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {label}' + (f'  ({detail})' if detail else ''))
    return ok


def main() -> int:
    print('1. ui_translations CSV loads')
    from warp.recognition.ui_translations import (
        normalize_map, synonyms, ocr_languages, augment_substring_phrases,
    )
    space_slot = normalize_map('space_slot')
    eq_first = normalize_map('eq_word_first')
    eq_second = normalize_map('eq_word_second')
    eq_single = normalize_map('eq_word_single')
    headers = normalize_map('screen_header')
    ok = True
    ok &= check('CSV non-empty', bool(space_slot), f'{len(space_slot)} space_slot entries')
    # Basic slot lookups
    ok &= check("'bug-waffen' → 'Fore Weapons'",
                space_slot.get('bug-waffen') == 'Fore Weapons')
    ok &= check("'bug waffen' → 'Fore Weapons'",
                space_slot.get('bug waffen') == 'Fore Weapons')
    ok &= check("'deflektor' (single) → 'Deflector'",
                eq_single.get('deflektor') == 'Deflector')
    ok &= check("'bug' (first) → 'Fore'",
                eq_first.get('bug') == 'Fore')
    ok &= check("'waffen' (second) → 'Weapons'",
                eq_second.get('waffen') == 'Weapons')
    ok &= check("'raumstationen' (header) → 'space stations'",
                headers.get('raumstationen') == 'space stations')
    ok &= check("OCR languages include 'de'", 'de' in ocr_languages(),
                f'languages={ocr_languages()}')
    # Consoles — the main gap the user flagged
    ok &= check("'konsolen' (second) → 'Consoles'",
                eq_second.get('konsolen') == 'Consoles')
    ok &= check("'taktik' (first) → 'Tactical'",
                eq_first.get('taktik') == 'Tactical')
    ok &= check("'technik' (first) → 'Engineering'",
                eq_first.get('technik') == 'Engineering')
    ok &= check("'wissenschaft' (first) → 'Science'",
                eq_first.get('wissenschaft') == 'Science')
    ok &= check("'taktikkonsolen' (space_slot) → 'Tactical Consoles'",
                space_slot.get('taktikkonsolen') == 'Tactical Consoles')
    ok &= check("'universalkonsolen' (space_slot) → 'Universal Consoles'",
                space_slot.get('universalkonsolen') == 'Universal Consoles')
    # Secondary Deflector
    ok &= check("'sekundärdeflektor' (space_slot) → 'Sec-Def'",
                space_slot.get('sekundärdeflektor') == 'Sec-Def')
    ok &= check("'sekundardeflektor' (space_slot) → 'Sec-Def'",
                space_slot.get('sekundardeflektor') == 'Sec-Def')
    # Experimental + Hangar
    ok &= check("'experimentell' (single) → 'Experimental'",
                eq_single.get('experimentell') == 'Experimental')
    ok &= check("'experimentelle waffe' (space_slot) → 'Experimental'",
                space_slot.get('experimentelle waffe') == 'Experimental')
    ok &= check("'hangar' (single) → 'Hangars'",
                eq_single.get('hangar') == 'Hangars')
    # Impulse / Singularity (what OCR actually sees, not SETS names)
    ok &= check("'impuls' (single) → 'Engines'",
                eq_single.get('impuls') == 'Engines')
    ok &= check("'singularität' (single) → 'Warp Core'",
                eq_single.get('singularität') == 'Warp Core'
                or eq_single.get('singularitat') == 'Warp Core',
                f'singularität={eq_single.get("singularität")!r} / '
                f'singularitat={eq_single.get("singularitat")!r}')

    print('\n2. eq_geometry classifier tables augmented')
    from warp.recognition import eq_geometry as eq
    ok &= check("SINGLE_LINE_KW['deflektor'] == 'Deflector'",
                eq.SINGLE_LINE_KW.get('deflektor') == 'Deflector')
    ok &= check("SINGLE_LINE_KW['impuls'] == 'Engines'",
                eq.SINGLE_LINE_KW.get('impuls') == 'Engines')
    ok &= check("SINGLE_LINE_KW['experimentell'] == 'Experimental'",
                eq.SINGLE_LINE_KW.get('experimentell') == 'Experimental')
    ok &= check("FIRST_LINE_KW['bug'] == 'Fore'",
                eq.FIRST_LINE_KW.get('bug') == 'Fore')
    ok &= check("FIRST_LINE_KW['taktik'] == 'Tactical'",
                eq.FIRST_LINE_KW.get('taktik') == 'Tactical')
    ok &= check("FIRST_LINE_KW['technik'] == 'Engineering'",
                eq.FIRST_LINE_KW.get('technik') == 'Engineering')
    ok &= check("SECOND_LINE_KW['waffen'] == 'Weapons'",
                eq.SECOND_LINE_KW.get('waffen') == 'Weapons')
    ok &= check("SECOND_LINE_KW['konsolen'] == 'Consoles'",
                eq.SECOND_LINE_KW.get('konsolen') == 'Consoles')
    ok &= check("GERMAN_ORDER has 'Bug-Waffen' or 'Bug Waffen' (row 0)",
                eq.GERMAN_ORDER.get('Bug-Waffen') == 0
                or eq.GERMAN_ORDER.get('Bug Waffen') == 0)
    ok &= check("GERMAN_ORDER has 'Taktikkonsolen' (row 12)",
                eq.GERMAN_ORDER.get('Taktikkonsolen') == 12)
    ok &= check("GERMAN_ORDER has Sec-Def variant (row 4)",
                any(v == 4 for k, v in eq.GERMAN_ORDER.items()
                    if 'deflektor' in k.lower() and 'sekund' in k.lower()))

    print('\n3. text_extractor screen-type classifier sees German')
    from warp.recognition.text_extractor import _detect_type_from_text
    # Simulated SPACE_EQ German screenshot OCR lines.
    de_space = ['bug-waffen', 'heck-waffen', 'deflektor', 'impuls',
                'warp', 'schilde', 'geraete', 'taktikkonsolen']
    st = _detect_type_from_text(de_space)
    ok &= check(f"_detect_type_from_text(de SPACE_EQ tokens) → SPACE",
                st == 'SPACE', f'got {st!r}')
    # Simulated BOFFS screen — raumstationen as space-boff header.
    de_boff = ['raumstationen', 'brückenoffizier']
    st = _detect_type_from_text(de_boff)
    ok &= check(f"_detect_type_from_text(de BOFFS tokens) → SPACE_BOFFS",
                st == 'SPACE_BOFFS', f'got {st!r}')
    # German traits
    de_traits = ['persönliche raumeigenschaften', 'raumschiffeigenschaften']
    st = _detect_type_from_text(de_traits)
    ok &= check(f"_detect_type_from_text(de TRAITS tokens) → SPACE_TRAITS",
                st == 'SPACE_TRAITS', f'got {st!r}')

    print('\n4. layout_detector SLOT_LABEL_ALIASES augmented')
    from warp.recognition.layout_detector import SLOT_LABEL_ALIASES
    ok &= check("SLOT_LABEL_ALIASES has 'bug-waffen' or 'bug waffen'",
                SLOT_LABEL_ALIASES.get('bug-waffen') == 'Fore Weapons'
                or SLOT_LABEL_ALIASES.get('bug waffen') == 'Fore Weapons')
    ok &= check("SLOT_LABEL_ALIASES has some Heck variant",
                any('heck' in k and SLOT_LABEL_ALIASES[k] == 'Aft Weapons'
                    for k in SLOT_LABEL_ALIASES))
    ok &= check("SLOT_LABEL_ALIASES['taktikkonsolen'] == 'Tactical Consoles'",
                SLOT_LABEL_ALIASES.get('taktikkonsolen') == 'Tactical Consoles')
    ok &= check("SLOT_LABEL_ALIASES['raumstationen'] == 'Boff Tactical'",
                SLOT_LABEL_ALIASES.get('raumstationen') == 'Boff Tactical')

    print('\n5. augment_substring_phrases preserves English + adds DE synonyms')
    phrases = augment_substring_phrases('screen_header', ('space stations',))
    ok &= check("'space stations' preserved", 'space stations' in phrases)
    ok &= check("'raumstationen' added", 'raumstationen' in phrases)
    eq_phrases = augment_substring_phrases('space_slot', ('impulse',))
    ok &= check("'impulse' preserved", 'impulse' in eq_phrases)
    ok &= check("'impuls' added for impulse", 'impuls' in eq_phrases)

    print('\n6. Ship type translation (German → English)')
    from warp.recognition.ui_translations import translate_ship_type
    ok &= check("Pakled-Wundertäter-Klumpenschiff → Pakled Miracle Worker Clumpship",
                translate_ship_type('Pakled-Wundertäter-Klumpenschiff')
                == 'Pakled Miracle Worker Clumpship')
    ok &= check("Umlaut-stripped variant also works",
                translate_ship_type('Pakled-Wundertater-Klumpenschiff')
                == 'Pakled Miracle Worker Clumpship')
    ok &= check("Schwerer Schlachtkreuzer → Heavy Battlecruiser",
                translate_ship_type('Schwerer Schlachtkreuzer')
                == 'Heavy Battlecruiser')
    ok &= check("English passthrough unchanged",
                translate_ship_type('Temporal Science Vessel')
                == 'Temporal Science Vessel')
    ok &= check("Unknown word passthrough",
                translate_ship_type('Galaxy') == 'Galaxy')

    print('\n' + ('OK' if ok else 'FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
