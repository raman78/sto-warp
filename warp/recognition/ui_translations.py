"""
UI translation table loader.

Reads warp/data/ui_translations.csv (admin-editable, see header comment in
the CSV for the schema) and exposes lookup helpers used by the recognition
pipeline. Lookup is case-insensitive; non-alphanumeric characters are kept
as-is (so spaces in multi-word labels still match).

This module is the single integration point for localized OCR strings. To
add a language, append rows to the CSV — no Python edits needed.

API:
    OCR_LANGUAGES        — list of EasyOCR language codes to load (e.g.
                           ['en', 'de']). Derived from CSV contents.
    normalize_map(category)
                         — dict {translation_lower: canonical_en} for a
                           single category (e.g. 'space_slot'). Includes
                           the canonical_en→canonical_en identity entries
                           so an English-only call site also works.
    synonyms(category, canonical_en)
                         — set of all known translations for one canonical
                           entry (lowercase, includes the canonical itself).
    augment_substring_phrases(category, english_phrases)
                         — given a tuple of lowercase English phrases used
                           for substring matching (e.g. _TRAIT_SPACE_HEADERS),
                           return a tuple expanded with every known synonym.

The CSV is loaded once on first access and cached at module scope.
"""
from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock

# Always-on baseline. Other languages are appended as they appear in the CSV.
_BASE_LANGUAGES: tuple[str, ...] = ('en',)

_CSV_PATH = Path(__file__).parent.parent / 'data' / 'ui_translations.csv'

# {category: {translation_lower: canonical_en}}
_TABLE: dict[str, dict[str, str]] | None = None
# {category: {canonical_en: set[translation_lower]}}
_BY_CANONICAL: dict[str, dict[str, set[str]]] | None = None
_LANGUAGES: tuple[str, ...] = _BASE_LANGUAGES
_LOAD_LOCK = Lock()


def _load() -> None:
    """Parse the CSV into the module-level caches. Idempotent."""
    global _TABLE, _BY_CANONICAL, _LANGUAGES
    with _LOAD_LOCK:
        if _TABLE is not None:
            return
        table: dict[str, dict[str, str]] = {}
        by_canon: dict[str, dict[str, set[str]]] = {}
        langs: set[str] = set(_BASE_LANGUAGES)
        if _CSV_PATH.exists():
            with _CSV_PATH.open(encoding='utf-8', newline='') as f:
                # Strip comment lines before handing off to csv.reader so
                # the header row is whatever first non-comment line is.
                rows = [ln for ln in f if ln.strip() and not ln.lstrip().startswith('#')]
            reader = csv.DictReader(rows)
            for row in reader:
                cat = (row.get('category') or '').strip()
                canon = (row.get('canonical_en') or '').strip()
                lang = (row.get('language') or '').strip().lower()
                tran = (row.get('translation') or '').strip()
                if not (cat and canon and lang and tran):
                    continue
                langs.add(lang)
                key = tran.lower()
                table.setdefault(cat, {})[key] = canon
                by_canon.setdefault(cat, {}).setdefault(canon, set()).add(key)
                # English passthrough — canonical also matches itself.
                en_key = canon.lower()
                table[cat].setdefault(en_key, canon)
                by_canon[cat][canon].add(en_key)
        _TABLE = table
        _BY_CANONICAL = by_canon
        # Stable order: English first, then other languages alphabetically.
        non_en = sorted(l for l in langs if l != 'en')
        _LANGUAGES = ('en', *non_en)


def normalize_map(category: str) -> dict[str, str]:
    """Return {translation_lower: canonical_en} for the given category."""
    _load()
    return dict(_TABLE.get(category, {}))


def synonyms(category: str, canonical_en: str) -> set[str]:
    """Return all lowercase synonyms (incl. canonical itself) for one entry.

    Lookup on canonical_en is case-insensitive — the CSV may use Title Case
    (e.g. 'Fore Weapons') while call sites may pass lowercase substring-match
    forms (e.g. 'fore weapons').
    """
    _load()
    cat = _BY_CANONICAL.get(category, {})
    if canonical_en in cat:
        return set(cat[canonical_en])
    target = canonical_en.lower()
    for key, vals in cat.items():
        if key.lower() == target:
            return set(vals)
    return set()


def augment_substring_phrases(category: str, english_phrases) -> tuple[str, ...]:
    """Expand a tuple of English substring-match phrases with all synonyms.

    Used by screen-type detection (text_extractor) where matching is
    substring-on-lowercase-joined-OCR. Each English phrase keeps its
    place; its synonyms are appended after it. De-duplicated, lowercase.
    """
    _load()
    out: list[str] = []
    seen: set[str] = set()
    for en in english_phrases:
        for s in (en, *sorted(synonyms(category, en))):
            sl = s.lower()
            if sl not in seen:
                seen.add(sl)
                out.append(sl)
    return tuple(out)


def ocr_languages() -> list[str]:
    """EasyOCR language list derived from CSV contents (always includes 'en')."""
    _load()
    return list(_LANGUAGES)
