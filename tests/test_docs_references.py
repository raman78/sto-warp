"""Documentation must cite constructs by name, never by line number.

Line numbers rot on every edit. Measured 2026-08-31, before the docs were
converted, 27% of the line references under `docs/` pointed at the wrong
construct — some of them only weeks old.

The two failure modes are not equivalent, which is the real argument. A stale
*name* fails loudly: grep returns nothing and the reader knows the doc is
behind. A stale *line number* fails silently — it lands on plausible-looking
code, and the reader may build on it without ever noticing.

For a block with no name of its own, quote its section comment (`Anchor 1c`);
that is just as greppable and does not move. If a block cannot be referred to
by name at all, the code should be given one.

Run standalone:
    python -m pytest tests/test_docs_references.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / 'docs'

# `warp_importer.py:927`, `warp/recognition/eq_geometry.py:526`, or a bare
# `:155` pointing into whichever module the doc named in its header.
LINE_REF = re.compile(r'`(?:[\w/]*\.py):\d+`|`:\d+`')


def _docs() -> list[Path]:
    return sorted(DOCS.glob('*.md'))


def test_docs_directory_is_present():
    """Guards the parametrised test below from silently covering nothing."""
    assert _docs(), f'no documentation found under {DOCS}'


@pytest.mark.parametrize('doc', _docs(), ids=lambda p: p.name)
def test_doc_cites_names_not_line_numbers(doc):
    offenders = [
        f'{doc.name}:{i}  {m.group(0)}'
        for i, line in enumerate(doc.read_text(encoding='utf-8').splitlines(), 1)
        for m in LINE_REF.finditer(line)
    ]

    assert not offenders, (
        'line-number references found; cite the construct by name instead:\n  '
        + '\n  '.join(offenders)
    )


# A companion check — "every file path a doc mentions must exist" — was built
# and then dropped. It fired on eight documents, and almost every hit was
# legitimate: runtime artefacts under `warp/models/` that are never committed,
# files a roadmap describes before they are written, and deliberate historical
# references (`CARGO_DATA_PLAN.md` documents a scraper deleted in August 2026
# and says so in the text). Keeping it would have meant an allowlist of
# exceptions, and that allowlist would rot exactly the way line numbers do.
#
# It did find one real defect on its single run — two `tools/*.py` paths that
# should have been `warp/tools/*.py` — so it is worth re-running by hand after
# a large rename. It is not worth carrying as a test.
