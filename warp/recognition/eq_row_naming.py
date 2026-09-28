"""Name the equipment rows nothing else could name, from what is in them.

The row labels (eq_geometry), the panel's shape (eq_stack) and the ship's
profile (`fill_unanchored_rows`) name most rows. A run of rows they leave
unnamed lies between two rows they did name — or runs to the panel's top or
bottom — and `space_eq_rows.ROWS` says which slots can sit there: those
between the two in the game's order, not already used on the panel, and
every mandatory one among them.

Each such arrangement is scored by the items the matcher reads in the rows:
a cell whose item could not sit in the row the arrangement gives it counts
against it ("could sit" straight from the cargo groups — a Universal console
is in every console group). The outcome per row is one of:

  'name'    the best arrangement is the only one, and leaves at most
            MAX_MISFIT of the read cells out of place
  'guess'   an arrangement fits, but not certainly — the user is asked to
            confirm or change it
  'unknown' no arrangement exists, or the row's own items cannot sit in the
            slot the best arrangement gives it
  'outside' a trailing row past the last named one that the best
            arrangement leaves out, as lying beyond the panel's bottom —
            it gets no boxes (its items count against leaving it out)

Where several arrangements tie, the one with fewer optional rows is taken as
the guess: most ships lack most optional rows.
"""
from __future__ import annotations

from itertools import combinations

from warp.recognition import space_eq_rows as _ROWS

MAX_MISFIT = 0.10  # share of read cells an arrangement may leave out of place


def arrangements(prev_slot: str | None, next_slot: str | None, k: int,
                 used: set[str]) -> list[list[str]]:
    """Every sequence of k slots the table allows between two named rows.

    Every mandatory row between the run's named neighbours has to be in it.
    With no named row above the run, the panel may be cut or covered at the
    top — a tooltip over the Deflector row once hid Fore Weapons above the
    visible stack, and requiring Fore at the top shifted every row by one —
    so there only the mandatory rows from the arrangement's own first row on
    are required. The bottom keeps the full rule: rows past the panel's
    bottom are handled by leaving trailing rows out (name_run, open_end),
    and relaxing it as well only made console rows ambiguous.
    """
    lo = _ROWS.INDEX[prev_slot] + 1 if prev_slot in _ROWS.INDEX else 0
    hi = _ROWS.INDEX[next_slot] if next_slot in _ROWS.INDEX else len(_ROWS.ROWS)
    cand = [r for r in _ROWS.ROWS[lo:hi] if r.slot not in used]
    out = []
    for combo in combinations(cand, k):
        slots = [r.slot for r in combo]
        first = lo if prev_slot in _ROWS.INDEX else (_ROWS.INDEX[slots[0]] if slots else lo)
        last = hi
        required = {r.slot for r in cand
                    if not r.optional and first <= _ROWS.INDEX[r.slot] < last}
        if required <= set(slots):
            out.append(slots)
    return out


def _misfits(items: list[list[str]], seq: list[str], allowed: dict[str, set]) -> list[int]:
    return [sum(1 for n in names if n not in allowed.get(slot, set()))
            for names, slot in zip(items, seq)]


def name_run(items: list[list[str]], prev_slot: str | None, next_slot: str | None,
             used: set[str], allowed: dict[str, set],
             open_end: bool = False) -> list[tuple[str, str | None]]:
    """(outcome, slot) for each row of an unnamed run, top to bottom.

    *items* are the item names read in each row's filled cells. *allowed*
    maps a slot to the item names that can sit in it (cargo groups). With
    *open_end* the run is the last one on the panel and its trailing rows
    may lie past the panel's bottom: arrangements that leave them out are
    scored too, and rows left out come back 'outside'.
    """
    k = len(items)
    options = []   # (cost, items left out, optional rows, -kept, seq)
    # With an open end, trailing rows may be left out as lying past the
    # panel's bottom — a label-less stack can run on into the specialisation
    # icons under it. A row left out explains none of its items, so each of
    # them counts as out of place: a real last row is not worth dropping,
    # a row of another panel's icons is.
    for kept in range(k, (0 if open_end else k) - 1, -1):
        dropped = sum(len(x) for x in items[kept:])
        for seq in arrangements(prev_slot, next_slot, kept, used):
            m = sum(_misfits(items[:kept], seq, allowed)) + dropped
            n_opt = sum(1 for s in seq if s in _ROWS.OPTIONAL)
            # At equal cost, the arrangement accounting for more of what was
            # read wins (a row with items in it is evidence of a row); then
            # the one with fewer optional rows.
            options.append((m, dropped, n_opt, -kept, seq))
    if not options:
        return [('unknown', None)] * k
    options.sort(key=lambda o: (o[0], o[1], o[2], o[3]))
    best_m = options[0][0]
    tied = [o for o in options if o[0] == best_m]
    pick = tied[0]
    kept, seq = -pick[3], pick[4]
    row_misfit = _misfits(items[:kept], seq, allowed)
    # Certainty is about the rows kept: how many of the items read in them
    # the arrangement leaves out of place. Items in rows left out were
    # already weighed in choosing the arrangement; they say nothing about
    # whether the kept rows are named right.
    read = sum(len(x) for x in items[:kept])
    fits = read > 0 and sum(row_misfit) <= max(1, MAX_MISFIT * read)
    certain = (len(tied) == 1 and fits) or (read == 0 and len(options) == 1)
    out: list[tuple[str, str | None]] = []
    for i in range(k):
        if i >= kept:
            out.append(('outside', None))
            continue
        slot = seq[i]
        agreed = all(-o[3] > i and o[4][i] == slot for o in tied)
        if items[i] and row_misfit[i] * 2 > len(items[i]):
            out.append(('unknown', None))            # most of its items cannot sit there
        elif certain or (agreed and fits):
            out.append(('name', slot))
        else:
            out.append(('guess', slot))
    return out
