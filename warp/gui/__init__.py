"""sto-warp standalone GUI.

`warp.gui.warp_window.WarpWindow` is the user-facing replacement for the
SETS-coupled `warp/warp_dialog.py`: open screenshots, run the recognition
pipeline, view the per-slot results, export JSON.
"""

from __future__ import annotations

from PySide6.QtCore import Qt


# ── Colours for the two states that are not an item ───────────────────────
#
# `__empty__` and `__inactive__` mean different things — an open slot the
# player has not filled, against one the ship or the character has not
# unlocked yet — and both used to be drawn in the same grey, so the review
# list could not tell them apart at a glance.
#
# Muted on purpose. The vivid colours are taken and each already means
# something: blue is auto-confirmed, mint is user-confirmed, orange is a
# community conflict, gold a slot-type mismatch, red unmatched. These two are
# states of *nothing being there*, so they sit quieter than any of them while
# still being two clearly different hues — lilac against sage.
#
# Defined here, where both the review list and the canvas tooltip can reach
# them, so the two views cannot drift apart on what an empty slot looks like.
VIRTUAL_COLOURS: dict[str, dict[str, str]] = {
    '__empty__':    {'confirmed': '#b3a6dd', 'pending': '#d0c8ec'},
    '__inactive__': {'confirmed': '#8fb8b0', 'pending': '#b6d4ce'},
}

# What the user reads instead of the internal marker.
VIRTUAL_LABELS: dict[str, str] = {
    '__empty__':    '[empty slot]',
    '__inactive__': '[inactive slot]',
}


def virtual_colour(name: str, confirmed: bool = True) -> str | None:
    """Colour for `__empty__` / `__inactive__`, or None for a real item."""
    entry = VIRTUAL_COLOURS.get(name)
    if entry is None:
        return None
    return entry['confirmed' if confirmed else 'pending']


def env_for_slot(slot: str, build_type: str = '') -> str | None:
    """Best-effort 'space' / 'ground' for a review item, for env-aware icons.

    Only a few traits collide on display name across environments with
    *different* icons, so the trait slot name (which self-describes its env,
    e.g. 'Personal Space Traits' vs 'Personal Ground Traits') is the primary
    signal; *build_type* is the fallback for slots that don't self-describe.
    Returns ``None`` when the environment can't be determined.
    """
    s = slot or ''
    if 'Ground' in s:
        return 'ground'
    if 'Space' in s or s == 'Starship Traits':
        return 'space'
    bt = (build_type or '').upper()
    if bt.startswith('GROUND'):
        return 'ground'
    if bt.startswith('SPACE'):
        return 'space'
    return None


def _tooltip_icon_html(thumb, name: str, size: int = 48,
                       env: str | None = None) -> str:
    """Return an ``<img>`` tag with a base64-encoded icon, or ``''``.

    *thumb* is a QImage (from the icon matcher).  If ``None``, the local
    reference-icon PNG is loaded from the cargo icons directory instead.
    *env* ('space'/'ground') disambiguates traits that share a display name
    across environments but have different icons.
    """
    import base64
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtGui import QImage

    img: QImage | None = None
    if isinstance(thumb, QImage) and not thumb.isNull():
        img = thumb
    elif name:
        try:
            from warp.data.cargo import ref_icon_path
            p = ref_icon_path(name, env)
            if p:
                img = QImage(str(p))
                if img.isNull():
                    img = None
        except Exception:
            pass
    if img is None:
        return ''
    if img.width() > size or img.height() > size:
        img = img.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, 'PNG')
    b64 = base64.b64encode(buf.data().data()).decode('ascii')
    return f'<img src="data:image/png;base64,{b64}" width="{img.width()}" height="{img.height()}"/>'


def _variant_note(name: str, variant: str) -> str:
    """'art: 23c' for the 34 items the wiki draws twice, else ''.

    Cargo has one row for such an item and the wiki has two pictures, so the
    variant is not a different item and must never reach the name — that goes
    to the build writer and on to SETS, which knows only the cargo name.
    It is worth saying out loud all the same: a player who wants the exact
    weapon in the screenshot needs to know they are looking for the
    23rd-century version of it.

    *variant* is the icon filename the match came from, e.g.
    'Phaser Dual Heavy Cannons (23c)'. Only the part that distinguishes it
    from *name* is shown; the rest is the name again.
    """
    if not variant or not name or variant == name:
        return ''
    tag = variant[len(name):].strip() if variant.startswith(name) else variant
    tag = tag.strip('()') or variant
    return f'<span style="color:#888">art: {tag}</span>'


def slot_tooltip_html(slot: str, name: str, conf: float, *,
                      confirmed: bool = False, auto_confirmed: bool = False,
                      orig_name: str = '', thumb=None, variant: str = '',
                      env: str | None = None) -> str:
    """Compose the canvas tooltip card for one recognised slot.

    The single composer for both canvases — WARP's results view and WARP CORE's
    annotation widget — so the two cannot show the same item differently. (The
    two review *trees* still build their own variants; they carry extra
    per-view context.)

    *confirmed* rows drop *thumb*: it is the detector's original match and goes
    stale the moment the user corrects the name, so the icon is resolved from
    the confirmed name instead. Unconfirmed rows keep it — that is what the
    detector actually saw.
    """
    from warp.recognition.boff_keys import pretty_slot

    slot_disp = pretty_slot(slot or '?')
    colour = ('#7effc8' if conf >= 0.85 else
              '#e8c060' if conf >= 0.70 else '#ff9966')

    # The item's name is what the card is about, so it carries the emphasis.
    # The slot is context — which row this is — and had the bold until
    # 2026-09-06, which put the weight on the one line the reader already
    # knows from where they are hovering.
    #
    # `__empty__` / `__inactive__` are shown as the words a user reads, in the
    # same two colours the review list uses, so a glance at either view says
    # the same thing.
    v_colour = virtual_colour(name, confirmed=True)
    if v_colour:
        name_disp = (f'<span style="color:{v_colour}">'
                     f'<b>{VIRTUAL_LABELS[name]}</b></span>')
    else:
        name_disp = f'<b>{name}</b>' if name else '— unmatched —'

    if confirmed:
        status = ('auto-confirmed by detector' if auto_confirmed
                  else 'confirmed by user')
        lines = [slot_disp, name_disp, f'<i>{status}</i>']
        if conf > 0.0:
            ml_text = orig_name if orig_name and orig_name != name else name
            ml_text = VIRTUAL_LABELS.get(ml_text, ml_text) or '— unmatched —'
            lines.append(f'ML: <span style="color:{colour}">{ml_text} ({conf:.1%})</span>')
        else:
            lines.append('<span style="color:#888">ML: unknown (previous session)</span>')
        info_html = '<br>'.join(lines)
    else:
        info_html = (f'{slot_disp}<br>{name_disp}'
                     f'<br>Confidence: <span style="color:{colour}">{conf:.1%}</span>')

    note = _variant_note(name, variant)
    if note:
        info_html += f'<br>{note}'

    return _tooltip_html(None if confirmed else thumb, name, info_html, env=env)


def _tooltip_html(thumb, name: str, info_html: str,
                  env: str | None = None) -> str:
    """Compose a hover tooltip: resolved icon (left) beside *info_html* (right).

    Shared by the Recognition Review tree and the annotation canvas so both
    lay the icon out identically.  When no icon resolves (empty *name* and no
    *thumb*) the plain *info_html* is returned unwrapped. *env* disambiguates
    same-named space/ground traits when the icon is resolved from *name*.
    """
    # `white-space:nowrap` so the card grows sideways instead of folding a
    # long item name over three lines. Qt sizes a rich-text tooltip to its
    # content and clamps it to the screen, so the only thing this gives up is
    # a narrow card for the longest names — and those are exactly the ones
    # that were unreadable wrapped. `Console - Advanced Engineering -
    # Isomagnetic Plasma Distribution Manifold` is 71 characters.
    nowrap = 'white-space:nowrap;vertical-align:middle'
    icon_html = _tooltip_icon_html(thumb, name, env=env)
    if not icon_html:
        return f'<div style="white-space:nowrap">{info_html}</div>'
    return (f'<table cellspacing="0" cellpadding="0"><tr>'
            f'<td style="vertical-align:middle;padding-right:6px">{icon_html}</td>'
            f'<td style="{nowrap}">{info_html}</td>'
            f'</tr></table>')
