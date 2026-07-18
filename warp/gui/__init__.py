"""sto-warp standalone GUI.

`warp.gui.warp_window.WarpWindow` is the user-facing replacement for the
SETS-coupled `warp/warp_dialog.py`: open screenshots, run the recognition
pipeline, view the per-slot results, export JSON.
"""

from __future__ import annotations

from PySide6.QtCore import Qt


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


def _tooltip_html(thumb, name: str, info_html: str,
                  env: str | None = None) -> str:
    """Compose a hover tooltip: resolved icon (left) beside *info_html* (right).

    Shared by the Recognition Review tree and the annotation canvas so both
    lay the icon out identically.  When no icon resolves (empty *name* and no
    *thumb*) the plain *info_html* is returned unwrapped. *env* disambiguates
    same-named space/ground traits when the icon is resolved from *name*.
    """
    icon_html = _tooltip_icon_html(thumb, name, env=env)
    if not icon_html:
        return info_html
    return (f'<table cellspacing="0" cellpadding="0"><tr>'
            f'<td style="vertical-align:middle;padding-right:6px">{icon_html}</td>'
            f'<td style="vertical-align:middle">{info_html}</td>'
            f'</tr></table>')
