"""sto-warp standalone GUI.

`warp.gui.warp_window.WarpWindow` is the user-facing replacement for the
SETS-coupled `warp/warp_dialog.py`: open screenshots, run the recognition
pipeline, view the per-slot results, export JSON.
"""

from __future__ import annotations

from PySide6.QtCore import Qt


def _tooltip_icon_html(thumb, name: str, size: int = 48) -> str:
    """Return an ``<img>`` tag with a base64-encoded icon, or ``''``.

    *thumb* is a QImage (from the icon matcher).  If ``None``, the local
    reference-icon PNG is loaded from the cargo icons directory instead.
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
            p = ref_icon_path(name)
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
