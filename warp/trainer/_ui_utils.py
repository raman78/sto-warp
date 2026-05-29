# warp/trainer/_ui_utils.py
# WARP CORE UI helpers — small Qt utility widgets, dot-icons, and the
# per-image match-summary table. Extracted from trainer_window.py during
# the Phase-0 refactor so workers + window mixins can share them without
# pulling in the giant WarpCoreWindow module.

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QPalette
from PySide6.QtWidgets import QStyledItemDelegate

from warp import userdata


class _ColorPreservingDelegate(QStyledItemDelegate):
    """Keep item's ForegroundRole color visible even when the row is selected."""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        if isinstance(brush, QBrush) and brush.color().isValid():
            option.palette.setColor(QPalette.ColorRole.HighlightedText, brush.color())


# ── Match summary table + history ──────────────────────────────────────
_RECOG_HISTORY_PATH = userdata.training_data_dir() / 'recog_history.json'
_DELTA_EPS = 0.03  # minimum absolute conf change to render an arrow


def _arrow(prev: float | None, curr: float) -> str:
    if prev is None:
        return ' new'
    d = curr - prev
    if abs(d) < _DELTA_EPS:
        return '  ─ '
    return f'{"↑" if d > 0 else "↓"}{abs(d):.2f}'


def _fmt_score(v: float) -> str:
    return f'{v:.2f}' if v > 0 else ' -  '


def _load_recog_history() -> dict:
    try:
        if _RECOG_HISTORY_PATH.exists():
            import json
            return json.loads(_RECOG_HISTORY_PATH.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}


def _save_recog_history(hist: dict) -> None:
    try:
        import json
        _RECOG_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RECOG_HISTORY_PATH.write_text(
            json.dumps(hist, indent=2), encoding='utf-8')
    except Exception:
        pass


def _log_match_summary(image_name: str, match_log: list[dict]) -> None:
    """
    Render the per-image match summary table and update recog_history.json.

    Columns: slot | name | win | embed | sess | tmpl | knldg | Δ (vs previous
    run for the same image+slot). Totals per source at the bottom.
    """
    from warp.debug import log as _slog
    if not match_log:
        return

    hist  = _load_recog_history()
    prev  = hist.get(image_name, {})
    curr: dict[str, dict] = {}
    totals: dict[str, int] = {}

    header = (f'{"slot":<28} {"name":<32} {"win":<6} '
              f'{"embed":>6} {"sess":>6} {"tmpl":>6} {"knldg":>6}   Δ')
    rows: list[str] = [header, '-' * len(header)]

    # When the same slot appears multiple times (e.g. Boff Tactical row),
    # disambiguate by appending index. recog_history is keyed by these labels.
    slot_seen: dict[str, int] = {}
    for entry in match_log:
        slot = entry.get('slot', '')
        idx  = slot_seen.get(slot, 0)
        slot_seen[slot] = idx + 1
        key  = slot if idx == 0 else f'{slot}#{idx}'

        name   = entry.get('name', '') or ''
        src    = entry.get('src',  '') or ''
        stages = entry.get('stages', {}) or {}
        e = float(stages.get('embed',     0.0))
        f = float(stages.get('soft',      0.0))
        s = float(stages.get('session',   0.0))
        t = float(stages.get('template',  0.0))
        k = float(stages.get('knowledge', 0.0))

        prev_conf = prev.get(key, {}).get('conf')
        arrow = _arrow(prev_conf, float(entry.get('conf', 0.0)))
        # Track the active ML score (embed or soft, whichever the matcher used).
        ml_score = e if e > 0 else f
        rows.append(
            f'{key[:28]:<28} {name[:32]:<32} {src[:6]:<6} '
            f'{_fmt_score(ml_score):>6} {_fmt_score(s):>6} '
            f'{_fmt_score(t):>6} {_fmt_score(k):>6}   {arrow}'
        )
        totals[src] = totals.get(src, 0) + 1
        curr[key] = {'name': name, 'conf': float(entry.get('conf', 0.0)),
                     'src':  src}

    totals_str = '  '.join(f'{src or "?"}={cnt}'
                            for src, cnt in sorted(totals.items()))
    _slog.info(f'WARP CORE: match summary  {image_name}  ({len(match_log)} items)')
    for r in rows:
        _slog.info(f'  {r}')
    _slog.info(f'  TOTAL: {totals_str}')

    hist[image_name] = curr
    _save_recog_history(hist)


# ── Dot icons (green = user confirmed, yellow = ML auto) ───────────────
def _make_dot_icon(color: str) -> 'QIcon':
    """Small 14×14 filled circle icon for green/yellow confirmation state."""
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtGui import QPixmap, QPainter, QColor, QIcon
    pix = QPixmap(14, 14)
    pix.fill(_Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(_Qt.PenStyle.NoPen)
    p.drawEllipse(1, 1, 12, 12)
    p.end()
    return QIcon(pix)


_ICON_USER_CONFIRMED: 'QIcon | None' = None   # green — created lazily
_ICON_ML_AUTO:        'QIcon | None' = None   # yellow — created lazily


def _get_user_icon() -> 'QIcon':
    global _ICON_USER_CONFIRMED
    if _ICON_USER_CONFIRMED is None:
        _ICON_USER_CONFIRMED = _make_dot_icon('#44dd66')
    return _ICON_USER_CONFIRMED


def _get_ml_icon() -> 'QIcon':
    global _ICON_ML_AUTO
    if _ICON_ML_AUTO is None:
        _ICON_ML_AUTO = _make_dot_icon('#ffcc00')
    return _ICON_ML_AUTO
