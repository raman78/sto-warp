"""Tests for AnnotationWidget hover-tooltip icon resolution.

Regression: after the user corrects and confirms a bbox in Recognition
Review, the canvas hover tooltip must show the icon of the *confirmed*
name (resolved by name, like the review tree) — not the stale ML thumb
from the original match.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage

import warp.trainer.annotation_widget as aw


@pytest.fixture
def widget(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])

    class _StubDataMgr:
        def get_annotations(self, path):
            return []

    w = aw.AnnotationWidget(_StubDataMgr())

    # Capture the thumb argument passed to the shared tooltip composer
    # instead of actually resolving/rendering an <img> tag.
    captured = {}

    def _fake_tooltip_html(thumb, name, info_html, env=None):
        captured["thumb"] = thumb
        captured["name"] = name
        captured["info_html"] = info_html
        captured["env"] = env
        return info_html

    monkeypatch.setattr(aw, "_tooltip_html", _fake_tooltip_html)
    w._captured = captured
    yield w
    w.close()


def _make_thumb() -> QImage:
    img = QImage(4, 4, QImage.Format.Format_RGB32)
    img.fill(0xFF00FF)
    return img


def test_confirmed_item_resolves_icon_by_name(widget):
    """Confirmed rows drop the stale ML thumb → icon resolved from name."""
    thumb = _make_thumb()
    widget.set_review_items([
        {"bbox": (0, 0, 10, 10), "state": "confirmed",
         "name": "Corrected Item", "slot": "fore_weapon", "thumb": thumb},
    ])

    widget._show_hover_tooltip(0)

    assert widget._captured["thumb"] is None
    assert widget._captured["name"] == "Corrected Item"


def test_pending_item_keeps_ml_thumb(widget):
    """Pending rows keep the ML match thumb — what the detector saw."""
    thumb = _make_thumb()
    widget.set_review_items([
        {"bbox": (0, 0, 10, 10), "state": "pending",
         "name": "Guessed Item", "slot": "fore_weapon", "thumb": thumb},
    ])

    widget._show_hover_tooltip(0)

    assert widget._captured["thumb"] is thumb


def test_user_confirmed_item_labelled_by_user(widget):
    """state=confirmed + auto_confirmed=False → 'confirmed by user'."""
    widget.set_review_items([
        {"bbox": (0, 0, 10, 10), "state": "confirmed", "auto_confirmed": False,
         "name": "Item", "slot": "fore_weapon", "conf": 0.9},
    ])

    widget._show_hover_tooltip(0)

    assert "confirmed by user" in widget._captured["info_html"]
    assert "auto-confirmed" not in widget._captured["info_html"]


def test_auto_confirmed_item_labelled_by_detector(widget):
    """state=confirmed + auto_confirmed=True → 'auto-confirmed by detector'."""
    widget.set_review_items([
        {"bbox": (0, 0, 10, 10), "state": "confirmed", "auto_confirmed": True,
         "name": "Item", "slot": "fore_weapon", "conf": 0.9},
    ])

    widget._show_hover_tooltip(0)

    assert "auto-confirmed by detector" in widget._captured["info_html"]
    assert "confirmed by user" not in widget._captured["info_html"]


def test_tooltip_html_wraps_icon_when_present(monkeypatch):
    """_tooltip_html wraps info beside the icon in a 2-col table."""
    import warp.gui as gui
    monkeypatch.setattr(gui, "_tooltip_icon_html",
                        lambda thumb, name, size=48, env=None: "<img/>")
    out = gui._tooltip_html(None, "X", "<b>info</b>")
    assert "<table" in out and "<img/>" in out and "<b>info</b>" in out


def test_tooltip_html_falls_back_without_icon(monkeypatch):
    """No resolvable icon → plain info_html returned unwrapped."""
    import warp.gui as gui
    monkeypatch.setattr(gui, "_tooltip_icon_html",
                        lambda thumb, name, size=48, env=None: "")
    out = gui._tooltip_html(None, "", "<b>info</b>")
    assert out == "<b>info</b>"


def test_env_for_slot_from_trait_slot_name():
    """Trait slot names self-describe their environment."""
    from warp.gui import env_for_slot
    assert env_for_slot("Personal Space Traits") == "space"
    assert env_for_slot("Personal Ground Traits") == "ground"
    assert env_for_slot("Space Reputation") == "space"
    assert env_for_slot("Ground Reputation") == "ground"
    assert env_for_slot("Starship Traits") == "space"


def test_env_for_slot_falls_back_to_build_type():
    """Ambiguous slot → derive env from the build_type prefix."""
    from warp.gui import env_for_slot
    assert env_for_slot("Fore Weapons", "SPACE_EQ") == "space"
    assert env_for_slot("Kit Modules", "GROUND_MIXED") == "ground"
    assert env_for_slot("", "") is None


def test_confirmed_trait_passes_env_to_tooltip(widget):
    """A space trait row resolves its icon with env='space'."""
    widget.set_review_items([
        {"bbox": (0, 0, 10, 10), "state": "confirmed",
         "name": "Adaptive Offense", "slot": "Personal Space Traits"},
    ])

    widget._show_hover_tooltip(0)

    assert widget._captured["env"] == "space"
