"""Schema warnings reach the user, and only reach GitHub on request.

Covers the WARP window's half of the export hook: violations produce the
warning dialog, and the pre-filled issue opens only when the user picks
'Report on GitHub'. Nothing is submitted by this code — the browser shows
a form the user still has to send.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from warp import sets_schema
from warp.gui import warp_window as ww


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    win = ww.WarpWindow()
    yield win
    win.close()


@pytest.fixture
def opened_urls(monkeypatch) -> list[str]:
    urls: list[str] = []

    class _FakeDesktopServices:
        @staticmethod
        def openUrl(url):
            urls.append(url.toString())
            return True

    monkeypatch.setattr(ww, 'QDesktopServices', _FakeDesktopServices)
    return urls


def _dialog_answering(button_text: str | None):
    """QMessageBox subclass that skips the modal loop and picks a button."""

    class _Box(QMessageBox):
        def exec(self):
            return 0

        def clickedButton(self):
            if button_text is None:
                return None
            return next(b for b in self.buttons() if b.text().startswith(button_text))

    return _Box


@pytest.fixture
def violations() -> list[sets_schema.Violation]:
    return [sets_schema.Violation('/space/boff_specs[0][0]', 'seat_profession',
                                  'non-empty str', 'None')]


class _Report:
    ship = 'Avenger Battlecruiser'


def test_report_button_opens_a_prefilled_issue(window, monkeypatch, opened_urls, violations):
    monkeypatch.setattr(ww, 'QMessageBox', _dialog_answering('Report'))

    window._offer_schema_issue(violations, _Report())

    assert len(opened_urls) == 1
    assert opened_urls[0].startswith(sets_schema.ISSUE_BASE_URL)
    assert 'labels=sets-schema' in opened_urls[0]


def test_closing_the_dialog_sends_nothing(window, monkeypatch, opened_urls, violations):
    monkeypatch.setattr(ww, 'QMessageBox', _dialog_answering(None))

    window._offer_schema_issue(violations, _Report())

    assert opened_urls == []


def test_an_upstream_gap_is_explained_not_reported(window, monkeypatch, opened_urls):
    """Items missing from SETS' own tables are nobody's bug here.

    WARP resolves them from its harvested list; SETS reads the cargo
    tables that omit them and drops them. There is nothing to report, so
    the dialog explains it and offers no report button.
    """
    monkeypatch.setattr(ww, 'QMessageBox', _dialog_answering(None))
    upstream = [sets_schema.Violation(
        '/ground/weapons[0]', 'not_in_sets', 'an item SETS can resolve',
        "'Elite Fleet Colony Security Antiproton Blast Assault' (listing-scrape)")]

    window._offer_schema_issue(upstream, _Report())

    assert opened_urls == []


def test_a_real_defect_alongside_one_still_offers_the_report(window, monkeypatch,
                                                            opened_urls):
    monkeypatch.setattr(ww, 'QMessageBox', _dialog_answering('Report'))
    mixed = [
        sets_schema.Violation('/ground/weapons[0]', 'not_in_sets',
                              'an item SETS can resolve', "'X' (listing-scrape)"),
        sets_schema.Violation('/space/boff_specs[0][0]', 'seat_profession',
                              'non-empty str', 'None'),
    ]

    window._offer_schema_issue(mixed, _Report())

    assert len(opened_urls) == 1
    # Only the defect goes in the report; the upstream gap is not a bug.
    assert 'seat_profession' in opened_urls[0]
    assert 'listing-scrape' not in opened_urls[0]
