"""How much this install may still ask of the WARP backend today.

One budget, shared by everything that POSTs: the trainer's crop / screen-type
/ anchor uploads (`warp.trainer.sync`) and the knowledge client's pHash
contributions (`warp.knowledge.sync_client`). They talk to the same five
rate-limited endpoints and are counted together on the server, so counting
them apart on the client cannot work.

**The unit is a request.** `sets-warp-backend/main.py` admits
`MAX_REQ_PER_INSTALL` requests per UTC day (500 by default) and the same again
per IP; it does not care how many items a request carried, nor whether it
accepted them. Every counter this replaces measured something else — the
trainer counted crops it queued, the knowledge client counted contributions
the server accepted — so a day of refusals moved neither, and both kept
sending.

Measured on the maintainer's install, 2026-09-06: 127 corrected screen types
had been stuck "not yet shared" for days. Every POST was answered
`HTTP 429 Rate limit exceeded. Try again tomorrow.`, and the client's reaction
to that was to send more: the trainer walked all thirteen screen-type
directories, one refused POST each, then crops, then anchors; the knowledge
client retried each queued contribution three times, waited five minutes, and
retried again — on its own about 864 requests a day, all refused. Clearing the
backlog was what kept the door shut.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from warp import userdata
from warp.debug import syslog as log

# Kept under the server's 500 so this client stops on its own terms rather
# than by being refused, and because the server counts requests we may not
# have recorded — a retry inside urllib, or one that timed out after arriving.
MAX_DAILY_REQUESTS = 480


class BackendBudgetExhausted(Exception):
    """Nothing more may be sent today.

    An exception rather than a return value because it is not one channel's
    problem: the budget is shared, so once it is gone the whole upload run is
    over and every further POST would be a request spent on being told no.
    """


def _today() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d')


class DailyBudget:
    """The shared counter, and the record of a refusal we were actually given.

    Two independent stops, and both are needed:

    - **Our own request count**, against `MAX_DAILY_REQUESTS`. This is a
      *prediction* of the server's per-install bucket, and it is what keeps an
      ordinary day from ever reaching a refusal.
    - **A 429 we received.** The prediction cannot stand alone: the server
      also keeps a bucket per IP, which this install shares with anyone behind
      the same address and therefore cannot see; the buckets live in the
      server process, so a restart clears them; and the cap is an environment
      variable that can change under us. A refusal is ground truth, so it is
      honoured until the UTC day turns.

    State is a small JSON file in the config dir. Nothing here raises: a
    missing or corrupt file reads as "nothing spent today", which is the same
    position a new install is in, and a failed write costs at most a
    double-counted day.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else _default_path()
        self._day = _today()
        self._spent = 0
        self._refused_on = ''
        try:
            raw = json.loads(self._path.read_text())
            if raw.get('day') == self._day:
                self._spent = int(raw.get('requests', 0))
                self._refused_on = str(raw.get('refused_on', ''))
        except Exception:
            pass

    @property
    def spent(self) -> int:
        return self._spent

    def refused_today(self) -> bool:
        return self._refused_on == self._day

    def remaining(self) -> int:
        return max(0, MAX_DAILY_REQUESTS - self._spent)

    def blocked_reason(self) -> str:
        """Why nothing may be sent, or `''` when the door is open."""
        if self.refused_today():
            return ('the backend refused this install earlier today and asked '
                    'us to try again tomorrow')
        if self._spent >= MAX_DAILY_REQUESTS:
            return (f'{self._spent} requests already sent today, this client\'s '
                    f'daily budget of {MAX_DAILY_REQUESTS}')
        return ''

    def check(self) -> None:
        """Raise `BackendBudgetExhausted` when nothing may be sent."""
        reason = self.blocked_reason()
        if reason:
            raise BackendBudgetExhausted(reason)

    def note_request(self) -> None:
        self._spent += 1
        self._save()

    def note_refusal(self, detail: str = '') -> None:
        """Record a 429. Logged loudly and once — a refusal means the community
        dataset is not getting something this install confirmed, and the only
        other sign of it is a number in the trainer that stops moving."""
        if not self.refused_today():
            log.warning(
                f'WARP backend: refused until tomorrow after '
                f'{self._spent} request(s) today'
                + (f' — {detail}' if detail else '')
                + '. Nothing further will be sent until midnight UTC; the '
                  'confirmations stay on disk.')
        self._refused_on = self._day
        self._save()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({
                'day': self._day,
                'requests': self._spent,
                'refused_on': self._refused_on,
            }))
        except Exception:
            pass


def _default_path() -> Path:
    try:
        return userdata.backend_budget_file()
    except Exception:
        return Path.home() / '.config' / 'warp' / 'backend_budget.json'
