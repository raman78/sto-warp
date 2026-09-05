"""sto-warp — STO screenshot recognition (standalone WARP)."""

try:
    from warp._version import __version__
except ImportError:
    # Editable / source checkout without the hatch-vcs build hook having run.
    __version__ = '0.0.0+unknown'

_display_version: str | None = None


def display_version() -> str:
    """The version to *show* a person: what is running, not what was built.

    `__version__` is written into the package once, when it is built. An
    editable install therefore keeps reporting the release it was installed
    from however much the working tree has moved on — a checkout sitting on
    v1.0.36 with local edits was still announcing v1.0.32.dev7, which is
    worse than useless in a bug report.

    So when the code is running from a git checkout, ask git instead:

        v1.0.36                    on the tag, nothing local
        v1.0.36-1-g3a46caf         one commit past it
        v1.0.36-1-g3a46caf-dirty   with uncommitted changes

    The `-dirty` marker is the point — it says out loud that what is running
    is not any released build, which is the state a title is most worth
    reading in.

    An installed copy has no `.git` beside it and falls through to
    `__version__`, so users see a plain release number. The result is cached
    for the life of the process: it is read on every window title, and a
    subprocess per title would be absurd even at two milliseconds.

    Not used for the version reported to the backend (`WARP_VERSION` in
    `warp.knowledge.sync_client`), which stays the built one — that value
    goes into upload payloads and User-Agent headers, and changing its shape
    is a separate decision from what a window says.
    """
    global _display_version
    if _display_version is not None:
        return _display_version

    _display_version = __version__
    try:
        import subprocess
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        if (repo / '.git').exists():
            out = subprocess.run(
                ['git', 'describe', '--tags', '--dirty', '--always'],
                cwd=repo, capture_output=True, text=True, timeout=2,
            )
            described = out.stdout.strip()
            if out.returncode == 0 and described:
                # `git describe` prints the tag as written, `v1.0.36`; the
                # title adds its own V, so drop the tag's.
                _display_version = described.lstrip('v')
    except Exception:
        # Version reporting must never be the thing that stops a window
        # opening. Any failure leaves the built version in place.
        pass
    return _display_version
