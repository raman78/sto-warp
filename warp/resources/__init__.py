"""Static assets shipped with the sto-warp wheel.

Use `resource_path(name)` to obtain an on-disk `Path` to an icon or any
other resource. Works equally for `pip install -e .` checkouts and for
proper wheel installs.

Note: `importlib.resources.files(...)` returns a `Traversable`. For files
that live directly inside a regular installed package (the common case
here), this is already a real on-disk path. We surface it as `Path` for
callers — Qt's `QIcon(str)` constructor accepts a plain string path.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path


def resource_path(name: str) -> Path:
    """Return the absolute path of the resource file shipped with the package."""
    return Path(resources.files(__name__).joinpath(name))
