"""sto-warp — STO screenshot recognition (standalone WARP)."""

try:
    from warp._version import __version__
except ImportError:
    # Editable / source checkout without the hatch-vcs build hook having run.
    __version__ = '0.0.0+unknown'
