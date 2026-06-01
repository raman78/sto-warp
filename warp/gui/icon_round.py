"""Rounded-corner ('squircle') wrapper for the bundled app icon.

macOS Big Sur+ and modern Linux desktops (Plasma 6, GNOME 46) lean
toward icons with soft rounded corners rather than a hard square.
The bundled `warp/resources/SETS_icon_small.png` is a 593×593 RGBA
bitmap with an opaque black background; clipping its corners with a
rounded-rectangle alpha mask produces a clean, consistent shape on
every platform.

Callers obtain a Pillow image and hand it to `Image.save(...)` with
the platform-specific format (`ICNS` on macOS, `ICO` on Windows,
`PNG` on Linux). No on-disk caching here — each call site already
owns its own cache directory.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


# ~22.5% radius — close to Apple's Big-Sur 'squircle' template.
SQUIRCLE_RADIUS_FRAC = 0.225


def rounded_icon(
    src: Path,
    size: int = 1024,
    radius_frac: float = SQUIRCLE_RADIUS_FRAC,
) -> Image.Image:
    """Resize the source icon to `size×size` and clip its corners.

    The rounded-rectangle mask is rasterised at 4× resolution and
    downscaled with Lanczos so the corner edges stay antialiased.
    Any pre-existing transparency in the source is preserved — the
    mask is multiplied into the existing alpha channel rather than
    replacing it.
    """
    base = Image.open(src).convert('RGBA').resize((size, size), Image.LANCZOS)

    ss = 4
    big = (size * ss, size * ss)
    mask_big = Image.new('L', big, 0)
    radius = int(size * ss * radius_frac)
    ImageDraw.Draw(mask_big).rounded_rectangle(
        (0, 0, big[0] - 1, big[1] - 1), radius=radius, fill=255,
    )
    mask = mask_big.resize((size, size), Image.LANCZOS)

    _, _, _, alpha = base.split()
    base.putalpha(ImageChops.multiply(alpha, mask))
    return base
