"""Synthetic BOFF ability crop generator (one-off bootstrap tool).

Generates augmented 64×64 crops from cargo wiki PNGs to seed the embedder with
classes that have insufficient real training data — e.g. the ~100 ground BOFF
abilities the embedder currently does not know.

Run from CLI:
    python -m warp.trainer.synthetic_crop_generator --env ground -n 100

Wiki icons are 64×49 BGRA (ground/space alike). Real in-game BOFF crops are
~37×29 BGR at the detector's output. The central trainer
(sets-warp-backend/admin_train_metric.py) normalizes both to 64×64 via
cv2.resize before RandomResizedCrop → 224×224. We therefore emit 64×64 PNGs
so the synthetic-vs-real pixel domain is uniform from the trainer's view.

Augmentation goals (domain-realistic, not just cosmetic):
  - random STO-like dark BG patch (gradient + noise) padding the canvas
  - alpha composite (icons have transparent corners)
  - bbox jitter (±2-3 px shift before center-crop)
  - color jitter (HSV brightness/saturation/hue)
  - JPEG re-encode round-trip (gameplay screens are JPEG)
  - optional radial cooldown overlay (~10% samples)

NOT uploaded to HuggingFace — these are local bootstrap data only. Real crops
remain the community contribution path. See
project_ground_bootstrap_workstream.md for the full design.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import quote_plus

import cv2
import numpy as np

from warp import userdata
from warp.data.cargo import boff_abilities, icons_dir
from warp.debug import log


OUTPUT_SIZE   = 64                # final crop side (matches IMG_SIZE in central trainer)
CANVAS_PAD    = 4                 # extra px around canvas for shift room
ICON_H_RANGE  = (44, 60)          # icon height after scale jitter (px)
BBOX_JITTER   = 3                 # max px shift from canvas centre
JPEG_QUALITY  = (70, 95)
COLOR_BRIGHT  = 0.15              # ±15% brightness
COLOR_SAT     = 0.20              # ±20% saturation
COLOR_HUE_DEG = 5                 # ±5° hue
COOLDOWN_PROB = 0.10              # fraction of samples that get cooldown overlay


def _slug(name: str) -> str:
    """Filesystem-safe class slug. Matches the convention used elsewhere
    (TrainingDataManager): lowercase, non-alphanumeric → underscore."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _env_ability_names(env: str) -> list[str]:
    cache = boff_abilities()
    env_dict = cache.get(env) or {}
    names: set[str] = set()
    for _prof, rank_lists in env_dict.items():
        if not isinstance(rank_lists, (list, tuple)):
            continue
        for rank_dict in rank_lists:
            if isinstance(rank_dict, dict):
                names.update(rank_dict.keys())
    return sorted(names)


def _load_wiki_icon(name: str) -> np.ndarray | None:
    """Load the cargo wiki PNG for an ability, preserving alpha when present."""
    d = icons_dir()
    if d is None:
        return None
    p = d / f'{quote_plus(name)}.png'
    if not p.exists():
        return None
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


def _bg_patch(size: int, rng: np.random.Generator) -> np.ndarray:
    """Random dark gradient-and-noise background patch (size×size BGR uint8).
    STO UI surfaces are dark navy/grey/black with subtle gradient; we
    approximate that distribution rather than sample real screens (the
    bootstrap is meant to be reproducible without screen corpora).
    """
    base = rng.integers(8, 50, size=3).astype(np.float32)
    grad_dir = rng.uniform(0, 2 * np.pi)
    grad_mag = rng.uniform(5, 25)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    grad = (np.cos(grad_dir) * (x / size) + np.sin(grad_dir) * (y / size)) * grad_mag
    bg = np.broadcast_to(base, (size, size, 3)).astype(np.float32) + grad[..., None]
    bg += rng.normal(0, 4, (size, size, 3))
    return np.clip(bg, 0, 255).astype(np.uint8)


def _alpha_composite(icon_bgra: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    """Composite an icon (BGRA or BGR) onto a BGR background of equal size."""
    if icon_bgra.shape[2] == 4:
        alpha = icon_bgra[..., 3:4].astype(np.float32) / 255.0
        fg = icon_bgra[..., :3].astype(np.float32)
        bg = bg_bgr.astype(np.float32)
        out = fg * alpha + bg * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
    return icon_bgra


def _color_jitter(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # OpenCV H range is [0, 180] ≡ [0°, 360°], so degrees / 2.
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-COLOR_HUE_DEG, COLOR_HUE_DEG) / 2.0) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + rng.uniform(-COLOR_SAT, COLOR_SAT)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + rng.uniform(-COLOR_BRIGHT, COLOR_BRIGHT)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _jpeg_recode(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = int(rng.integers(JPEG_QUALITY[0], JPEG_QUALITY[1] + 1))
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img_bgr
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _cooldown_overlay(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Partial radial dim overlay mimicking STO's ability cooldown sweep."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    sweep = int(rng.uniform(60, 300))
    cv2.ellipse(mask, (w // 2, h // 2), (max(w, h), max(w, h)),
                0, -90, -90 + sweep, 255, -1)
    dimmed = (img_bgr.astype(np.float32) * 0.4).astype(np.uint8)
    return np.where(mask[..., None] > 0, dimmed, img_bgr)


def _synthesize_one(icon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a single 64×64 BGR augmented crop from an icon (BGRA or BGR)."""
    src_h, src_w = icon.shape[:2]
    target_h = int(rng.integers(ICON_H_RANGE[0], ICON_H_RANGE[1] + 1))
    target_w = max(1, round(target_h * src_w / src_h))
    icon_r = cv2.resize(icon, (target_w, target_h), interpolation=cv2.INTER_AREA)

    canvas = OUTPUT_SIZE + 2 * CANVAS_PAD
    bg = _bg_patch(canvas, rng)
    off_x = (canvas - target_w) // 2 + int(rng.integers(-BBOX_JITTER, BBOX_JITTER + 1))
    off_y = (canvas - target_h) // 2 + int(rng.integers(-BBOX_JITTER, BBOX_JITTER + 1))
    off_x = max(0, min(canvas - target_w, off_x))
    off_y = max(0, min(canvas - target_h, off_y))
    bg_region = bg[off_y:off_y+target_h, off_x:off_x+target_w]
    bg[off_y:off_y+target_h, off_x:off_x+target_w] = _alpha_composite(icon_r, bg_region)

    start = (canvas - OUTPUT_SIZE) // 2
    crop = bg[start:start+OUTPUT_SIZE, start:start+OUTPUT_SIZE]

    crop = _color_jitter(crop, rng)
    if rng.random() < COOLDOWN_PROB:
        crop = _cooldown_overlay(crop, rng)
    crop = _jpeg_recode(crop, rng)
    return crop


def generate_for_env(
    env: str,
    n_per_class: int,
    output_root: Path,
    seed: int = 42,
) -> tuple[int, list[str]]:
    """Generate `n_per_class` synthetic crops for every ability in `env`.
    Returns (n_written, missing_names) — `missing_names` lists abilities for
    which no local wiki PNG was found (should be empty when cargo cache is hot).
    """
    if env not in ('ground', 'space'):
        raise ValueError(f'env must be "ground" or "space", got {env!r}')

    names = _env_ability_names(env)
    log.info(f'SyntheticGen: env={env} classes={len(names)} n_per_class={n_per_class}')

    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    missing: list[str] = []
    for name in names:
        icon = _load_wiki_icon(name)
        if icon is None:
            missing.append(name)
            continue
        class_dir = output_root / _slug(name)
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            crop = _synthesize_one(icon, rng)
            cv2.imwrite(str(class_dir / f'{i:04d}.png'), crop)
            written += 1
    if missing:
        log.warning(
            f'SyntheticGen: {len(missing)} abilities without local wiki PNG '
            f'(skipped): {missing[:5]}{"..." if len(missing) > 5 else ""}'
        )
    log.info(f'SyntheticGen: wrote {written} crops to {output_root}')
    return written, missing


def _default_output(env: str) -> Path:
    return userdata.training_data_dir() / 'synthetic_crops' / env


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate synthetic BOFF ability crops from wiki PNGs for embedder bootstrap.'
    )
    parser.add_argument('--env', choices=['ground', 'space'], default='ground',
                        help='Which BOFF environment to generate for (default: ground)')
    parser.add_argument('-n', '--n-per-class', type=int, default=100,
                        help='Synthetic crops per class (default: 100)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output root (default: <training_data>/synthetic_crops/<env>)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed (default: 42)')
    args = parser.parse_args()

    out = args.output or _default_output(args.env)
    n_written, missing = generate_for_env(args.env, args.n_per_class, out, seed=args.seed)
    print(f'Wrote {n_written} crops across {len(_env_ability_names(args.env)) - len(missing)} classes '
          f'to {out}')
    if missing:
        print(f'WARNING: {len(missing)} classes skipped (no local wiki PNG): {missing[:5]}'
              f'{"..." if len(missing) > 5 else ""}')


if __name__ == '__main__':
    main()
