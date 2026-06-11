# warp/trainer/training_data.py
# Manages the local training dataset:
#   - Stores annotations (bbox + slot + name + state) per screenshot
#   - Exports cropped icon images for ML training
#   - Maintains annotations.json as the source of truth
#   - Provides API for SyncWorker to read data for upload
#
# Storage layout (under warp/training_data/):
#   annotations.json          — all annotations, keyed by image content sha256
#   crops/                    — exported icon crops (named by hash)
#   crops/crop_index.json     — maps crop filename → item name + slot
#
# annotations.json schema (current):
#   { "<sha16>": {
#       "filename":     "overview.png",     # last known filename — display only
#       "image_sha256": "<full 64-char>",   # forensic / dedup
#       "annotations":  [ann_dict, ...],
#   }, ... }
#
# sha16 = first 16 hex chars of sha256(file_bytes). 64 bits of entropy — for
# any realistic library size, collision probability is negligible (P < 10⁻¹²
# at 10k files). Keying by content hash means a screenshot saved as
# "overview.png" no longer inherits annotations from a totally different
# screenshot also saved as "overview.png" in the past.
#
# Legacy entries: pre-migration annotations.json keyed by filename (raw
# `list[dict]` values, not the wrapper above). They are loaded into
# `self._legacy_annotations` and IGNORED by `get_annotations` — so they
# never pollute a fresh screenshot's review panel. They stay in the JSON
# file untouched until the user runs the migration CLI
# (warp/trainer/migrate_annotations_to_hash.py).

from __future__ import annotations

import json
import hashlib
import shutil
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AnnotationState(str, Enum):
    CANDIDATE = "candidate"   # auto-detected, not yet reviewed
    PENDING   = "pending"     # drawn by user, not yet confirmed
    CONFIRMED = "confirmed"   # confirmed by user (has name + slot)
    SKIPPED   = "skipped"     # user chose to skip (unknown item)


# Slots where only one confirmed annotation is allowed per image.
# Accepting a new confirmed annotation for these slots removes any existing one
# at a different bbox position (prevents duplicates from misclicks).
SINGLE_INSTANCE_SLOTS: frozenset = frozenset({
    'Ship Name', 'Ship Type', 'Ship Tier',
    'Deflector', 'Sec-Def', 'Engines', 'Warp Core', 'Shield',
    'Kit', 'Body Armor', 'EV Suit', 'Personal Shield',
    'Primary Specialization', 'Secondary Specialization',
})

# Slots that contain text / fixed values — NOT icons.
# Bboxes for these slots are stored in annotations.json for layout learning
# (so OCR can be directed to the right region) but must NOT generate crop PNGs
# or crop_index entries, since ML cannot classify free-form text.
NON_ICON_SLOTS: frozenset = frozenset({
    'Ship Name',   # free-text, unique per ship — OCR only
    'Ship Type',   # fixed vocabulary, but text — OCR only
    'Ship Tier',   # T1–T6-X2 — OCR only
})

# Sub-categories of NON_ICON_SLOTS with different data-handling behaviour:
#
# POSITION_ONLY_SLOTS — bbox saved for layout anchoring only.
#   No crop PNG, no crop_index entry, no upload. (Privacy: Ship Name is the
#   player's character name — never stored beyond bbox coordinates.)
POSITION_ONLY_SLOTS: frozenset = frozenset({'Ship Name'})


def _bbox_iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (x, y, w, h) bboxes."""
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

# TEXT_LEARNING_SLOTS — crop PNG + confirmed text + ml_name (raw OCR) saved
#   and uploaded to HF staging so the central pipeline can build
#   ship_type_corrections.json via democratic voting.
TEXT_LEARNING_SLOTS: frozenset = frozenset({'Ship Type', 'Ship Tier'})

# Virtual item names — annotated by user to teach the ML what empty/inactive slots look like.
# Crop is saved and uploaded like any icon annotation, but must NOT be written to the SETS build.
VIRTUAL_ITEM_NAMES: frozenset = frozenset({'__empty__', '__inactive__'})


@dataclass
class Annotation:
    """One bounding-box annotation for an icon in a screenshot."""
    bbox:     tuple           # (x, y, w, h) in original image pixels
    slot:     str  = ""       # SETS slot name, e.g. "Fore Weapons"
    name:     str  = ""       # SETS item name
    state:    AnnotationState = AnnotationState.PENDING
    ann_id:   str  = ""       # unique ID (hash of image+bbox)
    ml_conf:  float = 0.0     # original ML recognition confidence (0.0 = unknown)
    ml_name:  str  = ""       # what ML originally recognised (may differ from confirmed name)
    crop_name: str = ""       # relative path of saved crop PNG (set by _export_crop)
    auto_confirmed: bool = False  # True if confirmed by auto-accept threshold (yellow), False if user-confirmed (green)
    # Community name the user rejected when resolving a community_conflict on
    # this annotation. Persists across restarts so the same community proposal
    # doesn't keep nagging the user — until the community DB flips to a
    # different name (or to the user's pick, in which case the field becomes
    # irrelevant because there's no conflict). Empty = no conflict ever
    # resolved here.
    community_rejected: str = ""
    # Layout metadata — persisted so the per-seat grouping survives a
    # restart without forcing the user to re-run Auto-Detect. seat_key
    # uses the same format `_remap_boff_seat_slots` produces (e.g.
    # 'Boff Seat L[T]_483'); slot_index is the in-row left→right ordinal
    # for sort-stable display. Empty / -1 = unknown (legacy entry).
    seat_key:   str = ""
    slot_index: int = -1

    def __post_init__(self):
        if not self.ann_id:
            self.ann_id = self._make_id()

    def _make_id(self) -> str:
        raw = f"{self.bbox[0]}_{self.bbox[1]}_{self.bbox[2]}_{self.bbox[3]}_{self.slot}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


class TrainingDataManager:
    """
    Manages all annotation data for the WARP CORE trainer.

    Thread safety: not thread-safe — call from Qt main thread only.
    """

    ANNOTATIONS_FILE    = "annotations.json"
    CROPS_DIR           = "crops"
    CROP_INDEX_FILE     = "crops/crop_index.json"
    SCREEN_TYPES_FILE   = "screen_types.json"
    USER_CONFIRMED_FILE = "screen_types_user_confirmed.json"

    def __init__(self, data_dir: Path):
        self._dir       = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / self.CROPS_DIR).mkdir(exist_ok=True)

        # Active store — keyed by 16-char sha256 prefix of file contents.
        self._annotations:  dict[str, list[dict]] = {}   # sha16 → list of ann dicts
        self._image_meta:   dict[str, dict]       = {}   # sha16 → {filename, image_sha256}
        self._crop_index:   dict[str, dict]       = {}   # crop_filename → metadata
        self._screen_types: dict[str, str]        = {}   # sha16 → screen type string
        self._screen_types_user_confirmed: set[str] = set()  # sha16 set

        # Legacy bucket — pre-migration annotations.json data, keyed by
        # filename. Loaded, persisted, but NEVER consulted by lookups.
        # Drained by `migrate_legacy_by_path` (one-shot CLI).
        self._legacy_annotations: dict[str, list[dict]] = {}
        self._legacy_screen_types: dict[str, str] = {}
        self._legacy_screen_types_user_confirmed: set[str] = set()

        # Hash cache — sha256 of a screenshot is recomputed only when
        # (path, mtime_ns, size) changes. Without this, every get_annotations
        # in the trainer's hot review loop would re-read the file.
        self._hash_cache:      dict[tuple, str] = {}
        self._full_hash_cache: dict[str, str]   = {}  # sha16 → full 64-char

        self._dirty = False

        self._load()
        # Auto-repair crop_index on startup (fixes entries from pre-fix versions)
        try:
            repaired = self.repair_crop_index()
            if repaired:
                logger.info(f'TrainingDataManager: auto-repaired {repaired} crop_index entries')
        except Exception as e:
            logger.warning(f'TrainingDataManager: repair_crop_index failed: {e}')
        # Sweep crops left behind by the pre-fix correction bugs:
        # - crop_index entries whose ann_id no longer matches any annotation
        # - crop_index entries whose slot/name diverged from the annotation
        # - crop PNG files on disk that no longer have any crop_index entry
        try:
            n_orphan, n_resync, n_stale_files = self.cleanup_orphaned_crops()
            if n_orphan or n_resync or n_stale_files:
                logger.info(
                    f'TrainingDataManager: cleanup — {n_orphan} orphaned entries, '
                    f'{n_resync} label re-syncs, {n_stale_files} unindexed PNGs removed')
        except Exception as e:
            logger.warning(f'TrainingDataManager: cleanup_orphaned_crops failed: {e}')

    # ---------------------------------------------------------------- image id

    def _image_id(self, image_path: Path) -> str:
        """Return the 16-char sha256 prefix of the image's contents.

        Caches per (path, mtime_ns, size) so the file is only re-read when
        actually modified — keeps the trainer's hot lookup loop cheap.

        If the file is missing the lookup degrades to a deterministic
        `missing__<filename>` sentinel: `get_annotations` will simply
        return nothing for it (no stale match by filename), and any
        write path that needs the actual bytes will fail downstream
        with the underlying I/O error.
        """
        try:
            st = image_path.stat()
        except OSError:
            return f'missing__{image_path.name}'
        cache_key = (str(image_path), st.st_mtime_ns, st.st_size)
        cached = self._hash_cache.get(cache_key)
        if cached is not None:
            return cached
        h = hashlib.sha256()
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        full = h.hexdigest()
        short = full[:16]
        self._hash_cache[cache_key] = short
        self._full_hash_cache[short] = full
        return short

    def _register_image(self, image_path: Path) -> str:
        """Hash + record metadata for write paths. Returns the sha16 key."""
        key = self._image_id(image_path)
        # Last-write-wins: trainer may see this image at a different path
        # in the future, but the bytes-derived id stays stable. We refresh
        # the human-readable filename so the JSON remains greppable.
        self._image_meta[key] = {
            'filename':     image_path.name,
            'image_sha256': self._full_hash_cache.get(key, ''),
        }
        return key

    # ---------------------------------------------------------------- annotation CRUD

    def get_annotations(self, image_path: Path) -> list[Annotation]:
        """Returns all annotations for the given image (as Annotation objects).

        Lookup is by content hash — annotations from a different screenshot
        that happened to share this one's filename are NOT returned.
        Legacy filename-keyed entries are inert until migrated.
        """
        key  = self._image_id(image_path)
        dicts = self._annotations.get(key, [])
        return [self._dict_to_ann(d) for d in dicts]

    def has_annotations(self, image_path: Path) -> bool:
        key = self._image_id(image_path)
        anns = self._annotations.get(key, [])
        return any(a.get("state") == AnnotationState.CONFIRMED for a in anns)

    def has_legacy_annotations(self, image_path: Path) -> bool:
        """True iff the filename has untranslated legacy entries.

        Used by the trainer UI to surface a 'Migrate legacy annotations?'
        prompt for files whose name matches a pre-hash entry — and only
        for those, so the prompt never appears for genuinely fresh files.
        """
        return image_path.name in self._legacy_annotations

    def add_annotation(
        self,
        image_path: Path,
        bbox: tuple,
        slot: str = "",
        name: str = "",
        state: AnnotationState = AnnotationState.PENDING,
        ml_conf: float = 0.0,
        ml_name: str = "",
        auto_confirmed: bool = False,
        community_rejected: str = "",
        seat_key: str = "",
        slot_index: int = -1,
    ) -> Annotation:
        """Add or update annotation, treating the same bbox as the same annotation.

        Lookup priority:
          1. Match by ann_id (exact, same bbox+slot — no change needed)
          2. Match by bbox alone — handles slot/name edits on existing bbox
             (old ann_id had slot baked in; new slot → different ann_id, same bbox)
          3. Single-instance slots: remove any other confirmed annotation for the
             same slot before inserting (prevents duplicate Deflector, Shield, etc.)
          4. Insert as new annotation
        """
        ann = Annotation(bbox=bbox, slot=slot, name=name, state=state,
                         ml_conf=ml_conf, ml_name=ml_name,
                         auto_confirmed=auto_confirmed,
                         community_rejected=community_rejected,
                         seat_key=seat_key, slot_index=slot_index)
        key = self._register_image(image_path)
        if key not in self._annotations:
            self._annotations[key] = []

        # 1. Update in-place if ann_id already exists (exact match — fast path).
        # ann_id = hash(bbox + slot), so step 1 fires when only name/state/ml_*
        # changed. We still call _sync_crop_index so the crop PNG gets renamed
        # to match the new name and crop_index entry is refreshed — otherwise
        # corrections (user edits the item name) would leave the old crop on
        # disk under the old filename and the index/HF upload would carry the
        # stale name forever.
        for i, d in enumerate(self._annotations[key]):
            if d.get('ann_id') == ann.ann_id:
                old_name = d.get('name', '')
                old_state = d.get('state', '')
                self._annotations[key][i] = asdict(ann)
                self._dirty = True
                if old_name != ann.name or old_state != ann.state:
                    try:
                        self._sync_crop_index(image_path, ann)
                    except Exception as e:
                        logger.warning(f"step1: crop_index sync failed for {image_path.name}: {e}")
                return ann

        # 2. Fallback: same bbox (exact or IoU overlap), slot was edited →
        # update in-place without duplicating.
        # Skip for NON_ICON_SLOTS — Ship Name/Type/Tier can share bbox coords
        # legitimately; a slot-agnostic match would overwrite one with another.
        bbox_t = tuple(bbox)
        if slot not in NON_ICON_SLOTS:
            best_iou_i, best_iou_val = -1, 0.0
            for i, d in enumerate(self._annotations[key]):
                d_bbox = tuple(d.get('bbox', []))
                if not d_bbox or len(d_bbox) < 4:
                    continue
                if d_bbox == bbox_t:
                    # Exact bbox match — update in-place immediately
                    best_iou_i = i
                    break
                # IoU dedup: a 1-2 px shifted bbox (e.g. user-drawn vs
                # detector grid) for the same slot must not create a
                # second annotation at the same physical position.
                if d.get('slot') == slot:
                    iou = _bbox_iou(bbox_t, d_bbox)
                    if iou > best_iou_val:
                        best_iou_val, best_iou_i = iou, i
            if best_iou_i >= 0 and (best_iou_val >= 0.5
                                     or tuple(self._annotations[key][best_iou_i].get('bbox', [])) == bbox_t):
                d = self._annotations[key][best_iou_i]
                old_ann_id = d.get('ann_id', '')
                self._annotations[key][best_iou_i] = asdict(ann)
                self._dirty = True
                if old_ann_id and old_ann_id != ann.ann_id:
                    self._cleanup_crops_for_ann_id(old_ann_id)
                try:
                    self._export_crop(image_path, ann)
                    self._sync_crop_index(image_path, ann)
                except Exception as e:
                    logger.warning(f"step2: export/sync failed for {image_path.name}: {e}")
                return ann

        # 3. Single-instance slots: drop any existing confirmed annotation for
        #    this slot at a different bbox (user re-drew at correct position)
        if state == AnnotationState.CONFIRMED and slot in SINGLE_INSTANCE_SLOTS:
            before = len(self._annotations[key])
            self._annotations[key] = [
                d for d in self._annotations[key]
                if not (d.get('slot') == slot and d.get('state') == 'confirmed')
            ]

        # 4. New annotation
        self._annotations[key].append(asdict(ann))
        self._dirty = True
        try:
            self._export_crop(image_path, ann)
            # Sync state in crop_index (export sets PENDING, update if CONFIRMED)
            self._sync_crop_index(image_path, ann)
        except Exception as e:
            logger.warning(f"Could not export crop for {image_path.name}: {e}")
        return ann

    def add_candidate(
        self,
        image_path: Path,
        slot: str,
        slot_index: int,
        bbox: tuple,
    ) -> bool:
        """
        Add an auto-detected candidate bbox if not already present.
        Returns True if added, False if duplicate.
        """
        key = self._register_image(image_path)
        existing = self._annotations.get(key, [])

        # Check for duplicate (same slot + index)
        for d in existing:
            if d.get("slot") == slot and d.get("slot_index") == slot_index:
                return False

        ann = Annotation(bbox=bbox, slot=slot, name="", state=AnnotationState.CANDIDATE)
        d   = asdict(ann)
        d["slot_index"] = slot_index
        if key not in self._annotations:
            self._annotations[key] = []
        self._annotations[key].append(d)
        self._dirty = True
        return True

    def update_annotation(
        self, image_path: Path, ann: Annotation,
        bbox: tuple | None = None,
    ):
        """Update an existing annotation in-place (matched by OLD ann_id).
        If bbox is provided, replaces ann.bbox before saving — the new ann_id
        is recomputed, but lookup MUST still use the original ann_id so the
        old dict can be found and the stale crop cleaned up.
        If state is CONFIRMED, also updates crop_index and re-exports crop."""
        from dataclasses import replace as dc_replace
        # Preserve the ORIGINAL ann_id for lookup before dc_replace recomputes it.
        old_ann_id = ann.ann_id
        if bbox is not None:
            ann = dc_replace(ann, bbox=bbox)
        key = self._image_id(image_path)
        dicts = self._annotations.get(key, [])
        for i, d in enumerate(dicts):
            if d.get("ann_id") == old_ann_id:
                dicts[i] = asdict(ann)
                self._dirty = True
                # bbox change → ann_id changes → old crop file/index entry are
                # now stale. Remove them and re-export under the new ann_id.
                if old_ann_id != ann.ann_id:
                    self._cleanup_crops_for_ann_id(old_ann_id)
                    try:
                        self._export_crop(image_path, ann)
                    except Exception as e:
                        logger.warning(f"update_annotation: export failed for {image_path.name}: {e}")
                # Sync crop_index state + re-export crop if confirmed
                self._sync_crop_index(image_path, ann)
                return

    def _cleanup_crops_for_ann_id(self, ann_id: str) -> int:
        """Remove crop_index entries + crop PNG files whose filename contains
        the given ann_id. Returns number of files removed.

        Used when an annotation's ann_id changes (slot or bbox edit) — the
        crop file is named with the OLD ann_id and would otherwise leak into
        the dataset under stale slot/name forever.
        """
        if not ann_id:
            return 0
        to_remove = [f for f in self._crop_index if ann_id in f]
        for fname in to_remove:
            del self._crop_index[fname]
            crop_path = self._dir / self.CROPS_DIR / fname
            if crop_path.exists():
                crop_path.unlink()
        if to_remove:
            self._dirty = True
        return len(to_remove)

    def remove_annotation(self, image_path: Path, ann: Annotation):
        """Remove an annotation by ann_id and clean up its crop_index entry + PNG."""
        key = self._image_id(image_path)
        dicts = self._annotations.get(key, [])
        self._annotations[key] = [d for d in dicts if d.get("ann_id") != ann.ann_id]
        self._dirty = True
        self._cleanup_crops_for_ann_id(ann.ann_id)

    # ---------------------------------------------------------------- persistence

    def save(self):
        """Write annotations.json + screen_types.json to disk.

        annotations.json carries both the new sha16-keyed wrapper entries
        and any untouched legacy filename-keyed lists side-by-side; on
        next load `_load` re-splits them by value shape. Same for
        screen_types — sha16 keys for migrated entries, filename keys
        kept inert for legacy.
        """
        ann_path = self._dir / self.ANNOTATIONS_FILE
        out: dict = {}
        for key, anns in self._annotations.items():
            meta = self._image_meta.get(key, {})
            out[key] = {
                'filename':     meta.get('filename', ''),
                'image_sha256': meta.get('image_sha256', ''),
                'annotations':  anns,
            }
        # Preserve legacy entries as raw lists so they round-trip untouched.
        for fname, anns in self._legacy_annotations.items():
            out[fname] = anns
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        idx_path = self._dir / self.CROP_INDEX_FILE
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(self._crop_index, f, indent=2)

        # Screen types: merge active (sha16-keyed) + legacy (filename-keyed)
        # back into one map so legacy data is preserved untouched.
        st_out: dict = dict(self._legacy_screen_types)
        st_out.update(self._screen_types)
        st_path = self._dir / self.SCREEN_TYPES_FILE
        with open(st_path, "w", encoding="utf-8") as f:
            json.dump(st_out, f, indent=2)

        uc_out = sorted(
            self._screen_types_user_confirmed
            | self._legacy_screen_types_user_confirmed)
        uc_path = self._dir / self.USER_CONFIRMED_FILE
        with open(uc_path, "w", encoding="utf-8") as f:
            json.dump(uc_out, f, indent=2)

        self._dirty = False
        logger.info(f"Training data saved to {self._dir}")

    def _migrate_clear_ship_name_text(self) -> None:
        """One-time migration: clear stored text from Ship Name annotations.
        Ship Name is position-only — content was never meant to be persisted (privacy).
        Ship Type and Ship Tier are now TEXT_LEARNING_SLOTS and keep their confirmed text.
        """
        _CLEAR_SLOTS = frozenset({'Ship Name'})
        changed = 0
        # Active sha16-keyed entries AND inert legacy ones — both pools may
        # contain unscrubbed Ship Name text from before this migration ran.
        for pool in (self._annotations, self._legacy_annotations):
            for anns in pool.values():
                for ann in anns:
                    if ann.get('slot') in _CLEAR_SLOTS and ann.get('name', '').strip():
                        ann['name'] = ''
                        changed += 1
        if changed:
            logger.info(f'Migration: cleared name text from {changed} Ship Name annotations')
            self.save()

    def _load(self):
        ann_path = self._dir / self.ANNOTATIONS_FILE
        if ann_path.exists():
            try:
                with open(ann_path, encoding='utf-8') as f:
                    raw = json.load(f)
                for key, val in raw.items():
                    # New schema: wrapper dict with explicit metadata.
                    if isinstance(val, dict) and 'annotations' in val:
                        self._annotations[key]  = val.get('annotations', [])
                        self._image_meta[key]   = {
                            'filename':     val.get('filename', ''),
                            'image_sha256': val.get('image_sha256', ''),
                        }
                        full = val.get('image_sha256', '')
                        if full:
                            self._full_hash_cache[key] = full
                    # Legacy schema: bare list of annotation dicts, keyed by filename.
                    # Drop empty placeholders — files the user opened in CORE
                    # but never annotated. They carry no info and would just
                    # inflate the "legacy entries remaining" count forever.
                    elif isinstance(val, list):
                        if val:
                            self._legacy_annotations[key] = val
                        else:
                            self._dirty = True   # so save() rewrites without them
                    # Anything else (corrupt) — skip and warn.
                    else:
                        logger.warning(
                            f'_load: skipped unrecognised annotations entry {key!r}')
            except Exception as e:
                logger.warning(f"Could not load annotations: {e}")
        self._migrate_clear_ship_name_text()

        idx_path = self._dir / self.CROP_INDEX_FILE
        if idx_path.exists():
            try:
                with open(idx_path, encoding='utf-8') as f:
                    self._crop_index = json.load(f)
            except Exception:
                pass

        st_path = self._dir / self.SCREEN_TYPES_FILE
        if st_path.exists():
            try:
                with open(st_path, encoding='utf-8') as f:
                    raw_st = json.load(f)
                for key, stype in raw_st.items():
                    if self._is_sha16(key):
                        self._screen_types[key] = stype
                    else:
                        self._legacy_screen_types[key] = stype
            except Exception:
                pass
        uc_path = self._dir / self.USER_CONFIRMED_FILE
        if uc_path.exists():
            try:
                with open(uc_path, encoding='utf-8') as f:
                    raw_uc = json.load(f)
                for key in raw_uc:
                    if self._is_sha16(key):
                        self._screen_types_user_confirmed.add(key)
                    else:
                        self._legacy_screen_types_user_confirmed.add(key)
            except Exception:
                pass

    @staticmethod
    def _is_sha16(key: str) -> bool:
        """True iff `key` looks like a 16-char lowercase hex sha-prefix."""
        if not isinstance(key, str) or len(key) != 16:
            return False
        return all(c in '0123456789abcdef' for c in key)

    def _sync_crop_index(self, image_path: Path, ann: Annotation):
        """
        Update crop_index entry for this annotation:
        - Updates state (PENDING → CONFIRMED etc.)
        - Updates name and slot (may change after user confirmation)
        - Re-exports crop if it doesn't exist yet
        - POSITION_ONLY_SLOTS are skipped — their bbox is kept in annotations.json
          for layout learning, but no crop PNG or crop_index entry is created.
        - TEXT_LEARNING_SLOTS get a crop PNG + crop_index entry that includes ml_name
          so SyncWorker can upload them for OCR correction training.
        """
        if ann.slot in POSITION_ONLY_SLOTS:
            return

        safe_slot = ann.slot.replace(" ", "_").lower()
        safe_name = (ann.name or "unknown").replace(" ", "_").lower()[:40]
        fname     = f"{safe_slot}__{safe_name}__{ann.ann_id}.png"
        out_path  = self._dir / self.CROPS_DIR / fname

        # Also look for any existing crop with this ann_id (name/slot may have changed)
        old_fname = next(
            (f for f, m in self._crop_index.items() if ann.ann_id in f), None)
        if old_fname and old_fname != fname:
            # Rename crop file to reflect new name/slot
            old_path = self._dir / self.CROPS_DIR / old_fname
            if old_path.exists():
                old_path.rename(out_path)
            del self._crop_index[old_fname]

        # Export crop if missing
        if not out_path.exists():
            try:
                self._export_crop(image_path, ann)
            except Exception as e:
                logger.warning(f'_sync_crop_index: export failed: {e}')

        # Update index entry with current state/name/slot.
        # auto_confirmed=True means the detector accepted on a confidence
        # threshold without user verification — keep crop_index PENDING so
        # get_confirmed_crops() does not upload it as ground truth. User accept
        # flips auto_confirmed=False, which lets the state promote to CONFIRMED.
        _idx_state = AnnotationState.PENDING if ann.auto_confirmed else ann.state
        entry: dict = {
            'slot':   ann.slot,
            'name':   ann.name,
            'state':  _idx_state,
            'source': image_path.name,
        }
        if ann.slot in TEXT_LEARNING_SLOTS:
            entry['ml_name'] = ann.ml_name
        self._crop_index[fname] = entry

    # ---------------------------------------------------------------- crop export

    def _export_crop(self, image_path: Path, ann: Annotation):
        """
        Crops the icon region from the original screenshot and saves it as PNG.
        Filename is derived from item name + slot (for easy dataset browsing).
        POSITION_ONLY_SLOTS are skipped — no crop needed (position anchor only).
        TEXT_LEARNING_SLOTS get a crop so the text region can be uploaded for OCR training.
        """
        if ann.slot in POSITION_ONLY_SLOTS:
            return
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return

        x, y, w, h = ann.bbox
        h_img, w_img = img.shape[:2]
        # Clamp to image bounds
        x  = max(0, x); y = max(0, y)
        x2 = min(w_img, x + w); y2 = min(h_img, y + h)
        crop = img[y:y2, x:x2]

        if crop.size == 0:
            return

        # Build filename: slot_name + ann_id
        safe_slot = ann.slot.replace(" ", "_").lower()
        safe_name = (ann.name or "unknown").replace(" ", "_").lower()[:40]
        fname     = f"{safe_slot}__{safe_name}__{ann.ann_id}.png"
        out_path  = self._dir / self.CROPS_DIR / fname

        cv2.imwrite(str(out_path), crop)

        # Write crop_name back into the annotation so seed_from_training_data
        # can find the file without having to reconstruct the path.
        relative = f'crops/{fname}'
        ann.crop_name = relative
        key = self._image_id(image_path)
        for d in self._annotations.get(key, []):
            if d.get('ann_id') == ann.ann_id:
                d['crop_name'] = relative
                self._dirty = True
                break

        # Update crop index
        _entry: dict = {
            "slot":   ann.slot,
            "name":   ann.name,
            "state":  ann.state,
            "source": image_path.name,
        }
        if ann.slot in TEXT_LEARNING_SLOTS:
            _entry["ml_name"] = ann.ml_name
        self._crop_index[fname] = _entry

    # ---------------------------------------------------------------- export for sync

    def repair_crop_index(self):
        """
        One-time repair: for every confirmed annotation in annotations.json,
        ensure the corresponding crop PNG exists and has state=CONFIRMED in crop_index.
        Call this once after upgrading from a version with the crop_index bug.
        """
        import cv2
        repaired = 0
        for image_key, ann_list in self._annotations.items():
            # image_key is sha16; humans browse crop_index by filename,
            # so resolve via the meta sidecar.
            image_name = self._image_meta.get(image_key, {}).get('filename', image_key)
            for d in ann_list:
                if d.get('state') != AnnotationState.CONFIRMED:
                    continue
                ann = self._dict_to_ann(d)
                if not ann.name:
                    continue
                safe_slot = ann.slot.replace(' ', '_').lower()
                safe_name = (ann.name or 'unknown').replace(' ', '_').lower()[:40]
                fname = f'{safe_slot}__{safe_name}__{ann.ann_id}.png'
                # Update crop_index entry to CONFIRMED
                entry = self._crop_index.get(fname)
                if entry is None or entry.get('state') != AnnotationState.CONFIRMED:
                    # Try to find any existing crop with this ann_id
                    existing = next(
                        (f for f in self._crop_index if ann.ann_id in f), None)
                    if existing and existing != fname:
                        old_path = self._dir / self.CROPS_DIR / existing
                        new_path = self._dir / self.CROPS_DIR / fname
                        if old_path.exists():
                            old_path.rename(new_path)
                        del self._crop_index[existing]
                    _repair_entry: dict = {
                        'slot':   ann.slot,
                        'name':   ann.name,
                        'state':  (AnnotationState.PENDING if ann.auto_confirmed
                                   else AnnotationState.CONFIRMED),
                        'source': image_name,
                    }
                    if ann.slot in TEXT_LEARNING_SLOTS:
                        _repair_entry['ml_name'] = ann.ml_name
                    self._crop_index[fname] = _repair_entry
                    # Export crop PNG if missing
                    crop_path = self._dir / self.CROPS_DIR / fname
                    if not crop_path.exists():
                        # Find original screenshot to re-export from
                        # We don't have its path here, so just mark as repaired
                        pass
                    repaired += 1

                # Backfill crop_name into annotation dict if missing
                if not d.get('crop_name'):
                    crop_path = self._dir / self.CROPS_DIR / fname
                    if crop_path.exists():
                        d['crop_name'] = f'crops/{fname}'
                        self._dirty = True

        if repaired or self._dirty:
            self.save()
            if repaired:
                logger.info(f'repair_crop_index: fixed {repaired} entries')
        return repaired

    def cleanup_orphaned_crops(self) -> tuple[int, int, int]:
        """Sweep stale data left by the pre-fix correction bugs.

        Returns (orphaned_entries_removed, label_resyncs, stale_png_files_removed).

        Three classes of pollution this handles:
          1. crop_index entry whose ann_id is not present in annotations.json
             at all (annotation was deleted, slot or bbox changed without
             cleanup). The entry + PNG are dropped.
          2. crop_index entry whose ann_id matches an existing annotation
             but the slot/name diverged from the current annotation values
             (user corrected the name; old code skipped _sync_crop_index).
             The PNG is renamed to match the current label and the index
             entry is overwritten.
          3. PNG files on disk inside crops/ that are not referenced by any
             crop_index entry. Removed to reclaim space.
        """
        # Build {ann_id: (image_key, dict)} for fast lookup. Include the legacy
        # bucket — its crops are still valid training data and would otherwise
        # look "orphaned" to the sweep and get deleted on first startup.
        ann_by_id: dict[str, tuple[str, dict]] = {}
        for pool in (self._annotations, self._legacy_annotations):
            for image_key, ann_list in pool.items():
                for d in ann_list:
                    aid = d.get('ann_id')
                    if aid:
                        ann_by_id[aid] = (image_key, d)

        orphaned = 0
        resynced = 0
        crops_dir = self._dir / self.CROPS_DIR

        # Pass 1+2: walk crop_index, fix or drop entries.
        for fname in list(self._crop_index.keys()):
            entry = self._crop_index.get(fname, {})
            # Filename schema: '{slot}__{name}__{ann_id}.png'
            stem = fname.rsplit('.', 1)[0]
            parts = stem.split('__')
            if len(parts) < 3:
                continue  # unknown shape, leave alone
            ann_id = parts[-1]
            match = ann_by_id.get(ann_id)
            if match is None:
                # Orphaned — annotation gone or ann_id changed without cleanup.
                del self._crop_index[fname]
                p = crops_dir / fname
                if p.exists():
                    p.unlink()
                orphaned += 1
                self._dirty = True
                continue
            _img_name, ann_d = match
            cur_slot = ann_d.get('slot', '')
            cur_name = ann_d.get('name', '')
            idx_slot = entry.get('slot', '')
            idx_name = entry.get('name', '')
            if cur_slot != idx_slot or cur_name != idx_name:
                # Rename file to match current label.
                safe_slot = cur_slot.replace(' ', '_').lower()
                safe_name = (cur_name or 'unknown').replace(' ', '_').lower()[:40]
                new_fname = f'{safe_slot}__{safe_name}__{ann_id}.png'
                old_path = crops_dir / fname
                new_path = crops_dir / new_fname
                if old_path.exists() and not new_path.exists():
                    old_path.rename(new_path)
                del self._crop_index[fname]
                new_entry = dict(entry)
                new_entry['slot'] = cur_slot
                new_entry['name'] = cur_name
                self._crop_index[new_fname] = new_entry
                resynced += 1
                self._dirty = True

        # Pass 3: walk crops/ dir, drop PNGs not in the index.
        stale_files = 0
        if crops_dir.exists():
            indexed = set(self._crop_index.keys())
            for p in crops_dir.iterdir():
                if not p.is_file() or p.suffix.lower() != '.png':
                    continue
                if p.name not in indexed:
                    try:
                        p.unlink()
                        stale_files += 1
                    except Exception as e:
                        logger.warning(f'cleanup: cannot remove {p.name}: {e}')

        if orphaned or resynced or stale_files:
            self.save()
        return orphaned, resynced, stale_files

    def get_confirmed_crops(self) -> list[dict]:
        """
        Returns list of confirmed crop metadata dicts for upload.
        Each dict: { path, slot, name, source }
        """
        result = []
        crops_dir = self._dir / self.CROPS_DIR
        for fname, meta in self._crop_index.items():
            if meta.get("state") == AnnotationState.CONFIRMED:
                p = crops_dir / fname
                if p.exists():
                    result.append({"path": str(p), **meta})
        return result

    def get_stats(self) -> dict:
        """Returns summary statistics for the dataset."""
        total      = sum(len(v) for v in self._annotations.values())
        confirmed  = sum(
            1 for v in self._annotations.values()
            for d in v if d.get("state") == AnnotationState.CONFIRMED
        )
        images     = len(self._annotations)
        slots: dict[str, int] = {}
        for v in self._annotations.values():
            for d in v:
                s = d.get("slot", "unknown")
                slots[s] = slots.get(s, 0) + 1
        return {
            "images":    images,
            "total":     total,
            "confirmed": confirmed,
            "slots":     slots,
        }

    # ---------------------------------------------------------------- screen type persistence

    def set_screen_type(self, image_path: Path, screen_type: str, user_confirmed: bool = False) -> Path:
        """
        Records the screen type for a screenshot (persisted to screen_types.json)
        and copies it into the classifier training folder.

        Persistent label: warp/training_data/screen_types.json  {filename: stype}
        Training copy:    warp/training_data/screen_types/<stype>/<filename>

        Returns the destination path of the training copy.
        """
        key = self._register_image(image_path)
        self._screen_types[key] = screen_type
        if user_confirmed:
            self._screen_types_user_confirmed.add(key)
        self.save()
        dest_dir = self._dir / 'screen_types' / screen_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / image_path.name
        shutil.copy2(image_path, dest)
        logger.info(f'Screen type set: {image_path.name} -> {screen_type}')
        return dest

    def get_screen_type(self, image_path: Path) -> str:
        """Returns the persisted screen type for a screenshot, or empty string if not set.

        Active (sha16-keyed) entries win; legacy filename-keyed entries are
        ignored so a fresh screenshot sharing a name with a pre-migration
        file does not inherit its label.
        """
        return self._screen_types.get(self._image_id(image_path), '')

    def get_all_screen_types(self) -> dict[str, str]:
        """Returns {filename: stype} for the trainer's file list.

        Storage is keyed by sha16 (content hash); the trainer iterates by
        Path and looks up by `p.name`, so we project back through
        `_image_meta`. Legacy filename-keyed entries are folded in last,
        meaning an active sha16 entry for a different file wins over a
        legacy entry sharing the same filename.
        """
        out: dict[str, str] = dict(self._legacy_screen_types)
        for sha, stype in self._screen_types.items():
            fname = self._image_meta.get(sha, {}).get('filename')
            if fname:
                out[fname] = stype
        return out

    def get_user_confirmed_set(self) -> set[str]:
        """Returns filenames confirmed explicitly by the user (green checkmark)."""
        out: set[str] = set(self._legacy_screen_types_user_confirmed)
        for sha in self._screen_types_user_confirmed:
            fname = self._image_meta.get(sha, {}).get('filename')
            if fname:
                out.add(fname)
        return out

    def is_user_confirmed(self, image_path: Path) -> bool:
        return self._image_id(image_path) in self._screen_types_user_confirmed

    def remove_user_confirmed(self, image_path: Path) -> None:
        """Remove user-confirmed mark. Type label in screen_types.json is kept."""
        key = self._image_id(image_path)
        if key in self._screen_types_user_confirmed:
            self._screen_types_user_confirmed.discard(key)
            self.save()

    def remove_screen_type(self, image_path: Path, screen_type: str) -> bool:
        """
        Removes a screenshot from a screen type training folder
        AND removes it from the persistent screen_types.json label.
        Returns True if anything was removed.
        """
        removed = False
        key = self._image_id(image_path)
        # Remove from persistent label dict and user-confirmed set
        if key in self._screen_types:
            del self._screen_types[key]
            removed = True
        self._screen_types_user_confirmed.discard(key)
        if removed:
            self.save()
        # Remove training copy from disk
        dest = self._dir / 'screen_types' / screen_type / image_path.name
        if dest.exists():
            dest.unlink()
            logger.info(f'Screen type removed: {image_path.name} (label + training copy)')
            removed = True
        return removed

    def get_screen_type_counts(self) -> dict[str, int]:
        """
        Returns the number of training examples per screen type class.
        Dict keys are class names (folder names), values are image counts.
        """
        screen_types_dir = self._dir / 'screen_types'
        if not screen_types_dir.exists():
            return {}
        return {
            d.name: len(list(d.glob('*.png')) + list(d.glob('*.jpg')) + list(d.glob('*.jpeg')))
            for d in screen_types_dir.iterdir()
            if d.is_dir()
        }

    # ---------------------------------------------------------------- legacy migration

    def migrate_legacy_by_path(self, image_path: Path) -> int:
        """Promote legacy filename-keyed entries to sha16 keys.

        Caller supplies the original screenshot file on disk; we hash it,
        copy legacy annotations / screen-type / user-confirmed flag from
        the filename bucket into the active sha16 bucket, then drop the
        legacy entries. Returns the number of legacy annotation rows
        promoted (0 if nothing matched).
        """
        fname = image_path.name
        leg_anns = self._legacy_annotations.get(fname)
        leg_stype = self._legacy_screen_types.get(fname)
        leg_uc = fname in self._legacy_screen_types_user_confirmed
        if not (leg_anns or leg_stype or leg_uc):
            return 0

        key = self._register_image(image_path)
        promoted = 0
        if leg_anns:
            existing = self._annotations.setdefault(key, [])
            seen_ids = {d.get('ann_id') for d in existing if d.get('ann_id')}
            for d in leg_anns:
                if d.get('ann_id') in seen_ids:
                    continue
                existing.append(d)
                promoted += 1
            del self._legacy_annotations[fname]
        if leg_stype and key not in self._screen_types:
            self._screen_types[key] = leg_stype
        if fname in self._legacy_screen_types:
            del self._legacy_screen_types[fname]
        if leg_uc:
            self._screen_types_user_confirmed.add(key)
            self._legacy_screen_types_user_confirmed.discard(fname)

        self._dirty = True
        self.save()
        logger.info(
            f'migrate_legacy_by_path: {fname} -> {key} ({promoted} ann row(s))')
        return promoted

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _dict_to_ann(d: dict) -> Annotation:
        return Annotation(
            bbox=tuple(d.get("bbox", (0, 0, 0, 0))),
            slot=d.get("slot", ""),
            name=d.get("name", ""),
            state=AnnotationState(d.get("state", AnnotationState.PENDING)),
            ann_id=d.get("ann_id", ""),
            ml_conf=float(d.get("ml_conf", 0.0)),
            ml_name=d.get("ml_name", ""),
            auto_confirmed=bool(d.get("auto_confirmed", False)),
            community_rejected=d.get("community_rejected", ""),
            seat_key=d.get("seat_key", ""),
            slot_index=int(d.get("slot_index", -1)),
        )

    def update_layout_fields(
        self, image_path: Path, ann_id: str,
        seat_key: str = "", slot_index: int = -1,
    ) -> bool:
        """Backfill `seat_key` / `slot_index` on an existing annotation.

        Used after Auto-Detect re-derives the seat-keyed layout for a
        legacy entry that was saved before these fields existed. Updates
        in place — no ann_id rehash, no crop re-export. Returns True if
        anything actually changed (caller decides when to flush via
        `save()`).
        """
        if not ann_id:
            return False
        key = self._image_id(image_path)
        for d in self._annotations.get(key, []):
            if d.get("ann_id") != ann_id:
                continue
            changed = False
            if seat_key and d.get("seat_key", "") != seat_key:
                d["seat_key"] = seat_key
                changed = True
            if slot_index >= 0 and int(d.get("slot_index", -1)) != int(slot_index):
                d["slot_index"] = int(slot_index)
                changed = True
            if changed:
                self._dirty = True
            return changed
        return False
