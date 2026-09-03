# warp/recognition/icon_matcher.py
#
# Matches cropped icon images against SETS item icon library.
#
# SETS stores downloaded item images in:
#   <config_folder>/images/<quote_plus(item_name)>.png
#
# RECOGNITION STRATEGY (in priority order):
#
#   Stage 1 — Multi-scale template matching (primary, no training needed)
#     For each slot crop:
#       a) Resize crop to MATCH_SIZE×MATCH_SIZE
#       b) For each template in index: cv2.matchTemplate (TM_CCOEFF_NORMED)
#       c) Best match above TEMPLATE_THRESHOLD wins
#     Advantages over pHash+histogram:
#       - Sensitive to icon shape, not just color distribution
#       - Robust to STO's icon rendering at different UI scales
#       - Works immediately from the SETS image cache (no ML training)
#
#   Stage 2 — Color histogram fallback
#     When template matching confidence is low, use HSV histogram correlation
#     as a secondary signal to break ties or rescue near-misses.
#
#   Stage 3 — ML classifier (optional, downloaded from HF Hub)
#     ONNX EfficientNet-B0 trained on SETS icon library.
#     Activated when both Stage 1+2 fail (conf < ML_TRIGGER_THRESHOLD).
#
# The public match() method returns (name, confidence, thumbnail_QImage).

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote_plus

import numpy as np

from warp import userdata
from warp.debug import log, syslog

# ── Tunable thresholds ─────────────────────────────────────────────────────────
MATCH_SIZE          = 64     # resize crop + template to this before matching
TEMPLATE_THRESHOLD  = 0.55   # min TM_CCOEFF_NORMED score to accept a match
HIST_WEIGHT         = 0.20   # weight of histogram score when blending with template
HIST_THRESHOLD      = 0.50   # min histogram correlation to contribute
ML_PRIMARY_THRESHOLD= 0.50   # ML conf >= this → ML is the source of truth
# Gallery entries enrolled from clean wiki art rather than from a confirmed
# in-game crop. The shipped `embedding_index.npz` covers 2963 names while the
# icon library holds 4406, so a third of the library has no entry at all and
# the embedder cannot return those names at any confidence. Worse, it does not
# fall silent on them: measured over 1500 crops with their own item hidden, it
# answers with the nearest thing it does know at mean similarity 0.484, and
# 29.5% of those wrong answers clear ML_PRIMARY_THRESHOLD and take the slot
# from the template stage, which had the right answer all along.
#
# Enrolling the wiki PNG gives it something correct to point at. No training
# is involved — the embedder is a function, so the clean art is simply run
# through it and appended to the gallery.
#
# Art sits further from an in-game crop than a real crop of the same item
# does: measured, 0.549 against 0.881, a 0.332 penalty from the domain gap.
# Left uncorrected, art entries lose to real crops *of other items* — 97% of
# the residual errors. The offset compensates. It is deliberately small: a
# sweep of the trade-off showed +0.10 lifts unknown items from 0% to 74.9%
# while known items move +0.1 pp (96.9 → 97.0), and every larger value buys
# unknown-item accuracy with the 99% case (+0.30 costs 1.2 pp, +0.50 costs
# 5.3 pp). Re-measure both columns before changing it, never just one.
ART_SIM_OFFSET      = 0.10
VIRTUAL_OVERRIDE_CONF = 0.40 # when ML returns a real icon with conf >= this,
                             # suppress virtual (__empty__/__inactive__)
                             # session/template candidates
# Poison-guard for virtual labels (__empty__/__inactive__): a session crop
# that matches a query pixel-perfectly almost certainly IS the same crop
# (self-match against a mislabeled training entry). When the embedder
# disagrees by returning any real icon at conf >= POISON_GUARD_ML_MIN, treat
# the session-virtual win as poison and suppress it. Numbers calibrated on
# the tactical-console / Kentari-launcher cases (sess=1.000, embed=0.33).
SESSION_PIXEL_PERFECT       = 0.95
POISON_GUARD_ML_MIN         = 0.15
# Visual sanity for virtual-labeled session crops: a real __empty__ /
# __inactive__ is uniformly dim, so a crop that is both bright AND colour-
# rich cannot be a real virtual. Thresholds match warp.tools.scrub_training_data.
# Recalibrated 2026-07-17 from a visual review of the 20 community-mirror
# crops the seed logged as POISON skip: genuine empty/inactive BOFF slots
# carry a navy portrait tint reaching ~12% bright / ~12% rich, while real
# mislabeled icons sit >= 19%. The old 0.07 gate false-flagged the BOFF
# slots as poison and dropped them from training; 0.15 clears them while
# still catching the real icons.
VIRTUAL_SEED_BRIGHT_RATIO   = 0.15
VIRTUAL_SEED_RICH_RATIO     = 0.15
VIRTUAL_LABELS              = frozenset({'__empty__', '__inactive__'})
# Embedder-based virtual suppression: when the top real-icon gallery entry
# beats the top virtual gallery entry by at least this cosine margin, treat
# the slot as real regardless of absolute ML confidence. Replaces the crude
# bright/rich heuristic for slots where the embedder has a clear preference
# but its absolute conf is below VIRTUAL_OVERRIDE_CONF (e.g. partially clipped
# edge bbox at y=-1).
EMBED_REAL_VS_VIRTUAL_MARGIN = 0.05
# Template matching cutoff (TM_CCOEFF_NORMED below this is silently dropped).
# The unrestricted floor (TEMPLATE_THRESHOLD * 0.7 = 0.385) is correct when
# the matcher must discriminate across all 4070+ wiki PNGs. When the caller
# pins down a slot type (candidate_names), the search space shrinks 10-20x,
# so a 0.30 cutoff is informative and rescues items the embedder gallery is
# missing — e.g. Elite Fleet Dranuur Quantum Torpedo Launcher (TM≈0.37 on
# a real edge-clipped crop, formerly dropped, now wins).
TEMPLATE_RESTRICTED_THRESHOLD = 0.30
# Multi-scale template matching: build templates at these MATCH_SIZE-relative
# sizes so cv2.matchTemplate slides the wiki PNG inside the query crop and
# tolerates small misalignment (e.g. bbox y=-1 edge slots, icon offset by 1-3
# px). Smaller scales = larger sliding search. Two scales keeps compute modest.
TEMPLATE_SCALES = (58, MATCH_SIZE)
# The sliding scale used by `_template_scores`. 58 in a 64 px crop gives 49
# offsets, which is what absorbs an edge-clipped bbox being 1-3 px out.
_TEMPLATE_SLIDE_SIZE = 58
# Adaptive histogram weight: when the embedder is weak (conf below this), drop
# HIST_WEIGHT for template scoring. Game crops carry Mk overlays and rarity
# borders that distort the HSV histogram relative to clean wiki PNGs; relying
# less on histogram in fallback mode lets raw template correlation dominate.
TEMPLATE_HIST_WEAK_EMBED_THRESHOLD = 0.50
TEMPLATE_HIST_WEIGHT_WEAK_EMBED    = 0.10
# Cross-validation: when the top candidate is within this margin of a runner-
# up that has more name-agreement (other sources point at the same name), the
# winner switches to the agreeing candidate. Acts as a pure tiebreaker — the
# displayed confidence is always the raw score of the winning source, never
# inflated. Margin sized to typical noise between source scales (cosine vs
# template-correlation).
SOURCE_AGREEMENT_TIEBREAKER_MARGIN = 0.05
ML_TRIGGER_THRESHOLD= 0.50   # if combined conf below this, try ML stage (legacy)
FUSION_THRESHOLD    = 0.75   # P8: run ML and fuse scores when template < this (legacy)
HIST_BINS           = [18, 16] # H×S bins for _hist_hsv — must match everywhere

HF_REPO_ID          = 'sets-sto/icon-classifier'
HF_MODEL_FILENAME   = 'icon_classifier.onnx'
HF_LABELS_FILE      = 'label_map.json'
# Sentinel file written after a failed availability check.
# Prevents repeated 401/404 download attempts across sessions.
HF_UNAVAILABLE_FILE = 'model_unavailable.flag'
# How many hours to wait before retrying after a failed check
HF_RETRY_HOURS      = 24


# Era-variant icon art. STO draws some gear differently in 23rd-century
# content, and the wiki files that second picture as its own page —
# `File:Impulse Engines (23c) icon.png` beside `File:Impulse Engines
# icon.png`. The *item* is unchanged: one name, one cargo row, and the
# article renders both pictures side by side. Our icon index is keyed by
# filename, so a variant would otherwise enter it under a name no cargo
# row carries and be dropped by every candidate filter downstream.
_ERA_VARIANT_RE = re.compile(r'^(?P<base>.+?) \(23c\.?\)$')


def _base_item_name(icon_name: str, known_names: set[str]) -> str:
    """Map era-variant icon art onto the item it depicts.

    Driven by the item names cargo currently carries rather than by the
    spelling of the tag, because the tag is not reliably a variant marker:
    `Modified Phaser Pistol (23c.)` is a real item name, tag and all. The
    order of the checks is what keeps both readings working, and keeps
    working as the wiki changes:

    * the name is already an item        → leave it (a tagged item name)
    * else the base name is an item      → variant art, fold onto the base
    * neither                            → leave it (nothing to fold onto;
      `Matter Anti-Matter Warp Core (23c)` is in this state today, and
      starts folding by itself if that item ever gains a cargo row)

    With no cargo available `known_names` is empty and nothing is folded,
    which is exactly the behaviour that predates this function.
    """
    if not known_names or icon_name in known_names:
        return icon_name
    m = _ERA_VARIANT_RE.match(icon_name)
    if m and m.group('base') in known_names:
        return m.group('base')
    return icon_name


def _real_crop_looks_blank(crop_bgr) -> bool:
    """The mirror of `_virtual_crop_looks_real`: a crop carrying a real item's
    name that is in fact an empty or inactive cell.

    Every other guard in this file looks one way — a colourful crop labelled
    `__empty__`. Nothing looked for the opposite, and it is the more damaging
    of the two: a blank cell filed under an item's name teaches the gallery
    that this item *is* what nothing looks like, and the recogniser then
    answers with that item on every blank cell it meets. Confirming those
    answers feeds the loop.

    Measured on the community mirror 2026-09-03: of 9227 crops carrying a real
    item name, 25 are blank cells, and 20 of those 25 are the same name —
    `Charged Particle Burst`, which is 20 of the 29 crops that class has. An
    inactive BOFF cell sits at cosine 0.92 from those 20 and at 0.45 from the
    9 genuine ones, so the confusion is entirely their doing.

    The judgement is `LayoutDetector._classify_cell`, the same function the
    pipeline uses to decide a cell is blank before any matching runs. Measured
    against user-confirmed ground truth it calls 2 of 5833 real icons blank
    (0.03%), which is the cost of this guard: those two crops do not become
    session examples, and there are hundreds of others for their classes.
    """
    try:
        from warp.recognition.layout_detector import LayoutDetector
        return LayoutDetector._classify_cell(crop_bgr) != 'active'
    except Exception:
        return False


def _virtual_crop_looks_real(crop_bgr) -> bool:
    """Visual sanity check for a virtual-labeled crop (__empty__/__inactive__).
    Returns True when the crop is too bright AND too colour-rich to be a real
    empty / inactive slot — i.e. it is almost certainly mislabeled poison.
    Mirrors warp.tools.scrub_training_data heuristic so the seed-time filter
    and the offline scrub agree."""
    try:
        import cv2
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        bright = float((v > 150).mean())
        rich   = float(((s > 100) & (v > 100)).mean())
        return (bright > VIRTUAL_SEED_BRIGHT_RATIO
                and rich > VIRTUAL_SEED_RICH_RATIO)
    except Exception:
        return False


class SETSIconMatcher:
    """
    Multi-stage icon recognition against the SETS image cache.

    match(crop_bgr) -> (item_name, confidence, thumbnail_QImage, used_session)
      name=''  if no match above threshold.
      used_session=True when autonomous ML/template recognition failed and the
      result came from confirmed training-data crops (session examples).
      Callers should log this as a training gap for future ML improvement.
    """

    # Session examples: confirmed crops added by user during this session.
    # Shared across all instances so every match() call benefits.
    # Entry origin tags (live-seed support):
    #   'user'       — user clicked Accept in WARP CORE this process; passes
    #                  through reset_ml_session() filtering so WARP detection
    #                  can use it immediately. Single bbox at a time, in-memory
    #                  only — does NOT count as reading annotations.json from
    #                  disk, so the CLAUDE.md WARP-vs-CORE rule still holds.
    #   'community'  — HF-mirrored approved-truth (allowed in WARP).
    #   'trainer_td' — seed_from_training_data bulk seed (WARP CORE path only;
    #                  dropped from the pool whenever WARP runs).
    #   'session'    — generic / legacy (treated as user-equivalent).
    _session_examples: list[dict] = []   # {name, tmpl64, hist_hsv, orig, origin, crop_hash}

    # Guard: prevent re-seeding from training data on every new matcher instance.
    _seeded_from_training_data: bool = False
    # Same one-shot guard for the HF-mirrored approved-truth crops.
    _seeded_from_community: bool = False
    # mtime of data/annotations.jsonl at last seed — re-seed only when the
    # mirror moves, so periodic sync ticks are cheap when nothing changed.
    _seeded_community_mtime: float = 0.0

    def __init__(self, sets_app=None, sync_client=None):
        # `sets_app` is accepted for backward compatibility with the SETS
        # call sites (trainer code that still passes `self._sets`). When
        # None or any non-SETS-object, `_get_images_dir` falls back to
        # `warp.data.cargo.icons_dir()`. May also be a `str` / `Path`
        # pointing directly at the icon library.
        self._sets        = sets_app
        self._index: list[dict] = []   # {name, tmpl64, hist_hsv, path}
        # Flattened, TM_CCOEFF_NORMED-shaped template matrices, parallel to
        # `_index`. float32 and not float16: the smaller dtype halves the
        # ~395 MB and costs nothing in accuracy, but NumPy has no BLAS path
        # for float16 and falls back to a scalar loop — measured 5698 ms for
        # the same product float32 does in 6.3 ms. Storing float16 and casting
        # per query is no better; the cast alone is 64 ms.
        self._tmpl_mat58 = None        # np.ndarray (N, 58*58*3) float32
        self._tmpl_mat64 = None        # np.ndarray (N, 64*64*3) float32
        self._ml_session  = None
        self._ml_disabled = False      # True after first failed download attempt
        self._label_map: dict[int, str] = {}
        # Metric-learning path: when icon_embedder.pt is present, _ml_session is
        # the embedder model and _gallery_* hold the k-NN search index. When
        # _ml_kind=='classifier' (legacy softmax), _gallery_* stay None.
        self._ml_kind: str = ''        # 'embedder' | 'classifier' | ''
        self._gallery_emb = None       # np.ndarray (N, D) float32, L2-normed
        self._gallery_lbl = None       # np.ndarray (N,) int32 — indices into _label_map
        # True for gallery rows enrolled from wiki art rather than from a
        # confirmed crop. Drives ART_SIM_OFFSET and the 'art' match source.
        self._gallery_is_art = None    # np.ndarray (N,) bool
        self._last_embed_was_art = False
        # Diagnostic: source of the most recent match() decision.
        # Values: 'ml' (embedder/classifier), 'template' (wiki PNG histogram),
        # 'session' (confirmed training crop), 'knowledge' (pHash override),
        # 'none' (no signal above threshold), '' (no match attempted).
        # Read by warp_importer to expose match source in autodetect logs.
        self._last_match_src: str = ''
        # Filename of the art actually shown, when it is one of the 34
        # era-variant pictures folded onto a shared cargo name. Empty when the
        # picture is the item's only one — which is the usual case, so an
        # empty value must read as "nothing to say", not "unknown".
        self._last_match_variant: str = ''
        # When _last_match_src == 'session', this carries the winning entry's
        # `origin` tag ('user', 'community', 'trainer_td', or 'session'). Lets
        # warp_importer tag user-aided matches as [USER] in autodetect logs so
        # they're visibly distinguished from autonomous detection.
        self._last_match_origin: str = ''
        # Per-stage raw scores from the most recent match() call. Filled in
        # before every return path (knowledge / no-candidates / final winner).
        # Consumed by RecognitionWorker to build the per-image match summary
        # table. Keys: 'embed', 'soft', 'session', 'template', 'knowledge'.
        self._last_stage_scores: dict[str, float] = {
            'embed': 0.0, 'soft': 0.0, 'session': 0.0,
            'template': 0.0, 'knowledge': 0.0,
        }
        self._sync_client = sync_client  # WARPSyncClient | None
        self._build_index()

    # ── Public ─────────────────────────────────────────────────────────────────

    def match(
        self,
        crop_bgr: np.ndarray,
        candidate_names: set[str] | None = None,
    ) -> tuple[str, float, object, bool]:
        """
        Match a slot crop against the SETS icon library.

        candidate_names: optional set of allowed item names.
          When provided, only entries in this set are considered.

        ML-primary design (2026-05-15):
          Stage 0 — community pHash knowledge override (hard override, trust=1.0)
          Stage 1 — ML classifier (local PyTorch / HF ONNX) — PRIMARY SOURCE
                    when ml_conf >= ML_PRIMARY_THRESHOLD AND result is in candidate_names
          Fallback (only when Stage 1 is uncertain / out of candidates):
            Stage 2 — template matching + histogram (SETS wiki-icon cache)
            Stage 3 — session examples (confirmed training-data crops)
            Stage 4 — last resort: weak ML result (better than nothing)

        Rationale: ML is trained on real game-screenshot crops (via
        sync.py → admin_train.py), so it generalizes to actual rendered
        icons including virtual states (__empty__, __inactive__). Template
        matching against wiki PNGs and session examples suffer from HSV-
        distribution mismatch on dimly-rendered cells, producing false
        positives (e.g. filled icon → __empty__). Treating ML as primary
        eliminates that class of error; the fallback chain only kicks in
        for items genuinely missing from the model's label_map.

        Returns:
            (item_name, confidence, thumbnail_QImage, used_session)
            item_name='' and confidence=0.0 if nothing matched.
            used_session=True means Stage 3 (session example) rescued the
            result — a training gap signal for the caller.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            self._last_match_src = ''
            self._last_match_variant = ''
            self._last_match_origin = ''
            self._last_stage_scores = {'embed': 0.0, 'soft': 0.0,
                                       'session': 0.0, 'template': 0.0,
                                       'knowledge': 0.0}
            return '', 0.0, None, False

        import cv2
        self._last_match_src = ''
        self._last_match_variant = ''
        self._last_match_origin = ''
        self._last_stage_scores = {'embed': 0.0, 'soft': 0.0,
                                   'session': 0.0, 'template': 0.0,
                                   'knowledge': 0.0}
        # Embedder real-vs-virtual diagnostics, populated by _classify_ml_embed
        # when candidate_names is provided. Used by suppress_virtual logic below.
        self._last_embed_sim_real    = 0.0
        self._last_embed_sim_virtual = 0.0

        crop64 = cv2.resize(crop_bgr, (MATCH_SIZE, MATCH_SIZE),
                            interpolation=cv2.INTER_AREA)
        q_hist = self._hist_hsv(crop64)

        # Virtual labels (__empty__ / __inactive__) are orthogonal to the
        # caller's slot-type restriction — they answer "is this slot blank?",
        # not "which ability is this?". Always allow them through the
        # candidate_names filter so the embedder's virtual prediction is
        # never silenced by a restriction set built from the abilities cache.
        # Defense-in-depth: anti-virtual-bias rules below still suppress
        # false-positive virtual wins on real icons.
        if candidate_names is not None:
            candidate_names = candidate_names | {'__empty__', '__inactive__'}

        # Stage 0: community pHash knowledge override (hard override).
        # Embedder result is reused later by Stage 1, so cache it across the
        # cross-check + main flow.
        ml_name, ml_conf = ('', 0.0)
        ml_computed = False
        if self._sync_client is not None:
            try:
                from warp.knowledge.sync_client import _compute_phash
                phash     = _compute_phash(crop64)
                overrides = self._sync_client.get_knowledge()
                if phash in overrides:
                    name = overrides[phash]
                    # Defense-in-depth: never let knowledge.json hard-override a
                    # crop to a virtual class (__empty__ / __inactive__) or a
                    # leftover dev-test entry. Such entries pollute Stage 0 and
                    # used to silently turn real icons into empty slots at
                    # conf=1.0. Skip the override — fall through to ML/template.
                    suppress = False
                    if name.startswith('__') or name == 'Test Item Name':
                        log.debug(f'WARPSync: pHash override {name!r} suppressed (virtual/test)')
                        suppress = True
                    elif candidate_names is not None and name not in candidate_names:
                        log.debug(f'WARPSync: pHash override {name!r} rejected — not valid for slot')
                        suppress = True
                    else:
                        # Embedder cross-check: stale community entries from
                        # the pre-bootstrap era mapped blank-icon pHashes to
                        # real ability names (e.g. blanks → "Charged Particle
                        # Burst"). The bootstrapped embedder now correctly
                        # identifies blanks as virtual — if it says virtual
                        # with decent confidence, refuse the override.
                        if not self._ml_disabled:
                            ml_name, ml_conf = self._classify_ml(crop64, candidate_names)
                            ml_computed = True
                            if (ml_name.startswith('__')
                                    and ml_conf >= VIRTUAL_OVERRIDE_CONF):
                                log.debug(
                                    f'WARPSync: pHash override {name!r} rejected '
                                    f'— embedder says {ml_name!r} '
                                    f'(conf={ml_conf:.2f}); likely poisoned entry'
                                )
                                suppress = True
                    if not suppress:
                        log.debug(f'WARPSync: knowledge override → {name!r}')
                        self._last_match_src = 'knowledge'
                        self._last_stage_scores['knowledge'] = 1.0
                        return name, 1.0, self._bgr_to_qimage(crop_bgr), False
            except Exception as e:
                log.debug(f'WARPSync: override lookup failed: {e}')

        # Stage 1: ML classifier — always consulted (one of three signals).
        # Reuse result from Stage 0 cross-check if already computed.
        if not self._ml_disabled and not ml_computed:
            ml_name, ml_conf = self._classify_ml(crop64, candidate_names)

        # Stage 2: template matching + histogram against wiki PNGs
        # Slot-restricted callers (candidate_names provided) use a lower cutoff
        # because the search space is much smaller and faint-but-discriminative
        # matches stop being noise. Multi-scale templates absorb small offsets;
        # histogram weight drops when the embedder is uncertain (game-crop
        # overlays distort HSV away from clean wiki PNGs).
        auto_name  = ''
        auto_score = 0.0
        auto_entry = None
        tm_cutoff = (TEMPLATE_RESTRICTED_THRESHOLD
                     if candidate_names is not None
                     else TEMPLATE_THRESHOLD * 0.7)
        weak_embed = ml_conf < TEMPLATE_HIST_WEAK_EMBED_THRESHOLD
        hist_w     = TEMPLATE_HIST_WEIGHT_WEAK_EMBED if weak_embed else HIST_WEIGHT
        tm_all = self._template_scores(crop64)
        for i, entry in enumerate(self._index):
            if candidate_names is not None and entry['name'] not in candidate_names:
                continue
            tm_score = float(tm_all[i])
            if tm_score < tm_cutoff:
                continue
            h_score = max(0.0, float(cv2.compareHist(
                q_hist, entry['hist_hsv'], cv2.HISTCMP_CORREL)))
            combined = tm_score * (1.0 - hist_w) + h_score * hist_w
            if combined > auto_score:
                auto_score = combined
                auto_name  = entry['name']
                auto_entry = entry

        # Stage 3: session examples (confirmed training-data crops)
        sess_name, sess_score, sess_entry = self._best_session_match(
            crop64, q_hist, candidate_names)

        # Record raw per-stage scores for the summary table.
        if self._ml_kind == 'embedder':
            self._last_stage_scores['embed'] = float(ml_conf)
        elif self._ml_kind == 'classifier':
            self._last_stage_scores['soft']  = float(ml_conf)
        self._last_stage_scores['template'] = float(auto_score)
        self._last_stage_scores['session']  = float(sess_score)

        # Combine all signals — strongest wins. No hard threshold here;
        # caller (warp_importer) applies MIN_ACCEPT_CONF as final gate.
        # Anti-virtual-bias rule: when ML returned a real icon with decent
        # confidence (>= VIRTUAL_OVERRIDE_CONF), suppress virtual session /
        # template matches (__empty__/__inactive__). This is the Bug 2 fix —
        # session-virtual was beating real ML on filled icons due to HSV
        # histogram bias of dim cells. ML is still NOT mandatory to win;
        # template/session with a real icon name can outscore it.
        ml_real = bool(ml_name) and not ml_name.startswith('__')

        def _virtual(n: str) -> bool:
            return bool(n) and n.startswith('__')

        # Query-side visual sanity: is the input crop itself bright + colour-
        # rich? Real __empty__/__inactive__ slots in STO are uniformly dim.
        # If the QUERY looks like a real icon, no virtual label can be
        # correct — regardless of session/template scores. Same heuristic
        # and thresholds as the seed-time filter / scrub tool.
        q_hsv  = cv2.cvtColor(crop64, cv2.COLOR_BGR2HSV)
        q_s    = q_hsv[:, :, 1]
        q_v    = q_hsv[:, :, 2]
        q_bright = float((q_v > 150).mean())
        q_rich   = float(((q_s > 100) & (q_v > 100)).mean())
        query_looks_real = (q_bright > VIRTUAL_SEED_BRIGHT_RATIO
                            and q_rich > VIRTUAL_SEED_RICH_RATIO)

        # Anti-virtual-bias suppression (three rules):
        #   (a) ML returned a real icon with conf >= VIRTUAL_OVERRIDE_CONF (0.40)
        #   (b) Session returned a virtual at pixel-perfect score (>= 0.95)
        #       AND ML disagrees by returning ANY real icon at conf >= 0.15
        #       → almost certainly a self-match against a poison crop, even
        #       if the embedder lacks confidence.
        #   (c) Query crop is itself bright + colour-rich AND session OR
        #       template returned a virtual label → the input cannot be
        #       __empty__/__inactive__, kill the virtual win.
        sess_virtual_perfect = (
            _virtual(sess_name) and sess_score >= SESSION_PIXEL_PERFECT
        )
        sess_or_tmpl_virtual = _virtual(sess_name) or _virtual(auto_name)
        # Embedder-based: best real-icon gallery sim beats best virtual sim by
        # a clear margin → semantically a real icon, regardless of absolute conf.
        embed_says_real = (
            self._last_embed_sim_real
            > self._last_embed_sim_virtual + EMBED_REAL_VS_VIRTUAL_MARGIN
        )
        suppress_virtual = (
            (ml_real and ml_conf >= VIRTUAL_OVERRIDE_CONF)
            or (ml_real and ml_conf >= POISON_GUARD_ML_MIN and sess_virtual_perfect)
            or (embed_says_real and sess_or_tmpl_virtual)
            or (query_looks_real and sess_or_tmpl_virtual)
        )
        if (sess_virtual_perfect and ml_real and ml_conf >= POISON_GUARD_ML_MIN
                and ml_conf < VIRTUAL_OVERRIDE_CONF):
            log.warning(
                f"WARP: poison-guard fired — session={sess_name!r} "
                f"score={sess_score:.3f} but embed top-1={ml_name!r} "
                f"conf={ml_conf:.2f} → suppressing virtual session win"
            )
        if query_looks_real and sess_or_tmpl_virtual and not (
                ml_real and ml_conf >= VIRTUAL_OVERRIDE_CONF):
            log.warning(
                f"WARP: query-sanity guard fired — query bright={q_bright:.1%} "
                f"rich={q_rich:.1%} (real icon), but session={sess_name!r}@"
                f"{sess_score:.2f} tmpl={auto_name!r}@{auto_score:.2f} → "
                f"suppressing virtual"
            )
        if embed_says_real and sess_or_tmpl_virtual and not (
                ml_real and ml_conf >= VIRTUAL_OVERRIDE_CONF):
            log.warning(
                f"WARP: embed-margin guard fired — embed real="
                f"{self._last_embed_sim_real:.2f} > virtual="
                f"{self._last_embed_sim_virtual:.2f} (+{EMBED_REAL_VS_VIRTUAL_MARGIN:.2f}), "
                f"but session={sess_name!r}@{sess_score:.2f} "
                f"tmpl={auto_name!r}@{auto_score:.2f} → suppressing virtual"
            )

        candidates = []
        if sess_name and not (suppress_virtual and _virtual(sess_name)):
            candidates.append(('session', sess_name, sess_score, sess_entry))
        if auto_name and not (suppress_virtual and _virtual(auto_name)):
            candidates.append(('template', auto_name, auto_score, auto_entry))
        if ml_name and (candidate_names is None or ml_name in candidate_names):
            candidates.append(('ml', ml_name, ml_conf, None))
        if not candidates:
            self._last_match_src = 'none'
            return '', 0.0, None, False
        # Cross-validation: count how many real-icon sources agree on each name.
        # Used as a tiebreaker only — when the highest-scoring candidate is
        # within SOURCE_AGREEMENT_TIEBREAKER_MARGIN of a runner-up that has
        # more agreement votes, switch to the agreeing one. The displayed
        # confidence is always the winner's raw score (no inflation).
        name_votes: dict[str, int] = {}
        for _src, _name, _score, _entry in candidates:
            if _name and not _name.startswith('__'):
                name_votes[_name] = name_votes.get(_name, 0) + 1
        ordered = sorted(candidates, key=lambda x: -x[2])
        winner = ordered[0]
        for runner in ordered[1:]:
            if winner[2] - runner[2] > SOURCE_AGREEMENT_TIEBREAKER_MARGIN:
                break  # candidates further down are even worse
            if not runner[1] or runner[1].startswith('__'):
                continue
            if name_votes.get(runner[1], 0) > name_votes.get(winner[1], 0):
                log.debug(
                    f"WARP: source-agreement tiebreaker → {runner[1]!r} "
                    f"({runner[0]}@{runner[2]:.2f}) over {winner[1]!r} "
                    f"({winner[0]}@{winner[2]:.2f}) — within "
                    f"{SOURCE_AGREEMENT_TIEBREAKER_MARGIN:.2f} margin, "
                    f"{name_votes[runner[1]]} sources agree"
                )
                winner = runner
        src, name, score, entry = winner
        # Cargo lifeline: wiki PNG template rescued a slot where the embedder
        # was uncertain (likely missing class in gallery). Logged so we can
        # spot which items would benefit most from extra training data.
        if (src == 'template'
                and ml_conf < TEMPLATE_HIST_WEAK_EMBED_THRESHOLD
                and candidate_names is not None):
            log.info(
                f"WARP: cargo lifeline → {name!r}@{score:.2f} (embed weak "
                f"@{ml_conf:.2f} — falling back to wiki PNG template)"
            )
        # Disambiguate ML source by model kind so logs distinguish the
        # ArcFace embedder from the legacy softmax classifier.
        if src == 'ml' and self._ml_kind == 'embedder':
            # 'art' when the winning gallery row came from a wiki PNG rather
            # than a confirmed crop, so the two can be told apart in the logs.
            self._last_match_src = 'art' if self._last_embed_was_art else 'embed'
        elif src == 'ml':
            self._last_match_src = 'soft'
        else:
            self._last_match_src = src
        if src == 'session' and entry is not None:
            self._last_match_origin = entry.get('origin', 'session')
        if entry is not None:
            # Session examples carry no 'variant' — they are confirmed crops,
            # not wiki art — so this is empty for them, correctly.
            self._last_match_variant = entry.get('variant', '') or ''
            thumb = self._bgr_to_qimage(entry.get('orig'))
        else:
            thumb = self._thumb_for_name(name, tm_all)
        return name, score, thumb, (src == 'session')

    def _thumb_for_name(self, name: str, tm_scores=None) -> object:
        """Return a QImage thumbnail for an item name, from the wiki PNG index.

        Used when the winner carries no entry of its own — an embedder match
        names an item without saying which picture of it was seen, because the
        gallery is keyed on names and knows nothing of variants.

        34 items have two pictures: the base art and a 23rd-century one, folded
        onto the same name by `_base_item_name` so the variant is recognisable
        at all. Returning the first of them is a coin toss, and it lands wrong
        often enough to notice: on the reported screenshot the two entries for
        'Phaser Dual Heavy Cannons' scored 0.443 and 0.574 against the crop,
        and the tooltip showed the 0.443 one — a visibly different weapon from
        the one on screen, under the right name.

        `tm_scores` is the per-entry template score `match()` has already
        computed for this crop, so picking the picture that actually matches
        costs nothing. Without it the old first-hit behaviour stands.

        Returns None for virtual items (__empty__/__inactive__) or when the
        name is not in the index.
        """
        if not name or name.startswith('__'):
            return None
        rows = [i for i, entry in enumerate(self._index) if entry['name'] == name]
        if not rows:
            return None
        if tm_scores is not None and len(rows) > 1:
            best = max(rows, key=lambda i: float(tm_scores[i]))
        else:
            best = rows[0]
        self._last_match_variant = self._index[best].get('variant', '') or ''
        return self._bgr_to_qimage(self._index[best].get('orig'))

    def _best_session_match(
        self,
        crop64: np.ndarray,
        q_hist: np.ndarray,
        candidate_names: set[str] | None,
    ) -> tuple[str, float, dict | None]:
        """Return (name, score, entry) for the best session example match."""
        import cv2
        expected_shape = tuple(HIST_BINS)
        sess_name  = ''
        sess_score = 0.0
        sess_entry = None
        for entry in self._session_examples:
            if candidate_names is not None and entry['name'] not in candidate_names:
                continue
            if entry['hist_hsv'].shape != expected_shape:
                continue
            res      = cv2.matchTemplate(crop64, entry['tmpl64'],
                                         cv2.TM_CCOEFF_NORMED)
            tm_score = float(res.max())
            h_score  = max(0.0, float(cv2.compareHist(
                q_hist, entry['hist_hsv'], cv2.HISTCMP_CORREL)))
            combined = tm_score * (1.0 - HIST_WEIGHT) + h_score * HIST_WEIGHT
            if combined > sess_score:
                sess_score = combined
                sess_name  = entry['name']
                sess_entry = entry
        return sess_name, sess_score, sess_entry

    def classify_patch(self, patch_bgr: np.ndarray) -> tuple[str, float]:
        """Classify a single BGR patch using ML only (fast path for dense scanning)."""
        import cv2
        if patch_bgr is None or patch_bgr.size == 0:
            return '', 0.0
        crop64 = cv2.resize(patch_bgr, (MATCH_SIZE, MATCH_SIZE), interpolation=cv2.INTER_AREA)
        return self._classify_ml(crop64)

    # ── Index building ──────────────────────────────────────────────────────────

    @staticmethod
    def _tm_vector(patch: 'np.ndarray') -> 'np.ndarray':
        """Flatten a patch the way TM_CCOEFF_NORMED compares it.

        That metric centres each channel on its own mean and then normalises
        over the whole block, so the correlation at one offset is the dot
        product of two such vectors. Getting the centring wrong — one mean
        over all channels instead of one per channel — still produces
        plausible numbers, just not the ones OpenCV would.
        """
        v = patch.astype(np.float32)
        v = v - v.reshape(-1, v.shape[2]).mean(axis=0)
        v = v.ravel()
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def _template_scores(self, crop64: 'np.ndarray') -> 'np.ndarray':
        """Best TM_CCOEFF_NORMED score of every indexed icon, in index order.

        Replaces a loop of `cv2.matchTemplate` calls — one per icon per scale,
        so ~8800 of them — with two matrix products. Each call does almost no
        arithmetic, so the loop was nearly all call overhead: measured 882 ms
        against 12 ms here, on the same 4406-icon index.

        The 58 px scale slides inside the 64 px crop to absorb the 1-3 px
        misalignment of an edge-clipped bbox, and it wins 94% of the per-icon
        maxima, so it cannot be dropped. Its 49 offsets become 49 columns of
        one matrix-matrix product rather than 49 separate passes.

        Verified against the `cv2` loop on 80 confirmed crops: identical item
        and identical confidence to four decimals for every one of them.
        """
        if self._tmpl_mat58 is None or self._tmpl_mat64 is None:
            return np.zeros(len(self._index), dtype=np.float32)
        s = _TEMPLATE_SLIDE_SIZE
        span = MATCH_SIZE - s + 1
        windows = np.stack(
            [self._tm_vector(crop64[dy:dy + s, dx:dx + s])
             for dy in range(span) for dx in range(span)],
            axis=1,
        ).astype(self._tmpl_mat58.dtype)
        slid = (self._tmpl_mat58 @ windows).max(axis=1).astype(np.float32)
        exact = (self._tmpl_mat64 @
                 self._tm_vector(crop64).astype(self._tmpl_mat64.dtype)
                 ).astype(np.float32)
        return np.maximum(slid, exact)

    def _build_index(self):
        """
        Load all PNG files from the SETS images directory and build
        a template + histogram index for fast matching.
        """
        images_dir = self._get_images_dir()
        if images_dir is None or not images_dir.exists():
            log.warning(
                'WARP: SETS images directory not found — '
                'icon matching disabled. '
                f'Expected: {images_dir}'
            )
            return

        import cv2
        # Item names as cargo has them, used to fold era-variant art onto the
        # item it depicts. Guarded: cargo being unavailable costs the folding
        # and nothing else, and every other source degrades the same way.
        try:
            from warp.data.cargo import canonical_names
            known_names = canonical_names()
        except Exception as exc:
            log.warning(f'WARP: icon index has no cargo names ({exc!r}) — '
                        'era-variant art will not be folded')
            known_names = set()

        count = 0
        folded = 0
        # Allocate the template matrices up front and write each icon straight
        # into its row. Building lists of per-icon vectors first and stacking
        # them at the end doubles the peak — ~395 MB of intermediates on top
        # of the ~395 MB result — and glibc does not return that to the OS,
        # so the process stays large for the rest of the run.
        # Sorted so the index order is the same on every machine. Some items
        # share one piece of art — 'Vulcan Lirpa' and 'Advanced Fleet Vulcan
        # Lirpa' score identically — and the winner among them is decided by
        # whichever comes first, so an unsorted glob made that depend on the
        # filesystem.
        pngs = sorted(images_dir.glob('*.png'))
        _d58 = _TEMPLATE_SLIDE_SIZE * _TEMPLATE_SLIDE_SIZE * 3
        _d64 = MATCH_SIZE * MATCH_SIZE * 3
        mat58 = np.empty((len(pngs), _d58), dtype=np.float32)
        mat64 = np.empty((len(pngs), _d64), dtype=np.float32)
        for png in pngs:
            name = unquote_plus(png.stem)
            raw_name = name
            base = _base_item_name(name, known_names)
            if base != name:
                # Two entries now answer to the same item name, one per era.
                # The index is a list scanned for the best score, so both
                # compete and the picture the screenshot actually shows wins.
                name = base
                folded += 1
            orig = cv2.imread(str(png))
            if orig is None:
                continue

            tmpl64 = cv2.resize(orig, (MATCH_SIZE, MATCH_SIZE),
                                 interpolation=cv2.INTER_AREA)
            self._index.append({
                'name':        name,
                # The filename this art came from, when era folding renamed
                # it. 34 items have both base and 23rd-century art under one
                # cargo name, so 'which picture' is real information the name
                # alone cannot carry — see `_last_match_variant`.
                'variant':     raw_name if raw_name != name else '',
                'hist_hsv':    self._hist_hsv(tmpl64),
                'orig':        orig,      # kept for thumbnail generation
            })
            mat58[count] = self._tm_vector(
                cv2.resize(orig, (_TEMPLATE_SLIDE_SIZE, _TEMPLATE_SLIDE_SIZE),
                           interpolation=cv2.INTER_AREA))
            mat64[count] = self._tm_vector(tmpl64)
            count += 1

        # Trim to what actually loaded: a PNG that failed to decode was
        # skipped, so `count` is the number of rows written.
        self._tmpl_mat58 = mat58[:count] if count else None
        self._tmpl_mat64 = mat64[:count] if count else None

        log.info(f'WARP: indexed {count} icons from {images_dir}'
                 + (f' ({folded} era-variant folded onto their item)'
                    if folded else ''))

    def _get_images_dir(self) -> Path | None:
        arg = self._sets
        # Direct path: trainer/importer can pass icons_dir explicitly.
        if isinstance(arg, (str, Path)):
            return Path(arg)
        # Legacy SETS app object: read its config dict.
        if arg is not None:
            try:
                return Path(arg.config['config_subfolders']['images'])
            except Exception:
                pass
            try:
                base = Path(arg.config['config_folder'])
                candidate = base / 'images'
                if candidate.exists():
                    return candidate
            except Exception:
                pass
        # Standalone sto-warp default: cargo-managed icons directory.
        try:
            from warp.data.cargo import icons_dir
            return icons_dir()
        except Exception:
            return None

    # ── Feature helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _hist_hsv(icon_bgr: np.ndarray) -> np.ndarray:
        """
        Normalised HSV histogram.
        Using H(18 bins) × S(16 bins) — ignores Value to be lighting-robust.
        """
        import cv2
        hsv  = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, HIST_BINS, [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist

    # ── ML helpers ──────────────────────────────────────────────────────────────

    def _classify_ml(
        self,
        crop64: np.ndarray,
        candidate_names: set[str] | None = None,
    ) -> tuple[str, float]:
        """Run local PyTorch classifier on a 64x64 BGR crop.
        Falls back to ONNX session for HuggingFace-downloaded model.

        Preprocessing must match admin_train.py CropDataset.__getitem__:
          1. BGR → RGB  (training uses cv2.COLOR_BGR2RGB)
          2. /255.0
          3. ImageNet mean/std normalization  (training uses T.Normalize)
        Missing either step produces a completely wrong input distribution
        (model was trained on normalized RGB, but would receive raw BGR).

        candidate_names: when provided, embedder k-NN selects the best label
        within that set. Prevents the slot from dropping to src=none when
        absolute top-1 is a non-slot-valid class (e.g. console picked for a
        weapon slot). Softmax classifier path is unaffected.
        """
        import cv2
        model = self._get_ml_session()
        if model is None:
            return '', 0.0
        # Metric-learning path: model is an Embedder, _gallery_* hold the k-NN index.
        if self._ml_kind == 'embedder':
            return self._classify_ml_embed(crop64, candidate_names)
        rgb = cv2.cvtColor(cv2.resize(crop64, (224, 224)), cv2.COLOR_BGR2RGB)
        inp = rgb.astype(np.float32) / 255.0
        # ImageNet normalization (same as T.Normalize in admin_train.py)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (inp - mean) / std
        inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), axis=0)
        try:
            if hasattr(model, 'run'):  # onnxruntime session (HuggingFace model)
                out   = model.run(None, {'input': inp})[0][0]
                probs = self._softmax(out)
            else:                      # PyTorch model (locally trained)
                import torch
                t = torch.from_numpy(inp)
                with torch.no_grad():
                    out = model(t)[0]
                probs = self._softmax(out.numpy())
            top = int(np.argmax(probs))
            return self._label_map.get(top, ''), float(probs[top])
        except Exception as e:
            log.debug(f'WARP: ML classify error: {e}')
            return '', 0.0

    def _enroll_wiki_art(self, models_dir: Path) -> None:
        """Give the embedder a reference for every icon the gallery lacks.

        The gallery ships pre-built from confirmed crops, so an item nobody
        has ever confirmed is unreachable — see ART_SIM_OFFSET for what the
        embedder does instead of staying quiet. The wiki PNG for that item is
        already on disk, and the embedder is just a function, so a usable
        reference costs one forward pass and no training at all.

        Vectors are cached next to the gallery: the first run pays ~7 ms per
        icon, later runs only embed art that has appeared since. The cache is
        keyed by the embedder's own file hash, so a new model invalidates it
        rather than mixing vectors from two different embedding spaces —
        which would be silent and would poison every comparison.

        Never raises. Failing to enrol leaves the shipped gallery exactly as
        it was, which is the previous behaviour.
        """
        import cv2

        try:
            images_dir = self._get_images_dir()
            if images_dir is None or not images_dir.exists():
                return

            # Fold era-variant art onto the item it depicts, exactly as
            # `_build_index` does. 34 PNGs are named 'X (23c)' for the same
            # item as 'X'; enrolled under the raw stem they would be labels
            # cargo has never heard of — unreachable as slot candidates, and
            # able to win the unrestricted top-1 with a name nothing
            # downstream can resolve.
            try:
                from warp.data.cargo import canonical_names
                known_names = canonical_names()
            except Exception as exc:
                log.warning(f'WARP: art enrolment has no cargo names ({exc!r}) — '
                            'era-variant art will not be folded')
                known_names = set()

            known = {self._label_map.get(int(l), '') for l in self._gallery_lbl}
            png_by_name: dict[str, Path] = {}
            for png in images_dir.glob('*.png'):
                name = _base_item_name(unquote_plus(png.stem), known_names)
                png_by_name.setdefault(name, png)
            missing = sorted(n for n in png_by_name if n and n not in known)
            if not missing:
                return

            # Key the cache on the embedder's *contents*. Size will not do:
            # the file is 17.6 MB because of the architecture (360 tensors,
            # 4.4 M parameters), so a retrained model of the same shape is
            # byte-for-byte the same length with entirely different weights.
            # Reusing art vectors across that boundary is meaningless — they
            # would live in the previous embedding space — and nothing would
            # report an error, only worse matches. `model_updater` copies new
            # models in without deleting anything, so a stale cache does
            # survive an update and has to be detected here.
            emb_path = models_dir / 'icon_embedder.pt'
            import hashlib
            digest = hashlib.sha256(emb_path.read_bytes()).hexdigest()[:16]
            fingerprint = f'{digest}-{self._gallery_emb.shape[1]}'
            cache_path = models_dir / 'art_index.npz'
            cached: dict[str, np.ndarray] = {}
            if cache_path.exists():
                try:
                    blob = np.load(str(cache_path), allow_pickle=False)
                    if str(blob['fingerprint']) == fingerprint:
                        cached = dict(zip((str(n) for n in blob['names']),
                                          blob['embeddings'].astype(np.float32)))
                except Exception as exc:
                    log.debug(f'WARP: art cache unreadable ({exc}) — rebuilding')

            todo = [n for n in missing if n not in cached]
            if todo:
                import time
                t0 = time.time()
                for name in todo:
                    img = cv2.imread(str(png_by_name[name]))
                    if img is None:
                        continue
                    crop64 = cv2.resize(img, (MATCH_SIZE, MATCH_SIZE),
                                        interpolation=cv2.INTER_AREA)
                    vec = self._embed_crop(crop64)
                    if vec is not None:
                        cached[name] = vec
                log.info(f'WARP: embedded {len(todo)} wiki icons for the gallery '
                         f'in {time.time() - t0:.1f}s')
                try:
                    names = sorted(cached)
                    np.savez_compressed(
                        str(cache_path), fingerprint=fingerprint,
                        names=np.array(names),
                        embeddings=np.stack([cached[n] for n in names]))
                except Exception as exc:
                    log.warning(f'WARP: could not cache art vectors: {exc}')

            usable = [n for n in missing if n in cached]
            if not usable:
                return

            next_id = (max(self._label_map) + 1) if self._label_map else 0
            new_ids = []
            for name in usable:
                self._label_map[next_id] = name
                new_ids.append(next_id)
                next_id += 1

            self._gallery_emb = np.concatenate(
                [self._gallery_emb, np.stack([cached[n] for n in usable])])
            self._gallery_lbl = np.concatenate(
                [self._gallery_lbl, np.array(new_ids, dtype=np.int32)])
            self._gallery_is_art = np.concatenate(
                [self._gallery_is_art, np.ones(len(usable), dtype=bool)])
            # Some icon filenames are outside cargo's vocabulary: skill-tree
            # nodes, traits carrying an environment suffix the cargo row does
            # not ('Adaptive Defense (ground)'), and a few items cargo simply
            # lacks. They can never satisfy `candidate_names`, so they are
            # inert on the slot-driven path. The template index already
            # carries every one of them, so this mirrors existing behaviour
            # rather than adding a new failure — but it is counted here so it
            # stays visible instead of being discovered later.
            outside = len([n for n in usable if known_names and n not in known_names])
            summary = (f'gallery + {len(usable)} icons enrolled from wiki art '
                       f'({len(known)} from confirmed crops'
                       + (f', {outside} outside cargo' if outside else '') + ')')
            log.info(f'WARP: {summary}')
            # Also to the system channel. `log` resolves whichever detection
            # channel is active, so this line lands in warp_detection.log when
            # WARP ran and warp_detection_core.log when WARP CORE did — and
            # what the gallery contained is a fact about the *model*, not
            # about one recognition. Mirroring it next to the model updates
            # gives one continuous record of which gallery produced which
            # results, which is what comparing runs over weeks needs.
            syslog.info(f'WARP: {summary}')
        except Exception as exc:
            log.warning(f'WARP: wiki-art enrolment skipped ({exc})')

    def _embed_crop(self, crop64: 'np.ndarray'):
        """Run one 64x64 BGR crop through the embedder. None on any failure.

        Shares the exact preprocessing of `_classify_ml_embed`; the two must
        not drift, or enrolled art would land in a different part of the
        space than the queries it is meant to answer.
        """
        import cv2
        try:
            import torch
            rgb = cv2.cvtColor(cv2.resize(crop64, (224, 224)), cv2.COLOR_BGR2RGB)
            inp = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            inp = (inp - mean) / std
            inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), axis=0)
            with torch.no_grad():
                return self._ml_session(torch.from_numpy(inp)).numpy()[0]
        except Exception as exc:
            log.debug(f'WARP: art embed failed: {exc}')
            return None

    def _classify_ml_embed(
        self,
        crop64: np.ndarray,
        candidate_names: set[str] | None = None,
    ) -> tuple[str, float]:
        """Embed a crop and return the nearest-neighbour label from the gallery.

        Confidence is the cosine similarity to the nearest gallery embedding,
        clamped to [0, 1] — same range as the softmax classifier's confidence,
        so the rest of the fallback chain treats both models interchangeably.

        candidate_names: when provided, k-NN is restricted to gallery entries
        whose label is in the set (plus virtual classes __empty__/__inactive__,
        which the upstream guard still gets to suppress). Without this filter,
        absolute top-1 may be a wrong-slot class (e.g. console on a weapon
        slot); upstream then drops it as not-in-candidates, leaving the slot
        with src=none even when a valid weapon was the runner-up.
        """
        import cv2
        if self._gallery_emb is None or self._gallery_lbl is None:
            return '', 0.0
        rgb = cv2.cvtColor(cv2.resize(crop64, (224, 224)), cv2.COLOR_BGR2RGB)
        inp = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (inp - mean) / std
        inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), axis=0)
        try:
            import torch
            t = torch.from_numpy(inp)
            with torch.no_grad():
                emb = self._ml_session(t).numpy()[0]    # (D,) already L2-normed
            raw_sims = self._gallery_emb @ emb         # (N,) cosine similarity
            # The offset compensates the domain gap so art entries can win the
            # *contest* against confirmed crops — see ART_SIM_OFFSET. It must
            # not travel into the reported confidence: clamped to [0, 1], an
            # art row at raw 0.92 would report 1.00, claiming the certainty of
            # a pixel-perfect confirmed crop on weaker evidence, and feeding
            # ML_PRIMARY_THRESHOLD, VIRTUAL_OVERRIDE_CONF and WARP CORE's
            # auto-accept. That is reachable, not theoretical: 30.8% of art
            # rows sit within 0.95 of another art row.
            #
            # So rank on the adjusted score, report the raw one.
            sims = raw_sims
            if self._gallery_is_art is not None and self._gallery_is_art.any():
                sims = raw_sims + ART_SIM_OFFSET * self._gallery_is_art
            if candidate_names is not None:
                # Split gallery into real-candidate and virtual partitions, so
                # the caller can compare best-real vs best-virtual similarity
                # (embedder-based virtual suppression — more reliable than the
                # bright/rich heuristic on edge-clipped crops).
                labels = np.array(
                    [self._label_map.get(int(lbl), '') for lbl in self._gallery_lbl]
                )
                real_mask    = np.array(
                    [n in candidate_names and n not in VIRTUAL_LABELS for n in labels],
                    dtype=bool,
                )
                virtual_mask = np.array([n in VIRTUAL_LABELS for n in labels], dtype=bool)
                # Raw scores on both sides. These two feed a different
                # question from the one the offset answers: not "which item is
                # this?" but "is there an item here at all?". Virtual entries
                # (__empty__/__inactive__) come from real crops and have no
                # art counterpart, so boosting only the real side would tilt
                # an empty slot towards being called occupied — a false item,
                # not merely a wrong name.
                if real_mask.any():
                    self._last_embed_sim_real = float(raw_sims[real_mask].max())
                if virtual_mask.any():
                    self._last_embed_sim_virtual = float(raw_sims[virtual_mask].max())
                allowed_mask = real_mask | virtual_mask
                if allowed_mask.any():
                    masked = np.where(allowed_mask, sims, -np.inf)
                    top = int(np.argmax(masked))
                else:
                    top = int(np.argmax(sims))
            else:
                top = int(np.argmax(sims))
            best_lbl = int(self._gallery_lbl[top])
            conf = float(max(0.0, min(1.0, raw_sims[top])))
            # Recorded so the match summary and recog_runs.jsonl can separate
            # answers backed by a confirmed crop from ones backed only by wiki
            # art — the whole point of enrolling is to be able to measure it.
            self._last_embed_was_art = bool(
                self._gallery_is_art is not None and self._gallery_is_art[top])
            return self._label_map.get(best_lbl, ''), conf
        except Exception as e:
            log.debug(f'WARP: ML embed error: {e}')
            return '', 0.0

    def _get_ml_session(self):
        if self._ml_disabled:
            return None
        if self._ml_session:
            return self._ml_session

        models_dir = userdata.models_dir()

        # Priority 0: metric-learning embedder (icon_embedder.pt + gallery index)
        # Uses embedder_label_map.json so its class space stays disjoint from
        # the softmax classifier's label_map.json (different class counts).
        emb_path     = models_dir / 'icon_embedder.pt'
        gallery_path = models_dir / 'embedding_index.npz'
        emb_label    = models_dir / 'embedder_label_map.json'
        if emb_path.exists() and gallery_path.exists() and emb_label.exists():
            try:
                import torch
                import torch.nn as nn
                import torch.nn.functional as F
                from torchvision.models import efficientnet_b0
                with open(emb_label, encoding='utf-8') as f:
                    raw = json.load(f)
                self._label_map = {int(k): v for k, v in raw.items()}
                # Match admin_train_metric.py architecture: backbone with no classifier,
                # plus a Linear projection to EMBED_DIM with L2-normalize on output.
                gallery = np.load(str(gallery_path))
                embed_dim = int(gallery['embeddings'].shape[1])
                backbone = efficientnet_b0(weights=None)
                in_features = backbone.classifier[1].in_features
                backbone.classifier = nn.Identity()

                class Embedder(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.backbone = backbone
                        self.proj = nn.Linear(in_features, embed_dim)
                    def forward(self, x):
                        f = self.backbone(x)
                        return F.normalize(self.proj(f), dim=1)

                model = Embedder()
                model.load_state_dict(torch.load(str(emb_path), map_location='cpu',
                                                  weights_only=True))
                model.eval()
                self._ml_session = model
                self._ml_kind = 'embedder'
                self._gallery_emb = gallery['embeddings'].astype(np.float32)
                self._gallery_lbl = gallery['labels'].astype(np.int32)
                self._gallery_is_art = np.zeros(len(self._gallery_lbl), dtype=bool)
                self._enroll_wiki_art(models_dir)
                log.info(f'WARP: metric-learning embedder loaded '
                         f'({len(self._label_map)} classes, '
                         f'gallery={len(self._gallery_emb)}, dim={embed_dim})')
                return self._ml_session
            except Exception as e:
                log.warning(f'WARP: embedder load failed: {e} — falling back to classifier')

        # Priority 1: locally trained PyTorch model (.pt)
        pt_path    = models_dir / 'icon_classifier.pt'
        label_path = models_dir / 'label_map.json'
        if pt_path.exists() and label_path.exists():
            try:
                import torch
                from torchvision.models import efficientnet_b0
                import torch.nn as nn
                with open(label_path, encoding='utf-8') as f:
                    raw = json.load(f)
                self._label_map = {int(k): v for k, v in raw.items()}
                n_classes = len(self._label_map)
                model = efficientnet_b0(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, n_classes)
                model.load_state_dict(torch.load(str(pt_path), map_location='cpu',
                                                  weights_only=True))
                model.eval()
                self._ml_session = model
                self._ml_kind = 'classifier'
                log.info(f'WARP: local PyTorch icon classifier loaded ({n_classes} classes)')
                return self._ml_session
            except Exception as e:
                log.warning(f'WARP: local .pt load failed: {e}')

        # Priority 2: ONNX model from HuggingFace Hub
        model_path = models_dir / HF_MODEL_FILENAME
        hf_label   = models_dir / HF_LABELS_FILE
        flag_path  = models_dir / HF_UNAVAILABLE_FILE

        if model_path.exists() and hf_label.exists():
            try:
                import onnxruntime as ort
                self._ml_session = ort.InferenceSession(str(model_path))
                with open(hf_label, encoding='utf-8') as f:
                    raw = json.load(f)
                    self._label_map = {int(k): v for k, v in raw.items()}
                self._ml_kind = 'classifier'
                log.info('WARP: HuggingFace ONNX icon classifier loaded')
                return self._ml_session
            except Exception as e:
                log.warning(f'WARP: HF ONNX load failed: {e}')
                self._ml_disabled = True
                return None

        # Check sentinel
        if flag_path.exists():
            import time
            age_h = (time.time() - flag_path.stat().st_mtime) / 3600
            if age_h < HF_RETRY_HOURS:
                self._ml_disabled = True
                return None
            flag_path.unlink(missing_ok=True)

        # Attempt HuggingFace download
        if not self._check_repo_exists():
            models_dir.mkdir(parents=True, exist_ok=True)
            flag_path.touch()
            self._ml_disabled = True
            return None

        if not self._download_model(model_path, hf_label):
            models_dir.mkdir(parents=True, exist_ok=True)
            flag_path.touch()
            self._ml_disabled = True
            return None

        try:
            import onnxruntime as ort
            self._ml_session = ort.InferenceSession(str(model_path))
            with open(hf_label, encoding='utf-8') as f:
                raw = json.load(f)
                self._label_map = {int(k): v for k, v in raw.items()}
            log.info('WARP: HuggingFace ONNX icon classifier loaded')
            return self._ml_session
        except Exception as e:
            log.warning(f'WARP: HF ONNX load failed: {e}')
            self._ml_disabled = True
            return None

    @staticmethod
    def _crop_hash(crop_bgr: 'np.ndarray') -> str:
        """Stable content hash for dedup / remove_session_example lookup."""
        import hashlib
        return hashlib.sha1(crop_bgr.tobytes()).hexdigest()

    @staticmethod
    def _template_is_degenerate(crop_bgr: 'np.ndarray') -> bool:
        """True for a crop of one flat colour, which cannot serve as a template.

        `TM_CCOEFF_NORMED` divides by the template's standard deviation. For a
        constant template that is 0/0, and OpenCV's guard resolves it to
        exactly 1.00 — against *any* query, colourful or not. Two such crops
        (pure black, labelled `__empty__`) reached the community pool, and
        every real icon in every screenshot was therefore also offered
        `__empty__` at 0.80 + 0.20·histogram ≈ 0.80–0.85. Measured over the
        301 rows of recog_runs.jsonl written before this guard: not one
        session score fell between 0 and 0.80, so no genuine match below that
        floor could ever surface, and the anti-virtual guards had to shoot the
        false `__empty__` down slot by slot.

        The reverse case is harmless and stays allowed — a constant *query*
        against a real template scores 0.00, which is the correct verdict.
        """
        import cv2
        return float(cv2.resize(crop_bgr, (MATCH_SIZE, MATCH_SIZE),
                                interpolation=cv2.INTER_AREA).std()) < 1e-6

    @classmethod
    def add_session_example(cls, crop_bgr: 'np.ndarray', name: str,
                            origin: str = 'session') -> None:
        """
        Add a user-confirmed crop to the in-memory session index.
        Immediately improves recognition for the rest of this session
        without any retraining.

        `origin` tags the source so reset_ml_session() can keep user / community
        seeds while dropping bulk training-data seeds when WARP takes over.

        Dedup rule: a new (origin='user') entry with the same crop hash REPLACES
        any prior 'user' entry on the same crop — covers the unconfirm/relabel
        case where the user changes their mind about a bbox.
        """
        import cv2
        if crop_bgr is None or crop_bgr.size == 0 or not name.strip():
            return
        if cls._template_is_degenerate(crop_bgr):
            # Info, not debug: this fires twice for the whole community pool,
            # and a crop silently dropped from the seed is exactly the kind of
            # thing that should be visible in the log when a name goes missing.
            log.info(f'WARP: session example rejected — {name!r} ({origin}) is a '
                     f'single flat colour, which would match every query at 1.00')
            return
        if not name.startswith('__') and _real_crop_looks_blank(crop_bgr):
            log.info(f'WARP: session example rejected — {name!r} ({origin}) is a '
                     f'blank cell, and seeding it would teach the matcher that '
                     f'an empty slot is that item')
            return
        crop_hash = cls._crop_hash(crop_bgr)
        if origin == 'user':
            cls._session_examples = [
                e for e in cls._session_examples
                if not (e.get('origin') == 'user'
                        and e.get('crop_hash') == crop_hash)
            ]
        tmpl64 = cv2.resize(crop_bgr, (MATCH_SIZE, MATCH_SIZE),
                             interpolation=cv2.INTER_AREA)
        hist = cls._hist_hsv(tmpl64)
        cls._session_examples.append({
            'name':      name,
            'tmpl64':    tmpl64,
            'hist_hsv':  hist,
            'orig':      crop_bgr,
            'origin':    origin,
            'crop_hash': crop_hash,
        })

    @classmethod
    def remove_session_example(cls, crop_bgr: 'np.ndarray',
                               origin: str = 'user') -> int:
        """Drop session entries matching this crop with the given origin.
        Called by trainer when user unconfirms / relabels a previously
        accepted bbox, so the stale entry stops leaking into WARP matches.
        Returns the number of entries removed.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return 0
        crop_hash = cls._crop_hash(crop_bgr)
        before = len(cls._session_examples)
        cls._session_examples = [
            e for e in cls._session_examples
            if not (e.get('origin') == origin
                    and e.get('crop_hash') == crop_hash)
        ]
        return before - len(cls._session_examples)

    @classmethod
    def seed_from_training_data(cls, training_data_dir) -> int:
        """
        Load all confirmed icon crops from annotations.json as session examples.
        Guarded by _seeded_from_training_data — runs only once per process
        lifetime (reset by reset_ml_session).
        Returns the number of crops loaded (0 if already seeded).
        """
        if cls._seeded_from_training_data:
            return 0

        import json
        import cv2
        from pathlib import Path

        training_data_dir = Path(training_data_dir)
        ann_path = training_data_dir / 'annotations.json'
        if not ann_path.exists():
            return 0
        try:
            data = json.loads(ann_path.read_text(encoding='utf-8'))
        except Exception as e:
            log.warning(f'WARP: seed_from_training_data: {e}')
            return 0

        # These slots have no crop PNGs — skip them
        _TEXT_SLOTS = frozenset({
            'Ship Type', 'Ship Tier',
            'Primary Specialization', 'Secondary Specialization',
        })
        crops_dir = training_data_dir / 'crops'
        count = 0
        skipped_auto = 0
        for _key, val in data.items():
            # New schema: {sha16: {'annotations': [...], ...}}
            # Legacy schema: {filename: [ann_dict, ...]}
            if isinstance(val, dict):
                annotations = val.get('annotations', [])
            elif isinstance(val, list):
                annotations = val
            else:
                continue
            for ann in annotations:
                if ann.get('state') != 'confirmed':
                    continue
                # Skip auto-accepted entries: they're the detector's own
                # guesses, not user-verified ground truth. Seeding from them
                # creates a self-amplification loop (today's high-conf match
                # becomes tomorrow's perfect session-example match).
                if ann.get('auto_confirmed'):
                    skipped_auto += 1
                    continue
                name = ann.get('name', '').strip()
                slot = ann.get('slot', '')
                if not name or slot in _TEXT_SLOTS:
                    continue

                # Primary: explicit crop_name field (newer annotations)
                crop_path = None
                crop_name = ann.get('crop_name', '')
                if crop_name:
                    p = training_data_dir / crop_name
                    if p.exists():
                        crop_path = p

                # Fallback: reconstruct filename from slot + name + ann_id
                # (matches TrainingDataManager._export_crop naming convention)
                if crop_path is None:
                    ann_id = ann.get('ann_id', '')
                    if ann_id:
                        safe_slot = slot.replace(' ', '_').lower()
                        safe_name = name.replace(' ', '_').lower()[:40]
                        fname = f'{safe_slot}__{safe_name}__{ann_id}.png'
                        p = crops_dir / fname
                        if p.exists():
                            crop_path = p

                if crop_path is None:
                    continue
                img = cv2.imread(str(crop_path))
                if img is None:
                    continue
                # Poison guard: virtual label but colourful crop → skip.
                # Prevents self-matching session pixel-perfectly on a real icon
                # that was mislabeled __empty__/__inactive__ by auto-accept.
                # `poison_reviewed=True` means the user already inspected the
                # crop via `scrub_training_data --review` and confirmed the
                # virtual label is correct — trust them and load it.
                if (name in VIRTUAL_LABELS
                        and not ann.get('poison_reviewed')
                        and _virtual_crop_looks_real(img)):
                    log.debug(
                        f'WARP: training-seed POISON skip — '
                        f'{crop_path.name} labeled {name!r} but looks colourful '
                        f'(run `python -m warp.tools.scrub_training_data --review` '
                        f'to clean)'
                    )
                    continue
                cls.add_session_example(img, name, origin='trainer_td')
                count += 1

        cls._seeded_from_training_data = True
        log.info(f'WARP: training data seed: {count} session examples from {len(data)} screenshots '
                 f'(skipped {skipped_auto} auto_confirmed) (path: {training_data_dir})')
        return count

    @classmethod
    def seed_from_community_crops(cls, force: bool = False) -> int:
        """Seed the session-example pool from the HF-mirrored approved truth.

        Reads `data/annotations.jsonl` + `data/crops/<sha>.png` from
        `warp.knowledge.community_crops`, so every install starts with the
        same recognition baseline. Cheap on repeat calls: skips when the
        annotations file mtime is unchanged (so the 5-min SyncCoordinator
        tick doesn't re-load thousands of PNGs needlessly).

        `force=True` bypasses both the boolean guard and the mtime check —
        used by `reset_ml_session()` callers.
        """
        import cv2
        from warp.knowledge.community_crops import (
            community_annotations_file, community_crops_dir,
            mirror_crop_path,
        )

        ann_path  = community_annotations_file()
        crops_dir = community_crops_dir()
        if not ann_path.exists() or not crops_dir.exists():
            cls._seeded_from_community = True
            return 0

        try:
            mtime = ann_path.stat().st_mtime
        except OSError:
            mtime = 0.0

        if not force and cls._seeded_from_community \
                and mtime == cls._seeded_community_mtime:
            return 0

        _TEXT_SLOTS = frozenset({
            'Ship Type', 'Ship Tier',
            'Primary Specialization', 'Secondary Specialization',
        })

        # Last-wins per sha so maintainer label corrections take effect.
        latest: dict[str, dict] = {}
        try:
            with open(ann_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    sha = d.get('crop_sha256')
                    if sha:
                        latest[sha] = d
        except Exception as e:
            syslog.warning(f'CommunitySeed: read failed: {e}')
            cls._seeded_from_community = True
            return 0

        count = 0
        for sha, d in latest.items():
            name = (d.get('name') or '').strip()
            slot = d.get('slot') or ''
            if not name or slot in _TEXT_SLOTS:
                continue
            # Sharded, with the flat path as a fallback for a mirror
            # that has not been through `_shard_local` yet.
            p = mirror_crop_path(f'{sha}.png')
            if not p.exists():
                p = crops_dir / f'{sha}.png'
            if not p.exists():
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            # Poison guard: virtual label but colourful crop → skip.
            if name in VIRTUAL_LABELS and _virtual_crop_looks_real(img):
                syslog.debug(
                    f'CommunitySeed: POISON skip — {sha[:10]} labeled {name!r} '
                    f'but looks colourful'
                )
                continue
            cls.add_session_example(img, name, origin='community')
            count += 1

        cls._seeded_from_community = True
        cls._seeded_community_mtime = mtime
        syslog.info(f'CommunitySeed: {count} session examples '
                    f'from {len(latest)} approved entries ({crops_dir})')
        return count

    @classmethod
    def reset_ml_session(cls, keep_origins: set[str] | None = None):
        """
        Force reload of the ML model on next inference call.
        Called after local training completes, and by WARP's `_get_matcher`
        to clear bulk trainer seeds before each run.

        `keep_origins`: entries whose `origin` is in this set survive the
        reset. Used by WARP path with {'user', 'community'} to preserve
        the live-seed pipeline (user's own confirmed crops + community
        approved truth) while dropping any 'trainer_td' seed that a prior
        WARP CORE session may have loaded from annotations.json.

        `None` (default) is a hard reset — wipes everything. Used by the
        model updater after a fresh model is installed.
        """
        # New SETSIconMatcher() instances will reload fresh from disk.
        # Existing instances keep their old model until garbage-collected.
        # (_shared_* attributes don't exist; instance attrs are _ml_session etc.)
        if keep_origins is None:
            cls._session_examples = []
        else:
            cls._session_examples = [
                e for e in cls._session_examples
                if e.get('origin', 'session') in keep_origins
            ]
        cls._seeded_from_training_data = False
        if keep_origins is None or 'community' not in keep_origins:
            cls._seeded_from_community  = False
            cls._seeded_community_mtime = 0.0
        log.info(f'WARP: ML session reset -- '
                 f'kept_origins={sorted(keep_origins) if keep_origins else "[]"}, '
                 f'pool_size={len(cls._session_examples)}')

    def _check_repo_exists(self) -> bool:
        """
        Do a lightweight HEAD request to check if the HF repo exists.
        Returns False silently on 401/404 or any network error.
        """
        try:
            import urllib.request
            url = f'https://huggingface.co/{HF_REPO_ID}'
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=6) as r:
                return r.status == 200
        except Exception:
            return False

    def _download_model(self, dest: Path, label_path: Path) -> bool:
        try:
            from huggingface_hub import hf_hub_download
            dest.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME,
                            local_dir=str(dest.parent))
            hf_hub_download(repo_id=HF_REPO_ID, filename=HF_LABELS_FILE,
                            local_dir=str(dest.parent))
            return dest.exists()
        except Exception as e:
            log.warning(f'WARP: model download failed: {e}')
            return False

    # ── Misc helpers ────────────────────────────────────────────────────────────

    def _find_sets_root(self) -> Path:
        p = Path(__file__).resolve()
        for _ in range(6):
            if (p / 'pyproject.toml').exists():
                return p
            p = p.parent
        return Path('.')

    def _bgr_to_qimage(self, img_bgr: np.ndarray | None):
        if img_bgr is None:
            return None
        try:
            import cv2
            from PySide6.QtGui import QImage
            rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            return QImage(rgb.data, w, h, 3 * w,
                          QImage.Format.Format_RGB888).copy()
        except Exception:
            return None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
