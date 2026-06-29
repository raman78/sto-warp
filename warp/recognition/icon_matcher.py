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
# rich cannot be a real virtual. Thresholds match warp.tools.scrub_training_data
# (real-virtual p90 = 2.7% bright / 6.8% rich → 0.07 leaves wide margin).
VIRTUAL_SEED_BRIGHT_RATIO   = 0.07
VIRTUAL_SEED_RICH_RATIO     = 0.07
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
# Cross-validation: when two or more sources (session/template/ml) return the
# same real-icon name, boost each agreeing candidate by this amount. Makes
# agreement break ties between sources whose absolute scales differ (cosine
# similarity vs template correlation).
SOURCE_AGREEMENT_BONUS = 0.05
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
        self._ml_session  = None
        self._ml_disabled = False      # True after first failed download attempt
        self._label_map: dict[int, str] = {}
        # Metric-learning path: when icon_embedder.pt is present, _ml_session is
        # the embedder model and _gallery_* hold the k-NN search index. When
        # _ml_kind=='classifier' (legacy softmax), _gallery_* stay None.
        self._ml_kind: str = ''        # 'embedder' | 'classifier' | ''
        self._gallery_emb = None       # np.ndarray (N, D) float32, L2-normed
        self._gallery_lbl = None       # np.ndarray (N,) int32 — indices into _label_map
        # Diagnostic: source of the most recent match() decision.
        # Values: 'ml' (embedder/classifier), 'template' (wiki PNG histogram),
        # 'session' (confirmed training crop), 'knowledge' (pHash override),
        # 'none' (no signal above threshold), '' (no match attempted).
        # Read by warp_importer to expose match source in autodetect logs.
        self._last_match_src: str = ''
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
            self._last_match_origin = ''
            self._last_stage_scores = {'embed': 0.0, 'soft': 0.0,
                                       'session': 0.0, 'template': 0.0,
                                       'knowledge': 0.0}
            return '', 0.0, None, False

        import cv2
        self._last_match_src = ''
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
        # matches stop being noise.
        auto_name  = ''
        auto_score = 0.0
        auto_entry = None
        tm_cutoff = (TEMPLATE_RESTRICTED_THRESHOLD
                     if candidate_names is not None
                     else TEMPLATE_THRESHOLD * 0.7)
        for entry in self._index:
            if candidate_names is not None and entry['name'] not in candidate_names:
                continue
            res      = cv2.matchTemplate(crop64, entry['tmpl64'],
                                         cv2.TM_CCOEFF_NORMED)
            tm_score = float(res.max())
            if tm_score < tm_cutoff:
                continue
            h_score = max(0.0, float(cv2.compareHist(
                q_hist, entry['hist_hsv'], cv2.HISTCMP_CORREL)))
            combined = tm_score * (1.0 - HIST_WEIGHT) + h_score * HIST_WEIGHT
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
        # When 2+ sources agree, boost each agreeing entry by SOURCE_AGREEMENT_BONUS.
        # Lets agreement break ties between sources whose scales differ (cosine
        # similarity vs template correlation). Virtual labels are excluded so a
        # session-vs-ml '__empty__' agreement doesn't override real candidates.
        name_votes: dict[str, int] = {}
        for _src, _name, _score, _entry in candidates:
            if _name and not _name.startswith('__'):
                name_votes[_name] = name_votes.get(_name, 0) + 1
        boosted = [
            (s, n, sc + SOURCE_AGREEMENT_BONUS * max(0, name_votes.get(n, 1) - 1), e)
            for (s, n, sc, e) in candidates
        ]
        src, name, score, entry = max(boosted, key=lambda x: x[2])
        if name_votes.get(name, 1) >= 2:
            agreeing = [s for (s, n, _, _) in candidates if n == name]
            log.debug(
                f"WARP: source-agreement bonus → {name!r} backed by "
                f"{agreeing} (+{SOURCE_AGREEMENT_BONUS:.2f})"
            )
        # Disambiguate ML source by model kind so logs distinguish the
        # ArcFace embedder from the legacy softmax classifier.
        if src == 'ml' and self._ml_kind == 'embedder':
            self._last_match_src = 'embed'
        elif src == 'ml':
            self._last_match_src = 'soft'
        else:
            self._last_match_src = src
        if src == 'session' and entry is not None:
            self._last_match_origin = entry.get('origin', 'session')
        if entry is not None:
            thumb = self._bgr_to_qimage(entry.get('orig'))
        else:
            thumb = self._thumb_for_name(name)
        return name, score, thumb, (src == 'session')

    def _thumb_for_name(self, name: str) -> object:
        """Return a QImage thumbnail for an item name by looking it up in the
        wiki PNG index. Returns None for virtual items (__empty__/__inactive__)
        or when the name is not in the index."""
        if not name or name.startswith('__'):
            return None
        for entry in self._index:
            if entry['name'] == name:
                return self._bgr_to_qimage(entry.get('orig'))
        return None

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

    def classify_ml_batch(
        self,
        thumbnails: list    # list[QImage | None]
    ) -> tuple[list[str], list[float]]:
        """Stage 3 batch classifier (ONNX EfficientNet-B0)."""
        session = self._get_ml_session()
        if session is None:
            return [''] * len(thumbnails), [0.0] * len(thumbnails)

        import cv2
        names, confs = [], []
        for thumb in thumbnails:
            arr = self._qimage_to_bgr(thumb)
            if arr is None:
                names.append(''); confs.append(0.0)
                continue
            name, conf = self._classify_ml(
                cv2.resize(arr, (MATCH_SIZE, MATCH_SIZE))
            )
            names.append(name)
            confs.append(conf)

        return names, confs

    # ── Index building ──────────────────────────────────────────────────────────

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
        count = 0
        for png in images_dir.glob('*.png'):
            name = unquote_plus(png.stem)
            orig = cv2.imread(str(png))
            if orig is None:
                continue

            tmpl64 = cv2.resize(orig, (MATCH_SIZE, MATCH_SIZE),
                                 interpolation=cv2.INTER_AREA)
            self._index.append({
                'name':     name,
                'tmpl64':   tmpl64,
                'hist_hsv': self._hist_hsv(tmpl64),
                'orig':     orig,      # kept for thumbnail generation
            })
            count += 1

        log.info(f'WARP: indexed {count} icons from {images_dir}')

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
            sims = self._gallery_emb @ emb              # (N,) cosine similarity
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
                if real_mask.any():
                    self._last_embed_sim_real = float(sims[real_mask].max())
                if virtual_mask.any():
                    self._last_embed_sim_virtual = float(sims[virtual_mask].max())
                allowed_mask = real_mask | virtual_mask
                if allowed_mask.any():
                    masked = np.where(allowed_mask, sims, -np.inf)
                    top = int(np.argmax(masked))
                else:
                    top = int(np.argmax(sims))
            else:
                top = int(np.argmax(sims))
            best_lbl = int(self._gallery_lbl[top])
            conf = float(max(0.0, min(1.0, sims[top])))
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

    def _qimage_to_bgr(self, qimg) -> np.ndarray | None:
        if qimg is None:
            return None
        try:
            import cv2
            from PySide6.QtGui import QImage
            q   = qimg.convertToFormat(QImage.Format.Format_RGB888)
            w, h = q.width(), q.height()
            arr  = np.frombuffer(q.bits(), dtype=np.uint8).reshape((h, w, 3)).copy()
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
