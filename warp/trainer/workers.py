# warp/trainer/workers.py
# WARP CORE background QThread workers — screen-type detection, OCR,
# icon matching, and full recognition pipeline. Extracted from
# trainer_window.py during the Phase-0 refactor.

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from warp import userdata
from warp.trainer.constants import SCREEN_TYPE_LABELS, SCREEN_TYPE_ICONS
from warp.trainer._ui_utils import _log_match_summary


class ScreenTypeDetectorWorker(QThread):
    # progress: (idx, total, filename, stype, conf)
    progress = Signal(int, int, str, str, float)
    # finished: {filename: (stype, conf)}  — conf is raw ML confidence (0.0 if no model)
    finished = Signal(dict)
    def __init__(self, paths: list, models_dir=None, confirmed_types: dict | None = None, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._models_dir = models_dir
        self._confirmed_types = confirmed_types or {}   # {Path: stype} — for k-NN pre-seed
        self._sets_root = self._find_sets_root() # Need sets_root for stats file

    def _find_sets_root(self) -> Path:
        p = Path(__file__).resolve()
        for _ in range(8):
            if (p / 'pyproject.toml').exists():
                return p
            p = p.parent
        return Path('.')

    def run(self):
        from warp.debug import log as _slog, use_detection_channel
        with use_detection_channel('detection_core'):
            self._run_inner(_slog)

    def _run_inner(self, _slog):
        results: dict[str, tuple] = {}   # filename → (stype, conf)
        total = len(self._paths)
        classifier = None

        # Statistics collection
        stats_total_per_type = {st: 0 for st in SCREEN_TYPE_LABELS.keys()}
        stats_correct_per_type = {st: 0 for st in SCREEN_TYPE_LABELS.keys()}
        overall_correct = 0

        if self._models_dir is not None:
            try:
                from warp.recognition.screen_classifier import ScreenTypeClassifier, CONF_THRESHOLD
                classifier = ScreenTypeClassifier(self._models_dir)
                _slog.info(f'ScreenTypeDetector: classifier loaded from {self._models_dir}')
            except Exception as e:
                _slog.warning(f'ScreenTypeDetector: classifier unavailable — {e}')
                _slog.info('ScreenTypeDetector: will use UNKNOWN for all (no model trained yet)')
        else:
            _slog.warning('ScreenTypeDetector: no models_dir — all results will be UNKNOWN')
        import cv2

        # Pre-seed k-NN with all user-confirmed types before running detection.
        # Clear first to prevent accumulation of stale examples across runs.
        if self._confirmed_types and classifier is not None:
            from warp.recognition.screen_classifier import ScreenTypeClassifier
            ScreenTypeClassifier.clear_session()
            seeded = 0
            for cpath, cstype in self._confirmed_types.items():
                img = cv2.imread(str(cpath))
                if img is not None:
                    ScreenTypeClassifier.add_session_example(img, cstype)
                    seeded += 1
            _slog.info(f'ScreenTypeDetector: k-NN pre-seeded with {seeded} confirmed examples')

        _slog.info(f'ScreenTypeDetector: starting — {total} screenshot(s)')
        for idx, path in enumerate(self._paths):
            if self.isInterruptionRequested():
                _slog.info('ScreenTypeDetector: interrupted')
                break
            stype = 'UNKNOWN'
            conf  = 0.0
            is_correct = False
            try:
                img = cv2.imread(str(path))
                if img is None:
                    _slog.warning(f'ScreenTypeDetector: cannot read {path.name}')
                elif classifier is None:
                    pass
                else:
                    ml_stype, ml_conf = classifier.classify(img)
                    conf = ml_conf
                    if ml_stype and ml_conf >= CONF_THRESHOLD:
                        stype = ml_stype
                        is_correct = True
            except Exception as e:
                _slog.warning(f'ScreenTypeDetector: [{idx+1}/{total}] {path.name} → error: {e}')

            results[path.name] = (stype, conf)
            stats_total_per_type[stype] = stats_total_per_type.get(stype, 0) + 1
            if is_correct:
                stats_correct_per_type[stype] = stats_correct_per_type.get(stype, 0) + 1
                overall_correct += 1

            self.progress.emit(idx + 1, total, path.name, stype, conf)

        self._log_screen_type_stats(total, overall_correct, stats_total_per_type, stats_correct_per_type)
        _slog.info(f'ScreenTypeDetector: done — {len(results)} processed')
        self.finished.emit(results)

    def _log_screen_type_stats(self, total_files: int, overall_correct: int,
                               stats_total_per_type: dict, stats_correct_per_type: dict):
        import json
        import datetime
        from warp.debug import log as _slog

        stats_path = userdata.screen_type_stats_file()

        # Load historical data
        try:
            history: list[dict] = json.loads(stats_path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            history = []

        current_session_stats = {
            "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
            "total_files": total_files,
            "overall_accuracy": round(overall_correct / total_files, 2) if total_files > 0 else 0.0,
            "types": {}
        }

        summary_lines = []
        summary_lines.append(f'ScreenTypeDetector: Summary for {total_files} files:')
        summary_lines.append(f'  Overall Accuracy: {current_session_stats["overall_accuracy"]:.0%}')

        for stype_key in sorted(SCREEN_TYPE_LABELS.keys()):
            if stype_key == 'UNKNOWN': continue # Skip UNKNOWN for detailed stats
            total_for_type = stats_total_per_type.get(stype_key, 0)
            correct_for_type = stats_correct_per_type.get(stype_key, 0)

            accuracy = round(correct_for_type / total_for_type, 2) if total_for_type > 0 else 0.0
            current_session_stats["types"][stype_key] = {
                "total": total_for_type,
                "correct": correct_for_type,
                "accuracy": accuracy
            }

            # Calculate trend
            trend_icon = '→'
            prev_accuracies = [
                s["types"][stype_key]["accuracy"] for s in history
                if stype_key in s["types"] and s["types"][stype_key]["total"] > 0
            ]
            if prev_accuracies:
                avg_prev_accuracy = sum(prev_accuracies) / len(prev_accuracies)
                if accuracy > avg_prev_accuracy + 0.02: # 2% improvement threshold
                    trend_icon = '↑'
                elif accuracy < avg_prev_accuracy - 0.02: # 2% degradation threshold
                    trend_icon = '↓'

            if total_for_type > 0:
                summary_lines.append(
                    f'  {SCREEN_TYPE_ICONS.get(stype_key, "?")} {SCREEN_TYPE_LABELS[stype_key]:<20}: '
                    f'{correct_for_type}/{total_for_type} ({accuracy:.0%}) {trend_icon}'
                )

        # Add UNKNOWN stats separately
        unknown_total = stats_total_per_type.get('UNKNOWN', 0)
        if unknown_total > 0:
            summary_lines.append(f'  {SCREEN_TYPE_ICONS.get("UNKNOWN", "?")} UNKNOWN             : {unknown_total} files')
            current_session_stats["types"]["UNKNOWN"] = {"total": unknown_total, "correct": 0, "accuracy": 0.0}


        # Log the summary
        for line in summary_lines:
            _slog.info(line)

        # Save current session stats to history (keep last 50 sessions)
        history.append(current_session_stats)
        history = history[-50:] # Keep only the last 50 sessions
        try:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
        except Exception as e:
            _slog.warning(f'ScreenTypeDetector: Failed to save stats history: {e}')


class OCRWorker(QThread):
    finished = Signal(int, str, float, object, str)  # row, text, conf, crop_bgr, ocr_raw

    def __init__(self, row: int, crop_bgr, slot: str, valid_tiers: list, valid_types: list, parent=None):
        super().__init__(parent)
        self.row = row
        self.crop_bgr = crop_bgr
        self.slot = slot
        self.valid_tiers = valid_tiers
        self.valid_types = valid_types

    def run(self):
        # Delegates to TextExtractor.refine_single_crop — single shared mechanism
        # with WARP's autodetection path (warp_importer.WarpImporter._process_image
        # calls refine_ship_info, which uses the same refine_single_crop).
        from warp.debug import log as _slog, use_detection_channel
        with use_detection_channel('detection_core'):
            try:
                from warp.recognition.text_extractor import TextExtractor
                extractor = TextExtractor()
                text, conf, ocr_raw = extractor.refine_single_crop(
                    self.crop_bgr, self.slot, self.valid_tiers, self.valid_types)
                _slog.info(f"ocr_worker slot={self.slot!r} raw={ocr_raw!r} → final={text!r} conf={conf:.2f}")
                self.finished.emit(self.row, text, conf, self.crop_bgr, ocr_raw)
            except Exception as e:
                _slog.warning(f'OCRWorker failed: {e}')
                self.finished.emit(self.row, '', 0.0, self.crop_bgr, '')


class MatchWorker(QThread):
    """Background icon matching (two-pass) to keep UI responsive during bbox draw."""
    finished = Signal(str, float, object, object, tuple)  # name, conf, thumb, crop_bgr, bbox

    def __init__(self, crop_bgr, bbox: tuple, candidate_names, sets_app, parent=None):
        super().__init__(parent)
        self._crop = crop_bgr
        self._bbox = bbox
        self._candidates = candidate_names
        self._sets = sets_app

    def run(self):
        from warp.debug import use_detection_channel
        with use_detection_channel('detection_core'):
            name, conf, thumb = '', 0.0, None
            try:
                from warp.recognition.icon_matcher import SETSIconMatcher
                from warp.debug import log as _slog
                # Seed confirmed crops as session examples (guarded — runs at most once)
                matcher = SETSIconMatcher(self._sets)
                SETSIconMatcher.seed_from_training_data(userdata.training_data_dir())
                SETSIconMatcher.seed_from_community_crops()
                name, conf, thumb, _ = matcher.match(
                    self._crop, candidate_names=self._candidates)
                _slog.info(f'match_worker → name={name!r} conf={conf:.2f} '
                           f'(pool={len(self._candidates) if self._candidates else "all"})')
                if conf < 0.40:
                    _slog.info(f'match_worker: conf {conf:.2f} < 0.40 — treating as unmatched')
                    name, conf, thumb = '', 0.0, None
            except Exception as e:
                from warp.debug import log as _slog
                _slog.warning(f'MatchWorker failed: {e}')
            self.finished.emit(name, conf, thumb, self._crop, self._bbox)


class RecognitionWorker(QThread):
    finished = Signal(list)
    error    = Signal(str)
    # (pct 0-100, stage label) — matches the importer's per-stage callback
    # in WARP so both tools surface the same progress breakdown.
    progress = Signal(int, str)

    def __init__(self, path, stype: str, sets_app, parent=None, skip_bboxes: list | None = None):
        super().__init__(parent)
        self._path = path
        self._stype = stype
        self._sets_app = sets_app
        self._skip_bboxes = list(skip_bboxes) if skip_bboxes else []
        # EQ panel geometry captured during detection; consumed by _on_recognition_done
        self.eq_geom = None

    def _stage_cb(self, pct: int, label: str) -> None:
        if self.isInterruptionRequested():
            raise InterruptedError('cancelled')
        self.progress.emit(pct, label)

    def run(self):
        from warp.debug import use_detection_channel
        with use_detection_channel('detection_core'):
            self._run_inner()

    def _run_inner(self):
        import cv2
        from warp.debug import log as _slog
        from warp.warp_importer import WarpImporter

        # Load image once — reused for inference, recognition pipeline, and crop extraction
        img = cv2.imread(str(self._path))
        if img is None:
            _slog.warning(f'RecognitionWorker: cannot read image {self._path}')
            self.finished.emit([])
            return
        _slog.info(f'RecognitionWorker: image loaded {img.shape[1]}x{img.shape[0]} px')

        # Map trainer screen type → WarpImporter build_type.
        # Single source of truth lives in warp_importer.SCREEN_TYPE_TO_BUILD_TYPE
        # so WARP and WARP CORE share the same mapping.
        from warp.warp_importer import SCREEN_TYPE_TO_BUILD_TYPE
        importer_type = SCREEN_TYPE_TO_BUILD_TYPE.get(self._stype)   # None → UNKNOWN

        # UNKNOWN screens default to SPACE; TRAITS screens stay as SPACE_TRAITS
        if importer_type is None:
            importer_type = 'SPACE'
            _slog.info(f'RecognitionWorker: UNKNOWN screen — defaulting to SPACE')

        _slog.info(f'RecognitionWorker: start {self._path.name} stype={self._stype} → importer={importer_type}')

        try:
            importer = WarpImporter(
                sets_app=self._sets_app, build_type=importer_type,
                from_trainer=True, progress_callback=self._stage_cb,
            )
            result = importer._process_image(img, str(self._path),
                                              skip_bboxes=self._skip_bboxes or None)
            _slog.info(f'RecognitionWorker: pipeline done — {len(result.items)} items found')
            # Capture EQ geometry from the layout detector's per-image cache so
            # the canvas can overlay the 6×N grid that detection actually used.
            try:
                self.eq_geom = importer._get_layout()._eq_geom_cache.get(id(img))
            except Exception:
                self.eq_geom = None
            for e in result.errors:
                _slog.warning(f'RecognitionWorker: pipeline error: {e}')

            # Cross-check layout vs content
            cross_check_failed_items = set()
            try:
                xcheck = WarpImporter(sets_app=self._sets_app)
                for item in result.items:
                    if not xcheck._item_valid_for_slot(item.name, item.slot):
                        _slog.info(f'RecognitionWorker: cross-check warning: {item.name!r} invalid for {item.slot!r}')
                        cross_check_failed_items.add((item.slot, item.name))
            except:
                pass
        except InterruptedError:
            _slog.info('RecognitionWorker: cancelled by user')
            self.error.emit('Cancelled')
            return
        except Exception as e:
            _slog.warning(f'RecognitionWorker: exception — {e}')
            self.error.emit(str(e))
            return

        items = []
        for ri in result.items:
            crop_bgr = None
            if ri.bbox is not None:
                try:
                    x, y, w, h = ri.bbox
                    crop_bgr = img[y:y+h, x:x+w].copy()
                except:
                    pass
            _slog.info(f'RecognitionWorker:   slot={ri.slot!r:25} name={ri.name!r:40} conf={ri.confidence:.2f} bbox={ri.bbox}')
            cross_check = (ri.slot, ri.name) in cross_check_failed_items
            items.append({'name': ri.name, 'slot': ri.slot, 'conf': ri.confidence, 'bbox': ri.bbox,
                          'state': 'pending', 'thumb': ri.thumbnail, 'crop_bgr': crop_bgr,
                          'orig_name': ri.name, 'ship_name': result.ship_name,
                          'cross_check_failed': cross_check,
                          # Carry the seat_key set by `_remap_boff_seat_slots`
                          # so the trainer review tree groups BOFF abilities by
                          # physical seat (Boff Tactical #1 / #2, …) the same
                          # way WARP Results does. Without it CORE's dicts saw
                          # an empty seat_key and `group_items_by_seat` fell
                          # back to slot-only grouping, collapsing all seats
                          # of one profession into a single parent row.
                          'seat_key': getattr(ri, 'seat_key', '') or '',
                          # slot_index = position within the slot (left-to-right
                          # in BOFF seats, top-to-bottom in trait columns). Used
                          # by `order_items_for_display` to sort group children
                          # in detection order instead of falling back to name.
                          'slot_index': getattr(ri, 'slot_index', 0) or 0,
                          'src': getattr(ri, 'src', '')})
        # Summary table: per-stage scores + Δ vs previous run for this image.
        try:
            _log_match_summary(self._path.name, getattr(importer, 'match_log', []))
        except Exception as e:
            _slog.debug(f'RecognitionWorker: summary table failed: {e}')

        _slog.info(f'RecognitionWorker: emitting {len(items)} items')
        self.finished.emit(items)
