"""
Global configuration for the WARP module.
Centralizes hardcoded timeouts, thresholds, and dimension logic.
"""

from __future__ import annotations

# ── Network & Sync Timeouts ───────────────────────────────────────────────────

# warp/knowledge/sync_client.py
SYNC_CONNECT_TIMEOUT    = 5      # seconds
SYNC_READ_TIMEOUT       = 15     # seconds — knowledge download (has cache fallback)
SYNC_CONTRIBUTE_TIMEOUT = 60     # seconds — longer: covers Render cold-start (~50 s)

# warp/data/asset_sync.py
ASSET_RETRY_DELAY_S     = 3      # seconds
ASSET_STALL_TIMEOUT_S   = 10     # seconds

# warp/trainer/model_updater.py
MODEL_CONNECT_TIMEOUT   = 5      # seconds
MODEL_READ_TIMEOUT      = 60     # seconds
MODEL_RETRY_DELAYS_MIN  = (1, 5, 15, 60)  # backoff schedule on network failure


# ── Recognition Thresholds ────────────────────────────────────────────────────

# warp/warp_importer.py
IMPORTER_TEMPLATE_CONF_THRESHOLD     = 0.72
IMPORTER_CONFIDENT_VIRTUAL_THRESHOLD = 0.70
IMPORTER_MIN_ACCEPT_CONF             = 0.35
IMPORTER_RECALIBRATION_MIN_CONF      = 0.85

# warp/recognition/layout_detector.py
OCR_CONF_THRESHOLD = 0.40
LABEL_FUZZY_CUTOFF = 0.68


# ── UI & Geometry Offsets ─────────────────────────────────────────────────────

# warp/build_writer.py
BOFF_Y_THRESHOLD_PX = 30
BOFF_X_THRESHOLD_PX = 50
