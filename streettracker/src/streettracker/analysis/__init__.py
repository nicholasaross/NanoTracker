"""Off-device post-processing: ALPR, recolor, make/model classification,
cross-session re-id.

Each module is standalone — it reads a closed session directory, runs a
per-record computation, and rewrites the JSONL + data.json + summary.html.
Pattern documented in NanoTracker's `scripts/recolor_session.py`.
"""
