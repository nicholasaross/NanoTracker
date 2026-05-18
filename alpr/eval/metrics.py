"""Scoring metrics for ALPR pipelines.

No external deps: Levenshtein distance is implemented inline via the standard
DP table, avoiding a python-Levenshtein install.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from alpr.pipelines.base import (
    LABEL_WILDCARD,
    canonical_for_scoring,
    canonical_translate,
    normalize_label_text,
    normalize_plate_text,
)


# Label sentinels written by alpr.cli.label
UNREADABLE = "__UNREADABLE__"
NO_PLATE = "__NO_PLATE__"
SKIP_VALUES = {UNREADABLE, NO_PLATE, "", None}


def levenshtein(a: str, b: str, wildcard: str = "") -> int:
    """Pure-Python edit distance.  Plate strings are short (≤8 chars), so O(n*m) is fine.

    If `wildcard` is non-empty, occurrences of that char in `b` (the "truth" side) match
    any character in `a` at substitution cost 0.  Insertions/deletions still cost 1.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            char_cost = 0 if (ca == cb or (wildcard and cb == wildcard)) else 1
            sub = prev[j - 1] + char_cost
            cur[j] = min(ins, dele, sub)
        prev = cur
    return prev[-1]


def plate_match(ocr: str, truth: str, mode: str = "exact") -> bool:
    """Compare an OCR string to a (possibly-wildcarded) truth string.

    mode="exact"  — strict per-char match; '.' in truth matches any single OCR char.
    mode="canon"  — same but after collapsing OCR confusions (O↔0, I↔1, S↔5, Z↔2, B↔8).
    """
    if len(ocr) != len(truth):
        return False
    if mode == "canon":
        ocr = canonical_for_scoring(ocr)
        truth = canonical_translate(truth)  # preserves '.' wildcards
    return all(o == t or t == LABEL_WILDCARD for o, t in zip(ocr, truth))


def per_pipeline_detection_rate(records: list[dict]) -> dict[str, float]:
    """% of snaps where the pipeline emitted a detection bbox."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [hits, total]
    for r in records:
        p = r["pipeline"]
        counts[p][1] += 1
        if r.get("det_bbox"):
            counts[p][0] += 1
    return {p: (h / max(1, t)) for p, (h, t) in counts.items()}


def per_pipeline_read_rate(records: list[dict]) -> dict[str, float]:
    """% of snaps where the pipeline returned a non-empty OCR string."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        p = r["pipeline"]
        counts[p][1] += 1
        if r.get("ocr_text"):
            counts[p][0] += 1
    return {p: (h / max(1, t)) for p, (h, t) in counts.items()}


def cross_pipeline_agreement(records: list[dict]) -> dict[str, float]:
    """Pairwise exact-match and canonical-match agreement between every pair of pipelines.

    Returns a flat dict like {"bespoke_vs_preferred_exact": 0.41, "bespoke_vs_preferred_canon": 0.55}.
    """
    by_image: dict[str, dict[str, str | None]] = defaultdict(dict)
    pipelines: set[str] = set()
    for r in records:
        img = r["image"]
        p = r["pipeline"]
        pipelines.add(p)
        by_image[img][p] = r.get("ocr_text")

    pipelines_sorted = sorted(pipelines)
    out: dict[str, float] = {}
    for i, a in enumerate(pipelines_sorted):
        for b in pipelines_sorted[i + 1 :]:
            exact, canon, total = 0, 0, 0
            for img, reads in by_image.items():
                ta, tb = reads.get(a), reads.get(b)
                if not ta or not tb:
                    continue
                total += 1
                if ta == tb:
                    exact += 1
                if canonical_for_scoring(ta) == canonical_for_scoring(tb):
                    canon += 1
            out[f"{a}_vs_{b}_exact"] = exact / max(1, total)
            out[f"{a}_vs_{b}_canon"] = canon / max(1, total)
            out[f"{a}_vs_{b}_overlap_n"] = total
    return out


def label_based_accuracy(
    records: list[dict], labels: dict[str, dict]
) -> dict[str, dict]:
    """For each pipeline: exact-match accuracy, canonical-match accuracy, mean edit distance.

    Only counts images where the user labeled a real plate string (not __UNREADABLE__).
    """
    per_pipe: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "exact": 0, "canon": 0, "edit_sum": 0, "edit_n": 0}
    )

    valid_labels = {
        img: normalize_label_text(info["plate"])  # preserves '.' wildcards
        for img, info in labels.items()
        if info.get("plate") not in SKIP_VALUES
    }

    for r in records:
        img = r["image"]
        if img not in valid_labels:
            continue
        truth = valid_labels[img]
        if not truth:
            continue
        p = r["pipeline"]
        per_pipe[p]["n"] += 1
        ocr = r.get("ocr_text") or ""
        if plate_match(ocr, truth, mode="exact"):
            per_pipe[p]["exact"] += 1
        if plate_match(ocr, truth, mode="canon"):
            per_pipe[p]["canon"] += 1
        per_pipe[p]["edit_sum"] += levenshtein(ocr, truth, wildcard=LABEL_WILDCARD)
        per_pipe[p]["edit_n"] += 1

    summary = {}
    for p, s in per_pipe.items():
        n = max(1, s["n"])
        summary[p] = {
            "labeled_n": s["n"],
            "exact_accuracy": s["exact"] / n,
            "canonical_accuracy": s["canon"] / n,
            "mean_edit_distance": s["edit_sum"] / max(1, s["edit_n"]),
        }
    return summary


def per_track_best_of_n(
    records: list[dict], labels: dict[str, dict]
) -> dict[str, dict]:
    """Per-pipeline 'did any snap of this track produce the correct plate?' rate.

    A track is 'labeled' if at least one of its main_snaps has a real plate label.
    The truth is taken from the most-recently-labeled snap of that track (assumed consistent;
    the same plate is on every snap of a given track).
    """
    # Group label by track via filename parsing.
    from alpr.pipelines.base import parse_snap_filename
    track_truth: dict[int, str] = {}
    for img, info in labels.items():
        parsed = parse_snap_filename(img)
        if not parsed:
            continue
        plate = info.get("plate")
        if plate in SKIP_VALUES:
            continue
        _, tid, _ = parsed
        track_truth[tid] = normalize_label_text(plate)

    # Group OCR reads by (pipeline, track_id)
    per_pipeline: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("ocr_text"):
            per_pipeline[r["pipeline"]][r["track_id"]].append(r["ocr_text"])

    out: dict[str, dict] = {}
    for p, tracks in per_pipeline.items():
        labeled_tracks = [tid for tid in tracks if tid in track_truth]
        hits = 0
        for tid in labeled_tracks:
            truth = track_truth[tid]
            if any(plate_match(read, truth, mode="exact") for read in tracks[tid]):
                hits += 1
        out[p] = {
            "labeled_tracks": len(labeled_tracks),
            "best_of_n_accuracy": hits / max(1, len(labeled_tracks)),
        }
    return out


def char_confusion_table(
    records: list[dict], labels: dict[str, dict]
) -> dict[str, list[tuple[str, str, int]]]:
    """Top character substitutions per pipeline, useful for spotting model biases.

    For each labeled snap, aligns truth and OCR by left-padding the shorter to the longer
    length with '_', then counts char-pair mismatches.  Returns top-10 substitutions per pipeline.
    """
    counters: dict[str, Counter] = defaultdict(Counter)
    valid_labels = {
        img: normalize_label_text(info["plate"])
        for img, info in labels.items()
        if info.get("plate") not in SKIP_VALUES
    }
    for r in records:
        img = r["image"]
        if img not in valid_labels:
            continue
        truth = valid_labels[img]
        ocr = r.get("ocr_text") or ""
        if not truth or not ocr:
            continue
        n = max(len(truth), len(ocr))
        t = truth.rjust(n, "_")
        o = ocr.rjust(n, "_")
        for ct, co in zip(t, o):
            if ct == LABEL_WILDCARD:
                continue  # truth char unknown — no specific substitution to record
            if ct != co:
                counters[r["pipeline"]][(ct, co)] += 1

    out: dict[str, list[tuple[str, str, int]]] = {}
    for p, c in counters.items():
        out[p] = [(t, o, n) for (t, o), n in c.most_common(10)]
    return out
