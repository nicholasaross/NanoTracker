"""Debug HSV color voting on saved vehicle crops.  Standalone (no TRT)."""
import sys
import numpy as np
import cv2

# Inlined to avoid the nano_tracker -> trt_engine -> pycuda import chain.
COLOR_RANGES = [
    ((0, 0, 200),   (180, 30, 255),  "white"),
    ((0, 0, 0),     (180, 40, 40),   "black"),
    ((0, 0, 40),    (180, 40, 200),  "grey"),
    ((0, 60, 50),   (10, 255, 255),  "red"),
    ((170, 60, 50), (180, 255, 255), "red"),
    ((100, 80, 50), (130, 255, 255), "blue"),
    ((36, 80, 50),  (85, 255, 255),  "green"),
    ((20, 50, 180), (30, 150, 255),  "silver"),
    ((20, 80, 80),  (35, 255, 255),  "yellow"),
]
ACHROMATIC = {"white", "black", "grey", "silver"}
CHROMATIC_PREFER_FRAC = 0.15
MIN_INNER_PIXELS = 2000
PAD_FRAC = 0.2


def pick(counts, total):
    chromatic = {k: v for k, v in counts.items() if k not in ACHROMATIC}
    achromatic = {k: v for k, v in counts.items() if k in ACHROMATIC}
    best_chrom = max(chromatic.values()) if chromatic else 0
    if achromatic:
        best_ach_name = max(achromatic, key=lambda k: achromatic[k])
        if best_ach_name in ("white", "black") and achromatic[best_ach_name] > best_chrom:
            return best_ach_name + " (white/black plurality)"
    if chromatic:
        best_chrom_name = max(chromatic, key=lambda k: chromatic[k])
        if chromatic[best_chrom_name] >= CHROMATIC_PREFER_FRAC * total:
            return best_chrom_name + " (chromatic-over-grey rule)"
    return max(counts, key=lambda k: counts[k]) + " (fallback max)"


def analyze(path):
    img = cv2.imread(path)
    if img is None:
        print("Failed to read:", path)
        return
    h, w = img.shape[:2]
    inset_x = int(w * PAD_FRAC / (1.0 + 2.0 * PAD_FRAC))
    inset_y = int(h * PAD_FRAC / (1.0 + 2.0 * PAD_FRAC))
    inner = img[inset_y:max(inset_y + 1, h - inset_y),
                inset_x:max(inset_x + 1, w - inset_x)]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    counts = {}
    for low, high, name in COLOR_RANGES:
        m = cv2.inRange(hsv, np.array(low), np.array(high))
        counts[name] = counts.get(name, 0) + int(cv2.countNonZero(m))
    total = sum(counts.values())
    inner_size = inner.shape[0] * inner.shape[1]
    print("\n=== {} ===".format(path))
    print("  crop {}x{}, inner {}x{}".format(w, h, inner.shape[1], inner.shape[0]))
    print("  voted pixels: {} / {} ({:.0f}% coverage)".format(
        total, inner_size, 100.0 * total / inner_size if inner_size else 0))
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        if c > 0:
            pct = 100.0 * c / total if total else 0
            print("    {:<8} {:>7}  ({:.1f}% of voted)".format(name, c, pct))
    if inner_size < MIN_INNER_PIXELS:
        print("  RESULT: unknown (inner crop {} < {} min pixels)".format(inner_size, MIN_INNER_PIXELS))
    elif total == 0:
        print("  RESULT: unknown (no pixels voted)")
    else:
        print("  RESULT: {}".format(pick(counts, total)))
    # Also report unvoted (pixels that fell in no range) stats:
    unvoted = inner_size - total
    if unvoted > 0:
        # Compute hue/sat distribution of unvoted pixels
        all_in_any = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high, _ in COLOR_RANGES:
            all_in_any = cv2.bitwise_or(all_in_any, cv2.inRange(hsv, np.array(low), np.array(high)))
        unv_mask = (all_in_any == 0)
        unv_hsv = hsv[unv_mask]
        if unv_hsv.shape[0] > 0:
            print("  unvoted ({} px): H mean={:.0f} std={:.0f}, S mean={:.0f}, V mean={:.0f}".format(
                unv_hsv.shape[0],
                unv_hsv[:, 0].mean(), unv_hsv[:, 0].std(),
                unv_hsv[:, 1].mean(), unv_hsv[:, 2].mean(),
            ))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
