"""Per-track main-stream-snapshot scheduler.

Python-3.6 backport of ``streettracker.device.snap_planner``. Both must
stay behaviour-equivalent — the canonical implementation lives in
``StreetTracker``; this file exists only so NanoTracker can pick up the
fix before phase-3 cutover lands.

Background
~~~~~~~~~~

The previous ``nano_tracker.py`` logic fired the first snapshot the
instant a track's bbox crossed ``area_threshold_frac`` (default 5% of
frame area), then re-fired on each 1.5x area growth. Inspection of
``session_20260518_203417`` (44 vehicle snaps, 22 tracks): of the 13
tracks whose plate was OCR-readable in post-processing, 8 read on snap
2 and only 5 on snap 1. The first snap fired when the vehicle was still
too far away for the plate to be readable at 4K.

The planner here treats every frame as a candidate, tracks a running
quality score (area times sharpness, soft-saturating), and fires when
the score is either:

- **decaying** past the peak (vehicle has moved past closest point),
- **about to leave the frame** in the direction of motion, or
- **plateaued** (slow / stopped vehicle).

A finalize-time fallback rescues tracks that never produced a normal
fire. Pure-Python, no I/O — easy to unit-test against synthetic
trajectories. The actual HTTP snapshot dispatch is the caller's job.
"""

# Python 3.6 compatible: no @dataclass, no PEP-604 unions, typing-module
# generics only. f-strings ARE in 3.6 so we use them freely.

from typing import Dict, Optional, Tuple


# ----------------------------------------------------------------------
# Config + decision value objects (plain classes with __slots__ — works
# back to Py3.6 without the dataclass backport).


class SnapPlannerConfig(object):
    """Tunable knobs. Defaults calibrated against the May-2026 capture
    failure pattern (snap-1 success rate << snap-2 success rate).

    Field-level rationale:

    - ``area_threshold_frac``: minimum bbox area as a fraction of frame
      area before we even consider a track. Matches the original
      ``snapshot.area_threshold_frac`` config knob.
    - ``max_per_track``: hard cap on snaps committed per track. Each
      fire is an HTTP round-trip that queues on the snapshotter.
    - ``min_sharpness``: minimum variance-of-Laplacian on the bbox
      centre crop. 0 disables. Blurred frames are skipped, not
      counted against budget.
    - ``decay_ratio``: score must drop below ``peak * decay_ratio``
      before we accept that the peak has passed.
    - ``plateau_frames``: if the peak hasn't moved for this many
      frames, fire anyway (covers slow / stopped vehicles where no
      decay event will arrive).
    - ``max_wait_frames``: safety net — never sit on a track forever.
    - ``exit_margin_frac``: bbox within this fraction of the frame
      edge counts as "about to leave" and forces an immediate fire.
    - ``reset_factor``: after a fire, ``peak_score *= reset_factor``,
      so a refire only triggers on a genuinely larger second peak.
      Values >= 1.0 prevent decay-tail noise from re-peaking.
    """

    __slots__ = ("area_threshold_frac", "max_per_track", "min_sharpness",
                 "decay_ratio", "plateau_frames", "max_wait_frames",
                 "exit_margin_frac", "post_fire_cooldown_frames",
                 "right_half_only", "road_gate")

    def __init__(self,
                 area_threshold_frac=0.05,
                 max_per_track=1,
                 min_sharpness=0.0,
                 decay_ratio=0.92,
                 plateau_frames=20,
                 max_wait_frames=90,
                 exit_margin_frac=0.04,
                 post_fire_cooldown_frames=30,
                 right_half_only=True,
                 road_gate=None):
        self.area_threshold_frac = float(area_threshold_frac)
        self.max_per_track = int(max_per_track)
        self.min_sharpness = float(min_sharpness)
        self.decay_ratio = float(decay_ratio)
        self.plateau_frames = int(plateau_frames)
        self.max_wait_frames = int(max_wait_frames)
        self.exit_margin_frac = float(exit_margin_frac)
        self.post_fire_cooldown_frames = int(post_fire_cooldown_frames)
        # When True: only fire while the bbox centre is in the right
        # half of the frame (x_center >= frame_w / 2).  That is the
        # plate-readable zone for this camera angle -- right-to-left
        # vehicles ENTER through it (front plate close, large), and
        # left-to-right vehicles EXIT through it (rear plate close,
        # large).  Inside the gate the planner skips the peak/decay
        # wait and fires on the first eligible frame.
        self.right_half_only = bool(right_half_only)
        # Operator-traced road polygon + axis triggers.  When set,
        # supersedes right_half_only and the legacy peak/decay path.
        # See RoadGateConfig.
        self.road_gate = road_gate  # type: Optional[RoadGateConfig]


class RoadGateConfig(object):
    """Operator-traced road polygon plus a list of trigger positions
    along the polygon's principal axis.

    Mirrors :class:`streettracker.device.snap_planner.RoadGateConfig`.

    Fields:
      - ``polygon_frac``: list of ``(x_frac, y_frac)`` polygon vertices
        in fractional frame coordinates (so the same config works at
        any source resolution).
      - ``trigger_t_prime``: list of values in ``[0, 1]`` along the
        usable portion of the polygon's principal axis.  Each entry
        defines a virtual trigger line perpendicular to the axis;
        when a tracked vehicle's bbox centre crosses one, a snap
        fires.
      - ``t_usable_frac``: ``(lo, hi)`` clip range over the raw
        polygon t-range.  Lets us trim a "distant tip" portion of
        the polygon where number plates are at the wrong angle.
    """

    __slots__ = ("polygon_frac", "trigger_t_prime", "t_usable_frac")

    def __init__(self, polygon_frac, trigger_t_prime, t_usable_frac=(0.0, 1.0)):
        # type: (list, list, tuple) -> None
        self.polygon_frac = [(float(x), float(y)) for x, y in polygon_frac]
        self.trigger_t_prime = [float(t) for t in trigger_t_prime]
        lo, hi = t_usable_frac
        self.t_usable_frac = (float(lo), float(hi))


# Trigger reason strings (mirror the Literal in the StreetTracker port).
REASON_INELIGIBLE = "ineligible"
REASON_BLUR_GATE = "blur_gate"
REASON_BUDGET_EXHAUSTED = "budget_exhausted"
REASON_TRACKING = "tracking"
REASON_PEAK_DECAY = "peak_decay"
REASON_EXIT_IMMINENT = "exit_imminent"
REASON_PLATEAU = "plateau"
REASON_WAIT_TIMEOUT = "wait_timeout"
REASON_FINALIZE_LAST_CHANCE = "finalize_last_chance"
REASON_RIGHT_HALF_ENTRY = "right_half_entry"
REASON_OUT_OF_GATE = "out_of_gate"
REASON_TRIGGER_CROSSING = "trigger_crossing"
REASON_OUTSIDE_ROAD_POLYGON = "outside_road_polygon"


class SnapDecision(object):
    """What the planner decided for a single frame.

    ``snap_index`` is the 1-based fire ordinal — only populated when
    ``should_fire`` is True. Matches the existing
    ``vehicle_<id>_main_<N>.jpg`` filename convention.
    """

    __slots__ = ("should_fire", "reason", "score", "snap_index")

    def __init__(self, should_fire, reason, score, snap_index=None):
        self.should_fire = bool(should_fire)
        self.reason = str(reason)
        self.score = float(score)
        self.snap_index = snap_index  # int or None


class _TrackState(object):
    """Per-track scheduler state. Internal."""

    __slots__ = ("fires_committed", "peak_score", "frame_idx_at_peak",
                 "frames_eligible", "last_bbox", "cooldown_until_frame",
                 "ever_in_right_half", "zones_fired",
                 "triggers_fired", "prev_t_prime", "ever_in_road_gate")

    def __init__(self):
        self.fires_committed = 0
        self.peak_score = 0.0
        self.frame_idx_at_peak = -1
        self.frames_eligible = 0
        self.last_bbox = None  # Optional[Tuple[float, float, float, float]]
        self.cooldown_until_frame = -1
        # Sticky once True: needed so finalize-fallback only fires for
        # tracks that visited the plate-readable zone at some point.
        self.ever_in_right_half = False
        # Set of zone indices that have already fired for this track.
        # In right_half_only mode the right half is split into
        # ``max_per_track`` equal-width zones; each zone fires once.
        self.zones_fired = set()  # type: set
        # Road-gate mode state: trigger indices already fired and the
        # previous frame's normalised t' value (None on first frame
        # inside the polygon -- crossings need two consecutive samples).
        self.triggers_fired = set()  # type: set
        self.prev_t_prime = None     # type: Optional[float]
        self.ever_in_road_gate = False


# ----------------------------------------------------------------------
# Pure scoring helpers (used by the planner; exposed for testing).


def compute_quality_score(bbox, frame_size, sharpness=None):
    # type: (Tuple[float, float, float, float], Tuple[int, int], Optional[float]) -> float
    """Quality score in ``[0, 1]`` for a single bbox observation.

    Combines:
    - ``area_frac``: bbox area / frame area. More pixels-on-plate is
      always better, up to soft saturation at 25 % frame coverage.
      Beyond that the vehicle fills the frame and bigger isn't more
      useful.
    - ``sharpness``: variance-of-Laplacian, soft-saturated at 100.
      ``None`` is treated as a fully-sharp frame.

    Position is deliberately NOT in the score: a bbox clipped against
    a frame edge already has reduced area, so the bias is captured by
    ``area_frac`` alone.
    """
    x1, y1, x2, y2 = bbox
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw == 0 or bh == 0:
        return 0.0
    fw, fh = frame_size
    frame_area = max(1.0, float(fw) * float(fh))
    area_frac = (bw * bh) / frame_area
    area_score = min(1.0, area_frac / 0.25)
    if sharpness is None:
        sharp_score = 1.0
    else:
        sharp_score = min(1.0, max(0.0, sharpness) / 100.0)
    return area_score * sharp_score


def is_exit_imminent(bbox, prev_bbox, frame_size, exit_margin_frac):
    # type: (Tuple[float, float, float, float], Optional[Tuple[float, float, float, float]], Tuple[int, int], float) -> bool
    """True iff the bbox is hugging the frame edge in its direction
    of motion.

    Without a previous bbox we can't tell direction, so we fall back
    to "bbox within ``2 * exit_margin_frac`` of an edge" as a
    pessimistic proxy.
    """
    fw, fh = frame_size
    margin_x = exit_margin_frac * fw
    margin_y = exit_margin_frac * fh
    x1, y1, x2, y2 = bbox

    if prev_bbox is None:
        return (x1 < margin_x * 2
                or y1 < margin_y * 2
                or x2 > fw - margin_x * 2
                or y2 > fh - margin_y * 2)

    px1, py1, px2, py2 = prev_bbox
    dx = (x1 + x2) * 0.5 - (px1 + px2) * 0.5
    dy = (y1 + y2) * 0.5 - (py1 + py2) * 0.5

    if dx > 0 and x2 > fw - margin_x:
        return True
    if dx < 0 and x1 < margin_x:
        return True
    if dy > 0 and y2 > fh - margin_y:
        return True
    if dy < 0 and y1 < margin_y:
        return True
    return False


def _bbox_area_frac(bbox, frame_size):
    # type: (Tuple[float, float, float, float], Tuple[int, int]) -> float
    x1, y1, x2, y2 = bbox
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    fw, fh = frame_size
    frame_area = max(1.0, float(fw) * float(fh))
    return (bw * bh) / frame_area


def right_half_zone_index(bbox_center_x, frame_w, num_zones):
    # type: (float, int, int) -> int
    """Zone index in ``[0, num_zones-1]`` for a bbox centre in the
    right half. Zone 0 is the inner edge (adjacent to the midline),
    zone ``num_zones - 1`` is the outer edge (the frame's right edge).
    Caller is expected to have already established that the centre
    is inside the right half; values just inside the midline still
    map to zone 0 and values past the right edge clamp to the last
    zone, but ``num_zones`` itself must be >= 1.
    """
    if num_zones <= 1:
        return 0
    half_w = frame_w * 0.5
    if half_w <= 0:
        return 0
    rel = (bbox_center_x - half_w) / half_w  # 0.0 at midline, 1.0 at right edge
    if rel < 0.0:
        rel = 0.0
    if rel > 0.9999999:
        rel = 0.9999999
    return int(rel * num_zones)


def _point_in_polygon(px, py, poly):
    # type: (float, float, list) -> bool
    """Standard ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


class RoadGate(object):
    """Geometry helper for road-polygon + axis-trigger mode.

    Constructed once per session from a ``RoadGateConfig`` plus frame
    dimensions.  Mirrors :class:`streettracker.device.snap_planner.RoadGate`
    -- must stay behaviour-equivalent.
    """

    __slots__ = ("polygon_px", "axis_x", "axis_y",
                 "centroid_x", "centroid_y",
                 "t_min_raw", "t_max_raw",
                 "t_usable_lo", "t_usable_hi",
                 "triggers_t_prime")

    def __init__(self, cfg, frame_w, frame_h):
        # type: (RoadGateConfig, int, int) -> None
        if len(cfg.polygon_frac) < 3:
            raise ValueError("road polygon needs at least 3 vertices")
        if not cfg.trigger_t_prime:
            raise ValueError("road gate has no triggers")
        # Polygon in pixel coords for the active frame size.
        self.polygon_px = [(fx * frame_w, fy * frame_h) for fx, fy in cfg.polygon_frac]
        # Centroid in pixel coords.
        n = len(self.polygon_px)
        cx = sum(p[0] for p in self.polygon_px) / n
        cy = sum(p[1] for p in self.polygon_px) / n
        # 2x2 covariance over the vertex set.
        sxx = syy = sxy = 0.0
        for x, y in self.polygon_px:
            dx = x - cx
            dy = y - cy
            sxx += dx * dx
            syy += dy * dy
            sxy += dx * dy
        sxx /= n
        syy /= n
        sxy /= n
        # Largest eigenvalue of [[sxx, sxy], [sxy, syy]] and its eigenvector.
        tr = sxx + syy
        det = sxx * syy - sxy * sxy
        disc = max(0.0, (tr * tr) * 0.25 - det)
        lam = tr * 0.5 + disc ** 0.5
        if abs(sxy) > 1e-9:
            ex, ey = lam - syy, sxy
        else:
            if sxx >= syy:
                ex, ey = 1.0, 0.0
            else:
                ex, ey = 0.0, 1.0
        norm = (ex * ex + ey * ey) ** 0.5
        ex, ey = ex / norm, ey / norm
        # Orient axis so it points toward "near" (image y increases
        # downward; the closest road segment has the largest y in our
        # camera setup).
        if ey < 0:
            ex, ey = -ex, -ey
        self.axis_x = ex
        self.axis_y = ey
        self.centroid_x = cx
        self.centroid_y = cy
        # Project each polygon vertex onto the axis to find t-range.
        ts = [(x - cx) * ex + (y - cy) * ey for x, y in self.polygon_px]
        self.t_min_raw = min(ts)
        self.t_max_raw = max(ts)
        lo, hi = cfg.t_usable_frac
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(
                "t_usable_frac must satisfy 0 <= lo < hi <= 1, got {0}".format(
                    cfg.t_usable_frac))
        self.t_usable_lo = lo
        self.t_usable_hi = hi
        self.triggers_t_prime = tuple(cfg.trigger_t_prime)

    def contains(self, px, py):
        # type: (float, float) -> bool
        return _point_in_polygon(px, py, self.polygon_px)

    def t_prime(self, px, py):
        # type: (float, float) -> Optional[float]
        """Image-space point -> normalised ``t'`` in ``[0, 1]`` over
        the usable t-range.  Returns None if outside the usable band."""
        t_raw = (px - self.centroid_x) * self.axis_x + (py - self.centroid_y) * self.axis_y
        span = self.t_max_raw - self.t_min_raw
        if span <= 0:
            return None
        t_norm = (t_raw - self.t_min_raw) / span
        if t_norm < self.t_usable_lo or t_norm > self.t_usable_hi:
            return None
        usable_span = self.t_usable_hi - self.t_usable_lo
        return (t_norm - self.t_usable_lo) / usable_span

    def crossings(self, prev_tp, cur_tp, already_fired):
        # type: (Optional[float], float, set) -> list
        """Trigger indices whose t' lies strictly between prev_tp and
        cur_tp (in either direction), excluding any already fired.
        Order: closest-to-prev first."""
        if prev_tp is None:
            return []
        if prev_tp <= cur_tp:
            lo, hi = prev_tp, cur_tp
        else:
            lo, hi = cur_tp, prev_tp
        hits = []
        for idx, t in enumerate(self.triggers_t_prime):
            if idx in already_fired:
                continue
            if lo < t <= hi:
                hits.append((abs(t - prev_tp), idx))
        hits.sort()
        return [idx for _, idx in hits]

    @property
    def max_per_track(self):
        return len(self.triggers_t_prime)


# ----------------------------------------------------------------------
# Planner: per-session decision engine.


class SnapPlanner(object):
    """Per-frame decision engine. One instance per session.

    Usage::

        planner = SnapPlanner(frame_w, frame_h)

        for tr in active_tracks:
            decision = planner.consider(
                track_id=tr.id,
                bbox=(tr.points[-1].x1, tr.points[-1].y1,
                      tr.points[-1].x2, tr.points[-1].y2),
                frame_idx=frame_idx,
                sharpness=measure_sharpness(...),
            )
            if decision.should_fire:
                fire_snapshot(tr, decision.snap_index)

        for expired_tr in expired_tracks_this_frame:
            decision = planner.on_track_finalize(expired_tr.id)
            if decision.should_fire:
                fire_snapshot(expired_tr, decision.snap_index)
            planner.forget(expired_tr.id)
    """

    __slots__ = ("_cfg", "_frame_size", "_state", "_road_gate")

    def __init__(self, frame_width, frame_height, config=None):
        # type: (int, int, Optional[SnapPlannerConfig]) -> None
        self._cfg = config if config is not None else SnapPlannerConfig()
        self._frame_size = (int(frame_width), int(frame_height))
        self._state = {}  # type: Dict[int, _TrackState]
        self._road_gate = None  # type: Optional[RoadGate]
        if self._cfg.road_gate is not None:
            self._road_gate = RoadGate(self._cfg.road_gate,
                                       self._frame_size[0],
                                       self._frame_size[1])

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def config(self):
        return self._cfg

    @property
    def road_gate(self):
        return self._road_gate

    def consider(self, track_id, bbox, frame_idx, sharpness=None):
        # type: (int, Tuple[float, float, float, float], int, Optional[float]) -> SnapDecision
        """Decide whether to fire a snap for ``track_id`` on this frame.

        Algorithm:

        1. If bbox below area threshold: ``ineligible``, no fire, no
           peak update. Tracks can re-enter eligibility after a dip.
        2. Compute the quality score, update the running peak.
        3. Fire iff ANY of:
           - ``exit_imminent``: bbox is leaving frame in motion direction.
           - ``peak_decay``: score < peak * decay_ratio.
           - ``plateau``: peak hasn't moved for ``plateau_frames``.
           - ``wait_timeout``: been eligible for too long.
        4. Blur gate applied AFTER trigger decision: blurred frames
           postpone the fire without consuming budget.
        5. After firing, ``peak_score *= reset_factor`` so a genuinely
           larger future peak can still re-trigger.
        """
        cfg = self._cfg
        st = self._state.get(track_id)
        if st is None:
            st = _TrackState()
            self._state[track_id] = st

        prev_bbox = st.last_bbox
        st.last_bbox = bbox

        score = compute_quality_score(bbox, self._frame_size, sharpness)
        area_frac = _bbox_area_frac(bbox, self._frame_size)

        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) * 0.5
        bbox_center_y = (y1 + y2) * 0.5
        frame_w = self._frame_size[0]
        in_right_half = bbox_center_x >= (frame_w * 0.5)
        if in_right_half:
            st.ever_in_right_half = True

        # Road-gate mode supersedes the other paths entirely.  Area
        # threshold + cooldown still apply as basic sanity checks.
        if self._road_gate is not None:
            return self._consider_road_gate(
                st, frame_idx, sharpness, score, area_frac,
                bbox_center_x, bbox_center_y,
            )

        if area_frac < cfg.area_threshold_frac:
            # Transition-rescue: a track that was eligible last frame,
            # never fired live, and is now dropping below the threshold
            # would otherwise wait for finalize -- which fires too late
            # (vehicle has left frame).  Fire once on the eligible to
            # ineligible transition instead.  Covers the "just-barely-
            # eligible peak" case where the next post-peak frame is
            # below threshold before any decay can trigger.
            #
            # In right_half_only mode the rescue is itself gated on
            # the track having been observed in the right half -- a
            # rescue snap from the left half is exactly the failure
            # mode that motivated the gate.
            should_rescue = (
                st.frames_eligible > 0
                and st.fires_committed == 0
                and st.peak_score > 0
                and frame_idx >= st.cooldown_until_frame
                and (not cfg.right_half_only or st.ever_in_right_half)
            )
            st.frames_eligible = 0
            if should_rescue:
                st.fires_committed += 1
                st.cooldown_until_frame = frame_idx + cfg.post_fire_cooldown_frames
                return SnapDecision(True, REASON_EXIT_IMMINENT,
                                    score, st.fires_committed)
            return SnapDecision(False, REASON_INELIGIBLE, score)

        st.frames_eligible += 1

        # Cooldown: after a fire, suppress new triggers until the
        # decay tail subsides. Peak / eligibility state still updates
        # so clocks track correctly, but no trigger fires.
        in_cooldown = frame_idx < st.cooldown_until_frame

        if score > st.peak_score:
            st.peak_score = score
            st.frame_idx_at_peak = frame_idx

        frames_since_peak = frame_idx - st.frame_idx_at_peak

        if in_cooldown:
            return SnapDecision(False, REASON_TRACKING, score)

        # right_half_only short-circuit: split the right half into
        # ``max_per_track`` equal-width zones and fire ASAP on the
        # first eligible frame in each zone the track enters.  Each
        # zone fires at most once, so an oscillating tracker can't
        # re-fire the same one.  Out of gate, suppress (but
        # peak/eligibility above still ran so the planner is "armed"
        # the moment the bbox crosses in).
        if cfg.right_half_only:
            if not in_right_half:
                return SnapDecision(False, REASON_OUT_OF_GATE, score)
            if st.fires_committed >= cfg.max_per_track:
                return SnapDecision(False, REASON_BUDGET_EXHAUSTED, score)
            zone = right_half_zone_index(bbox_center_x, frame_w,
                                         cfg.max_per_track)
            if zone in st.zones_fired:
                return SnapDecision(False, REASON_TRACKING, score)
            if cfg.min_sharpness > 0 and (sharpness is None or sharpness < cfg.min_sharpness):
                return SnapDecision(False, REASON_BLUR_GATE, score)
            st.fires_committed += 1
            st.zones_fired.add(zone)
            st.cooldown_until_frame = frame_idx + cfg.post_fire_cooldown_frames
            st.peak_score = 0.0
            st.frame_idx_at_peak = frame_idx
            return SnapDecision(True, REASON_RIGHT_HALF_ENTRY,
                                score, st.fires_committed)

        reason = None
        if is_exit_imminent(bbox, prev_bbox, self._frame_size, cfg.exit_margin_frac):
            reason = REASON_EXIT_IMMINENT
        elif st.peak_score > 0 and score < st.peak_score * cfg.decay_ratio:
            reason = REASON_PEAK_DECAY
        elif frames_since_peak >= cfg.plateau_frames:
            reason = REASON_PLATEAU
        elif st.frames_eligible >= cfg.max_wait_frames:
            reason = REASON_WAIT_TIMEOUT

        if reason is None:
            return SnapDecision(False, REASON_TRACKING, score)

        if st.fires_committed >= cfg.max_per_track:
            return SnapDecision(False, REASON_BUDGET_EXHAUSTED, score)
        if cfg.min_sharpness > 0 and (sharpness is None or sharpness < cfg.min_sharpness):
            return SnapDecision(False, REASON_BLUR_GATE, score)

        st.fires_committed += 1
        # Arm cooldown and clear peak so the next post-cooldown frame
        # establishes a fresh peak baseline.
        st.cooldown_until_frame = frame_idx + cfg.post_fire_cooldown_frames
        st.peak_score = 0.0
        st.frame_idx_at_peak = frame_idx
        return SnapDecision(True, reason, score, st.fires_committed)

    def _consider_road_gate(self, st, frame_idx, sharpness, score,
                            area_frac, cx, cy):
        # type: (_TrackState, int, Optional[float], float, float, float, float) -> SnapDecision
        """Road-polygon + axis-crossing trigger evaluation.  Mirrors
        :meth:`streettracker.device.snap_planner.SnapPlanner._consider_road_gate`."""
        cfg = self._cfg
        gate = self._road_gate

        if not gate.contains(cx, cy):
            st.prev_t_prime = None
            return SnapDecision(False, REASON_OUTSIDE_ROAD_POLYGON, score)

        cur_tp = gate.t_prime(cx, cy)
        if cur_tp is None:
            st.prev_t_prime = None
            return SnapDecision(False, REASON_OUT_OF_GATE, score)

        st.ever_in_road_gate = True

        if area_frac < cfg.area_threshold_frac:
            # In the gate but bbox too small; hold prev_t_prime so the
            # next sufficiently-large frame still produces a meaningful
            # crossing test.
            return SnapDecision(False, REASON_INELIGIBLE, score)

        prev_tp = st.prev_t_prime
        if prev_tp is None:
            st.prev_t_prime = cur_tp
            return SnapDecision(False, REASON_TRACKING, score)

        crossings = gate.crossings(prev_tp, cur_tp, st.triggers_fired)
        if not crossings:
            st.prev_t_prime = cur_tp
            return SnapDecision(False, REASON_TRACKING, score)

        if frame_idx < st.cooldown_until_frame:
            # Hold prev_t_prime; the trigger stays detectable next frame.
            return SnapDecision(False, REASON_TRACKING, score)
        if st.fires_committed >= gate.max_per_track:
            st.prev_t_prime = cur_tp
            return SnapDecision(False, REASON_BUDGET_EXHAUSTED, score)
        if cfg.min_sharpness > 0 and (sharpness is None or sharpness < cfg.min_sharpness):
            return SnapDecision(False, REASON_BLUR_GATE, score)

        trig_idx = crossings[0]
        st.fires_committed += 1
        st.triggers_fired.add(trig_idx)
        st.cooldown_until_frame = frame_idx + cfg.post_fire_cooldown_frames
        # Advance prev to the fired trigger position (not cur_tp) so
        # any further triggers in the same forward motion are still
        # detected on subsequent frames.
        st.prev_t_prime = gate.triggers_t_prime[trig_idx]
        return SnapDecision(True, REASON_TRIGGER_CROSSING, score,
                            st.fires_committed)

    def on_track_finalize(self, track_id):
        # type: (int) -> SnapDecision
        """Last-chance fire for a track that ended without any snap.

        Behaviour depends on the active firing mode:
        * Road-gate mode has no fallback: a track that never crossed a
          trigger doesn't earn a snap.
        * ``right_half_only`` mode requires the track to have visited
          the right half at some point.
        * Legacy mode fires the fallback whenever the track was
          eligible at some point.
        """
        st = self._state.get(track_id)
        if st is None:
            return SnapDecision(False, REASON_INELIGIBLE, 0.0)
        if st.fires_committed > 0:
            return SnapDecision(False, REASON_BUDGET_EXHAUSTED, st.peak_score)
        if self._road_gate is not None:
            return SnapDecision(False, REASON_OUT_OF_GATE, st.peak_score)
        if st.peak_score == 0:
            return SnapDecision(False, REASON_INELIGIBLE, 0.0)
        if self._cfg.right_half_only and not st.ever_in_right_half:
            return SnapDecision(False, REASON_OUT_OF_GATE, st.peak_score)
        st.fires_committed += 1
        return SnapDecision(True, REASON_FINALIZE_LAST_CHANCE,
                            st.peak_score, st.fires_committed)

    def forget(self, track_id):
        # type: (int) -> None
        """Drop state for ``track_id``. Call after a track is finalized."""
        self._state.pop(track_id, None)

    def state_snapshot(self):
        # type: () -> Dict[int, Dict]
        """Read-only dict view of per-track state — for live logging."""
        out = {}
        for tid, s in self._state.items():
            out[tid] = {"fires": s.fires_committed,
                        "peak": s.peak_score,
                        "frames_eligible": s.frames_eligible,
                        "ever_in_right_half": s.ever_in_right_half,
                        "ever_in_road_gate": s.ever_in_road_gate,
                        "triggers_fired_count": len(s.triggers_fired)}
        return out
