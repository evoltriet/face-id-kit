from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from .matching import FaceIdentifier
from .types import LiveResult


def iou(left, right):
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    overlap = max(0., min(lx + lw, rx + rw) - max(lx, rx)) * max(0., min(ly + lh, ry + rh) - max(ly, ry))
    union = lw * lh + rw * rh - overlap
    return overlap / union if union > 0 else 0.


@dataclass
class _Track:
    bbox: tuple[float, float, float, float]
    seen_at: float
    history: deque = field(default_factory=deque)


class TrackConsensus:
    """Conservative geometric association. Ambiguous overlaps start fresh tracks."""
    def __init__(self, *, window: int = 6, required: int = 4,
                 expiry_seconds: float = 1.0, min_iou: float = 0.3):
        if not 1 <= required <= window or expiry_seconds <= 0 or not 0 < min_iou <= 1:
            raise ValueError("Invalid tracking configuration")
        self.window, self.required = window, required
        self.expiry_seconds, self.min_iou = expiry_seconds, min_iou
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0

    def reset(self):
        self._tracks.clear()

    def update(self, boxes, identity_ids, *, now: float | None = None):
        if len(boxes) != len(identity_ids):
            raise ValueError("Boxes and identities must have the same length")
        now = time.monotonic() if now is None else now
        self._tracks = {k: v for k, v in self._tracks.items() if now - v.seen_at < self.expiry_seconds}
        options = [[tid for tid, track in self._tracks.items() if iou(box, track.bbox) >= self.min_iou] for box in boxes]
        counts = {tid: sum(tid in choices for choices in options) for tid in self._tracks}
        ambiguous = {tid for choices in options for tid in choices if len(choices) != 1 or counts[tid] != 1}
        for tid in ambiguous:
            del self._tracks[tid]
        result, used = [], set()
        for box, identity, choices in zip(boxes, identity_ids, options):
            if len(choices) == 1 and choices[0] in self._tracks and counts[choices[0]] == 1:
                tid = choices[0]
                track = self._tracks[tid]
            else:
                tid = self._next_id
                self._next_id += 1
                track = _Track(tuple(box), now, deque(maxlen=self.window))
                self._tracks[tid] = track
            if track.history and identity is not None and track.history[-1] not in (identity, None):
                track.history.clear()
            track.bbox, track.seen_at = tuple(box), now
            track.history.append(identity)
            used.add(tid)
            result.append((tid, identity is not None and track.history.count(identity) >= self.required))
        for tid, track in self._tracks.items():
            if tid not in used:
                track.history.append(None)
        return result


class LiveSession:
    def __init__(self, faces: FaceIdentifier, *, max_faces: int | None = 4,
                 central_only: bool = False, tracker: TrackConsensus | None = None):
        self.faces, self.max_faces, self.central_only = faces, max_faces, central_only
        self.tracker = tracker or TrackConsensus()
        self._revision = faces.store.revision

    def update(self, image_bgr):
        revision = self.faces.store.revision
        if revision != self._revision:
            self.tracker.reset()
            self._revision = revision
        results = self.faces.identify(image_bgr, max_faces=self.max_faces, central_only=self.central_only)
        tracks = self.tracker.update([f.bbox for f, _ in results], [m.identity_id for _, m in results])
        return [LiveResult(tid, face, match, stable) for (face, match), (tid, stable) in zip(results, tracks)]
