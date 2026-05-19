"""UltralyticsTracker — stub-based result conversion tests.

We don't load a real model here (ultralytics + torch are 2GB of wheels);
instead we test ``_ultralytics_results_to_detections`` against a stub that
quacks like an ``ultralytics.engine.results.Results`` object. That covers
the only StreetTracker-specific logic in the runner — Ultralytics itself
is exercised in phase-3 end-to-end tests on the Orin.
"""

from __future__ import annotations

from dataclasses import dataclass

from streettracker.inference.ultralytics_runner import (
    _ultralytics_results_to_detections,
)


@dataclass
class _FakeTensor:
    """Minimal stand-in for a torch.Tensor that supports .cpu().tolist()."""

    data: list

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> list:
        return self.data


@dataclass
class _FakeBoxes:
    xyxy: _FakeTensor
    conf: _FakeTensor
    cls: _FakeTensor
    id: _FakeTensor | None

    def __len__(self) -> int:
        return len(self.xyxy.data)


@dataclass
class _FakeResults:
    boxes: _FakeBoxes | None


def _make_results(xyxy, conf, cls, ids=None) -> _FakeResults:
    boxes = _FakeBoxes(
        xyxy=_FakeTensor(xyxy),
        conf=_FakeTensor(conf),
        cls=_FakeTensor(cls),
        id=_FakeTensor(ids) if ids is not None else None,
    )
    return _FakeResults(boxes=boxes)


def test_empty_results_yields_empty_list() -> None:
    out = _ultralytics_results_to_detections(_FakeResults(boxes=None))
    assert out == []


def test_empty_boxes_yields_empty_list() -> None:
    out = _ultralytics_results_to_detections(
        _make_results(xyxy=[], conf=[], cls=[], ids=[])
    )
    assert out == []


def test_single_detection_with_track_id() -> None:
    out = _ultralytics_results_to_detections(
        _make_results(
            xyxy=[[10.0, 20.0, 110.0, 220.0]],
            conf=[0.87],
            cls=[2.0],          # car
            ids=[42.0],
        )
    )
    assert len(out) == 1
    d = out[0]
    assert d.x1 == 10.0 and d.y1 == 20.0 and d.x2 == 110.0 and d.y2 == 220.0
    assert d.score == 0.87
    assert d.class_id == 2
    assert d.track_id == 42


def test_detection_without_track_id() -> None:
    out = _ultralytics_results_to_detections(
        _make_results(
            xyxy=[[0.0, 0.0, 10.0, 10.0]],
            conf=[0.4],
            cls=[7.0],
            ids=None,
        )
    )
    assert len(out) == 1
    assert out[0].track_id is None
    assert out[0].class_id == 7


def test_multiple_detections_are_ordered() -> None:
    out = _ultralytics_results_to_detections(
        _make_results(
            xyxy=[[0.0, 0.0, 10.0, 10.0], [50.0, 60.0, 100.0, 200.0]],
            conf=[0.9, 0.5],
            cls=[2.0, 5.0],
            ids=[1.0, 2.0],
        )
    )
    assert [d.track_id for d in out] == [1, 2]
    assert [d.class_id for d in out] == [2, 5]
