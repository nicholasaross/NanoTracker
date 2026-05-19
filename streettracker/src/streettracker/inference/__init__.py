"""Ultralytics YOLO + BotSORT inference wrappers.

Replaces NanoTracker's hand-rolled `trt_engine.py` (manual YOLOv8 decode +
numpy NMS) and bespoke IoU tracker with a single Ultralytics call:
``model = YOLO('best.engine'); model.track(frame, persist=True,
tracker='botsort.yaml')``.
"""
