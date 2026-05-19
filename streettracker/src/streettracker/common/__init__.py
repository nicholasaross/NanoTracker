"""Schemas, color voting, HTML summary, hourly rollup, JSON I/O.

Used by both the device runtime (`streettracker.device`) and off-device
analysis (`streettracker.analysis`). No CUDA, no Ultralytics, no OpenCV
imports at module level — keep this layer cheap to import.
"""
