"""Orin-only modules: live runtime loop, Reolink HTTP snapshotter, HTTP
dashboard, IR/night-mode detection.

These modules import device-specific libraries (TensorRT, GStreamer) and
should not be imported on the dev box.
"""
