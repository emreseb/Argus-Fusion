"""Fusion evaluation package (numpy-only core).

``detection_metrics`` and ``yolo_gt`` import with numpy alone; the CLI
``run_eval`` additionally needs OpenCV + the model stack and is meant to be run
as a script (``python fusion/eval/run_eval.py``).
"""

from .detection_metrics import evaluate, IOU_SWEEP
from .yolo_gt import load_yolo_labels, label_path_for

__all__ = ["evaluate", "IOU_SWEEP", "load_yolo_labels", "label_path_for"]
