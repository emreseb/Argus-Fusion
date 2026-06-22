#!/usr/bin/env python3
"""Evaluate ONE model (IR) on its test split. Nothing else.

Runs Ultralytics' built-in validation for the IR weights against the test split
declared in yaml-files/ir-model.yaml and prints P / R / mAP50 / mAP50-95.

Run with the env that has ultralytics, e.g.:
    ~/anaconda3/envs/YOLO-CUDA/bin/python eval_ir.py
"""

import os
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(_HERE, "fusion/models/ir/IR-150-epoch.pt")
DATA = os.path.join(_HERE, "yaml-files/ir-model.yaml")


def main():
    model = YOLO(MODEL)
    m = model.val(data=DATA, split="test")  # standard YOLO val on the test split
    b = m.box
    print("\n=== IR model — test split ===")
    print(f"precision  (P) : {b.mp:.4f}")
    print(f"recall     (R) : {b.mr:.4f}")
    print(f"mAP50          : {b.map50:.4f}")
    print(f"mAP50-95       : {b.map:.4f}")


if __name__ == "__main__":
    main()
