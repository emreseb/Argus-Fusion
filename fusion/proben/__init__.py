"""ProbEn — Probabilistic Ensembling for multimodal detection fusion.

Chen et al., "Multimodal Object Detection via Probabilistic Ensembling",
ECCV 2022. A drop-in late-fusion alternative to the plan's noisy-OR
``decision_logic``; depends only on numpy + the pipeline's Detection schema.
"""

from .proben import ProbEnConfig, proben_fuse, proben_two

__all__ = ["ProbEnConfig", "proben_fuse", "proben_two"]
