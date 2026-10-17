"""khmer_synth - fully synthetic Khmer handwriting trajectory + image generation.

IMPORTANT / SYNTHETIC BIAS WARNING
----------------------------------
This package produces *synthetic approximations* of handwriting trajectories that
are derived from font outlines (skeletonised filled shapes), NOT from recorded
human pen motion.  The stroke ORDER, stroke SEGMENTATION, pen VELOCITY and pen
PRESSURE produced here are heuristic constructions.  They must not be treated as
authentic Khmer human stroke-order ground truth or used to make linguistic /
palaeographic claims.

Intended use: OCR synthetic-data generation, representation pretraining and
bootstrapping of learned trajectory generators (later calibrated on a small
amount of real online-handwriting data).

See README.md sections "Limitations" and "Synthetic bias".
"""

from .version import __version__

__all__ = ["__version__"]
