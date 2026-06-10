"""Paths and corpus constants.

The training corpus is the Phenopacket Store bundle that the existing
exomiser-rerank-benchmark already downloads; we read it in place rather than
copying. Override any path with an environment variable if your layout differs.
"""

from __future__ import annotations

import os
from pathlib import Path

# Root of the sibling benchmark repo that holds the downloaded data.
_BENCH = Path(
    os.environ.get(
        "SYNTHVAE_BENCH_REPO",
        Path.home() / "PythonProject" / "exomiser-rerank-benchmark",
    )
)

# Phenopacket Store bundle: data/ppkts/<release>/<GENE>/*.json
STORE_DIR = Path(os.environ.get("SYNTHVAE_STORE_DIR", _BENCH / "data" / "ppkts"))
# OMIM disease -> gene mapping used to confirm unambiguous ground truth.
GENES_TO_DISEASE = Path(
    os.environ.get("SYNTHVAE_GENES_TO_DISEASE", _BENCH / "data" / "genes_to_disease.txt")
)
# HPO ontology (obo) — used later for ancestor propagation / information content.
HP_OBO = Path(os.environ.get("SYNTHVAE_HP_OBO", _BENCH / "data" / "hp.obo"))

# Where models, splits, and eval artifacts land.
ARTIFACTS = Path(os.environ.get("SYNTHVAE_ARTIFACTS", Path(__file__).resolve().parent.parent / "artifacts"))

# Default eligibility / split knobs (see DESIGN.md).
MIN_CASES = 20          # diseases with >= this many cases are kept
HOLDOUT_FRAC = 0.20     # per-disease stratified holdout
SPLIT_SEED = 0
