"""Load the eligible Phenopacket Store corpus and build train/holdout splits.

Eligibility (DESIGN.md): an OMIM disease is kept when it has >= MIN_CASES cases and
a *single* causal gene, so rank-scoring has unambiguous ground truth. The single-gene
test uses both the gene(s) actually seen in the store packets and (optionally) the
genes_to_disease mapping. Empirically the single-gene rule is nearly free (~99.3% of
store cases are single-gene).

A Case is reduced to the set of non-excluded, de-duplicated HPO term ids — the signal
the generator models. Demographics/variants are intentionally out of scope for v1.
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Case:
    case_id: str            # source file stem (unique within the store)
    omim: str               # OMIM disease id — the conditioning label + ground-truth anchor
    gene: str               # causal gene symbol
    terms: tuple[str, ...]  # sorted, de-duplicated, non-excluded HPO term ids


def _extract(ppkt: dict) -> tuple[list[str], set[str], list[str]]:
    """Return (omim ids, gene symbols, hpo term ids) from one phenopacket dict."""
    omims = [
        d.get("term", {}).get("id")
        for d in (ppkt.get("diseases") or [])
        if str(d.get("term", {}).get("id")).startswith("OMIM")
    ]
    genes: set[str] = set()
    for interp in ppkt.get("interpretations") or []:
        for g in interp.get("diagnosis", {}).get("genomicInterpretations") or []:
            sym = (
                ((g.get("variantInterpretation") or {}).get("variationDescriptor") or {})
                .get("geneContext", {})
                .get("symbol")
            )
            if sym:
                genes.add(sym)
    hpo = sorted(
        {
            f.get("type", {}).get("id")
            for f in (ppkt.get("phenotypicFeatures") or [])
            if not f.get("excluded") and f.get("type", {}).get("id")
        }
    )
    return omims, genes, hpo


def _load_g2d(path: Path) -> dict[str, set[str]]:
    g2d: dict[str, set[str]] = collections.defaultdict(set)
    if not path.exists():
        return g2d
    with path.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            g2d[row["disease_id"]].add(row["gene_symbol"])
    return g2d


def load_store_cases(store_dir: Path | None = None) -> list[Case]:
    """Walk the store bundle and return one Case per packet that has an OMIM id,
    exactly one gene *in that packet*, and >= 1 HPO term."""
    store_dir = store_dir or config.STORE_DIR
    files = glob.glob(str(store_dir / "**" / "*.json"), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"no phenopackets under {store_dir} — set SYNTHVAE_STORE_DIR or run the "
            "benchmark's download.sh first."
        )
    out: list[Case] = []
    for fp in files:
        try:
            ppkt = json.loads(Path(fp).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        omims, genes, hpo = _extract(ppkt)
        if not omims or len(genes) != 1 or not hpo:
            continue
        gene = next(iter(genes))
        for omim in omims:
            out.append(Case(Path(fp).stem, omim, gene, tuple(hpo)))
    return out


def build_corpus(
    min_cases: int = config.MIN_CASES,
    *,
    require_single_g2d: bool = True,
    store_dir: Path | None = None,
) -> list[Case]:
    """Filter to eligible diseases: >= min_cases, single gene in-store, and (optionally)
    a single genes_to_disease mapping. Returns the kept cases."""
    cases = load_store_cases(store_dir)
    by_disease: dict[str, list[Case]] = collections.defaultdict(list)
    genes_seen: dict[str, set[str]] = collections.defaultdict(set)
    for c in cases:
        by_disease[c.omim].append(c)
        genes_seen[c.omim].add(c.gene)

    g2d = _load_g2d(config.GENES_TO_DISEASE) if require_single_g2d else {}

    def eligible(omim: str) -> bool:
        if len(by_disease[omim]) < min_cases:
            return False
        if len(genes_seen[omim]) != 1:  # single gene actually observed in the store
            return False
        if require_single_g2d and len(g2d.get(omim, set())) != 1:
            return False
        return True

    kept = [c for omim in by_disease if eligible(omim) for c in by_disease[omim]]
    return kept


def stratified_split(
    cases: list[Case],
    holdout_frac: float = config.HOLDOUT_FRAC,
    seed: int = config.SPLIT_SEED,
) -> tuple[list[Case], list[Case]]:
    """Per-disease holdout: every disease contributes to both train and holdout, so no
    disease is unseen at eval time. Each disease holds out ceil(frac * n) cases (>=1)."""
    rng = random.Random(seed)
    by_disease: dict[str, list[Case]] = collections.defaultdict(list)
    for c in cases:
        by_disease[c.omim].append(c)
    train: list[Case] = []
    holdout: list[Case] = []
    for omim, group in sorted(by_disease.items()):
        g = group[:]
        rng.shuffle(g)
        n_hold = max(1, round(holdout_frac * len(g)))
        holdout.extend(g[:n_hold])
        train.extend(g[n_hold:])
    return train, holdout


def vocabulary(cases: list[Case]) -> list[str]:
    """Sorted list of HPO term ids appearing in the given cases (the encoding columns)."""
    return sorted({t for c in cases for t in c.terms})


def multihot(cases: list[Case], vocab: list[str]):
    """Return an (n_cases x n_terms) float32 multi-hot matrix. numpy required."""
    import numpy as np

    index = {t: i for i, t in enumerate(vocab)}
    m = np.zeros((len(cases), len(vocab)), dtype="float32")
    for r, c in enumerate(cases):
        for t in c.terms:
            j = index.get(t)
            if j is not None:
                m[r, j] = 1.0
    return m


def disease_labels(cases: list[Case]) -> tuple[list[str], dict[str, int]]:
    """Sorted disease vocabulary and an omim->index map for conditioning."""
    diseases = sorted({c.omim for c in cases})
    return diseases, {d: i for i, d in enumerate(diseases)}
