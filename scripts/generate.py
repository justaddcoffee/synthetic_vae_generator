#!/usr/bin/env python
"""Generate synthetic phenopackets and write them to disk.

This is the turnkey entry point for producing cases: it builds the real corpus,
trains a generator (VAE by default), samples a synthetic cohort, and writes one
Phenopacket-schema v2 JSON file per generated case.

Run from the repo root, e.g.::

    uv run python scripts/generate.py --generator vae --per-disease 5 --out out/

The generators and corpus loading all come from the ``synthvae`` package; the
only logic unique to this script is turning a ``Case`` (case_id / omim / gene /
HPO term ids) back into a phenopacket dict. Because ``Case`` does not carry HPO
or disease *labels*, we harvest them in one pass over the real store JSONs and
fill them in where known.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from synthvae import config
from synthvae.baselines import CooccurrenceSampler, MarginalSampler
from synthvae.data import Case, build_corpus, stratified_split
from synthvae.eval import disease_counts
from synthvae.vae import VAEGenerator

SCHEMA_VERSION = "2.0"
HPO_RESOURCE = {
    "id": "hp",
    "name": "human phenotype ontology",
    "url": "http://purl.obolibrary.org/obo/hp.owl",
    "namespacePrefix": "HP",
    "iriPrefix": "http://purl.obolibrary.org/obo/HP_",
}


def harvest_labels(store_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """One pass over the real store JSONs to collect id -> label maps.

    Returns ``(hpo_labels, disease_labels)`` keyed by HPO id and OMIM id. These
    are the same files the pipeline trains on; we only read the labels that
    ``synthvae.data`` drops on load.
    """
    hpo: dict[str, str] = {}
    dis: dict[str, str] = {}
    for fp in glob.glob(str(store_dir / "**" / "*.json"), recursive=True):
        try:
            ppkt = json.loads(Path(fp).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for f in ppkt.get("phenotypicFeatures") or []:
            term = f.get("type") or {}
            tid, label = term.get("id"), term.get("label")
            if tid and label and tid not in hpo:
                hpo[tid] = label
        for d in ppkt.get("diseases") or []:
            term = d.get("term") or {}
            did, label = term.get("id"), term.get("label")
            if did and label and did not in dis:
                dis[did] = label
    return hpo, dis


def to_phenopacket(
    case: Case,
    hpo_labels: dict[str, str],
    disease_labels: dict[str, str],
    provenance: dict,
) -> dict:
    """Render one synthetic Case as a Phenopacket-schema v2 dict."""
    features = [
        {"type": {"id": t, "label": hpo_labels.get(t, "")}} for t in case.terms
    ]
    disease_term = {"id": case.omim, "label": disease_labels.get(case.omim, "")}
    return {
        "id": case.case_id,
        "subject": {"id": case.case_id},
        "phenotypicFeatures": features,
        "diseases": [{"term": disease_term}],
        # Minimal interpretation block so the causal gene survives a round-trip
        # (mirrors the structure synthvae.data._extract reads back).
        "interpretations": [
            {
                "id": case.case_id,
                "progressStatus": "SOLVED",
                "diagnosis": {
                    "disease": disease_term,
                    "genomicInterpretations": [
                        {
                            "interpretationStatus": "CAUSATIVE",
                            "variantInterpretation": {
                                "variationDescriptor": {
                                    "geneContext": {"symbol": case.gene}
                                }
                            },
                        }
                    ],
                },
            }
        ],
        "metaData": {
            "createdBy": "synthvae",
            "resources": [HPO_RESOURCE],
            "phenopacketSchemaVersion": SCHEMA_VERSION,
        },
        # Non-standard provenance block: clearly marks these as synthetic.
        "_synthvae": provenance,
    }


def build_generator(name: str, seed: int):
    if name == "vae":
        return VAEGenerator(seed=seed)
    if name == "marginal":
        return MarginalSampler(seed=seed)
    if name == "cooccurrence":
        return CooccurrenceSampler(seed=seed)
    raise ValueError(f"unknown generator: {name}")


def resolve_counts(args, train: list[Case], holdout: list[Case]) -> dict[str, int]:
    """Decide how many cases to generate per disease."""
    trainable = disease_counts(train)  # diseases the generators actually learned
    if args.per_disease is not None:
        counts = {omim: args.per_disease for omim in trainable}
    else:
        # Mirror the real holdout distribution, but only for trainable diseases.
        counts = {o: n for o, n in disease_counts(holdout).items() if o in trainable}
    if args.diseases:
        wanted = {d.strip() for d in args.diseases.split(",") if d.strip()}
        counts = {o: n for o, n in counts.items() if o in wanted}
        missing = wanted - set(counts)
        if missing:
            print(f"warning: requested diseases not in trainable corpus: {sorted(missing)}")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate synthetic phenopackets and write them to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--generator", choices=["vae", "marginal", "cooccurrence"],
                   default="vae", help="which generator to use")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: artifacts/generated/<generator>/)")
    p.add_argument("--min-cases", type=int, default=config.MIN_CASES,
                   help="keep diseases with at least this many real cases")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the split and the generator")
    p.add_argument("--epochs", type=int, default=300,
                   help="VAE training epochs (ignored for baselines)")
    p.add_argument("--per-disease", type=int, default=None,
                   help="generate this many cases for every trainable disease "
                        "(default: mirror the real holdout distribution)")
    p.add_argument("--diseases", type=str, default=None,
                   help="comma-separated OMIM ids to restrict to, e.g. OMIM:154700,OMIM:130000")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the total number of cases written (for quick smoke runs)")
    args = p.parse_args()

    out_dir = args.out or (config.ARTIFACTS / "generated" / args.generator)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading corpus from {config.STORE_DIR} (min_cases={args.min_cases}) ...")
    corpus = build_corpus(min_cases=args.min_cases)
    train, holdout = stratified_split(corpus, seed=args.seed)
    print(f"  {len(corpus)} cases, {len(set(c.omim for c in corpus))} diseases "
          f"-> {len(train)} train / {len(holdout)} holdout")

    counts = resolve_counts(args, train, holdout)
    if not counts:
        raise SystemExit("nothing to generate: no diseases matched the given filters.")
    total = sum(counts.values())
    print(f"generator={args.generator} seed={args.seed} -> "
          f"{total} cases across {len(counts)} diseases")

    gen = build_generator(args.generator, args.seed)
    if args.generator == "vae":
        print(f"training VAE ({args.epochs} epochs) ...")
        gen.fit(train, epochs=args.epochs)
    else:
        gen.fit(train)

    cases = gen.sample_cohort(counts)
    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"harvesting labels from {config.STORE_DIR} ...")
    hpo_labels, disease_labels = harvest_labels(config.STORE_DIR)

    provenance = {
        "synthetic": True,
        "generator": args.generator,
        "seed": args.seed,
        "epochs": args.epochs if args.generator == "vae" else None,
        "min_cases": args.min_cases,
    }

    written = 0
    for case in cases:
        ppkt = to_phenopacket(case, hpo_labels, disease_labels, provenance)
        fname = re.sub(r"[^A-Za-z0-9._-]", "_", case.case_id) + ".json"
        (out_dir / fname).write_text(json.dumps(ppkt, indent=1))
        written += 1

    print(f"wrote {written} phenopackets to {out_dir}")


if __name__ == "__main__":
    main()
