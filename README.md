# synthvae — realistic synthetic phenopackets

`synthvae` generates **synthetic patient cases (phenopackets)** for benchmarking
Exomiser reranking. A small conditional VAE is trained on real cases from the
Phenopacket Store and learns the *co-occurrence structure* of HPO symptoms that
simple "pick symptoms independently" generators (like phenotype2phenopacket)
miss. The result: synthetic cases that are **realistic** (hard to tell apart from
real ones) yet **novel** (not memorized copies), so an LLM can't just look up the
answer. See `DESIGN.md` and `RESULTS.md` for the method and numbers.

## Setup

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <this repo>
cd synthetic_vae_generator
uv sync          # installs synthvae + deps, including torch (pinned in uv.lock)
```

## Data you must provide

The generators train on the **Phenopacket Store bundle** that the sibling
`exomiser-rerank-benchmark` repo downloads. Nothing here ships that data — you
point `synthvae` at an existing checkout.

By default (`synthvae/config.py`) it looks under
`~/PythonProject/exomiser-rerank-benchmark/`:

| Env var | Default | Needed for generation? |
|---|---|---|
| `SYNTHVAE_BENCH_REPO` | `~/PythonProject/exomiser-rerank-benchmark` | base for the paths below |
| `SYNTHVAE_STORE_DIR` | `$BENCH_REPO/data/ppkts` (`<release>/<GENE>/*.json`) | **yes** — the training cases |
| `SYNTHVAE_GENES_TO_DISEASE` | `$BENCH_REPO/data/genes_to_disease.txt` | **yes** — unambiguous ground truth |
| `SYNTHVAE_HP_OBO` | `$BENCH_REPO/data/hp.obo` | no (not read during generation) |
| `SYNTHVAE_ARTIFACTS` | `./artifacts` | output dir default |

If your checkout lives elsewhere, point the repo root at it:

```bash
export SYNTHVAE_BENCH_REPO=/path/to/exomiser-rerank-benchmark
```

or override individual paths (`SYNTHVAE_STORE_DIR`, `SYNTHVAE_GENES_TO_DISEASE`).

## Generate cases (the main path)

```bash
# VAE (recommended). Mirrors the real holdout's per-disease counts by default.
uv run python scripts/generate.py --generator vae --out out/

# Fixed number of cases per disease:
uv run python scripts/generate.py --generator vae --per-disease 5 --out out/

# Quick smoke run (instant baseline, capped output):
uv run python scripts/generate.py --generator marginal --per-disease 2 --limit 10 --out out/
```

Each generated case is written as one **Phenopacket-schema v2 JSON** file in the
output dir (default `artifacts/generated/<generator>/`). The packet carries the
HPO `phenotypicFeatures` (with labels), the ground-truth `diseases` (OMIM) and
causal gene in `interpretations`, and a `_synthvae` provenance block marking it
synthetic. Example file `vae_OMIM:103580_0.json`:

```json
{
 "id": "vae_OMIM:103580_0",
 "phenotypicFeatures": [{"type": {"id": "HP:0004322", "label": "Short stature"}}],
 "diseases": [{"term": {"id": "OMIM:103580", "label": "Pseudohypoparathyroidism Ia"}}],
 "interpretations": [{"diagnosis": {"genomicInterpretations": [
   {"variantInterpretation": {"variationDescriptor": {"geneContext": {"symbol": "GNAS"}}}}]}}],
 "_synthvae": {"synthetic": true, "generator": "vae", "seed": 0}
}
```

Useful flags: `--generator {vae,marginal,cooccurrence}`, `--per-disease N`,
`--diseases OMIM:103580,OMIM:122470` (restrict to specific diseases),
`--min-cases N` (disease eligibility threshold, default 20), `--epochs N` (VAE
training epochs, default 300), `--seed N`, `--limit N`. Run with `-h` for the
full list. A full VAE run over all ~99 eligible diseases takes ~20s on CPU.

## Programmatic API

To generate inside Python or a notebook instead of the CLI:

```python
from synthvae.data import build_corpus, stratified_split
from synthvae.eval import disease_counts
from synthvae.vae import VAEGenerator           # or MarginalSampler / CooccurrenceSampler

corpus = build_corpus(min_cases=20)             # load + filter the store
train, holdout = stratified_split(corpus, seed=0)

gen = VAEGenerator(seed=0).fit(train, epochs=300)
counts = disease_counts(holdout)                # {"OMIM:103580": 5, ...}
cases = gen.sample_cohort(counts)               # list[Case]
```

A `Case` is a frozen dataclass `(case_id, omim, gene, terms)`, where `terms` is a
tuple of bare HPO ids (e.g. `("HP:0004322", ...)`) — labels are not carried on
the `Case` (the CLI fills them back in from the store when writing phenopackets).
Baselines share the same `.fit(train).sample_cohort(counts)` interface.

## Background / how it works

- `synthvae_writeup.ipynb` — end-to-end narrative: the three generators, the PMI
  realism detector, the novelty (anti-memorization) check, and a robustness sweep.
  Run with `uv run jupyter notebook synthvae_writeup.ipynb`.
  Note: the notebook contains *inline copies* of the generators for a
  self-contained read; the package in `synthvae/` is the canonical implementation
  the CLI and the API above use.
- `DESIGN.md`, `RESULTS.md` — design rationale and headline numbers (VAE is harder
  to distinguish from real cases than the baselines, and generates novel rather
  than memorized combinations).

## Troubleshooting

- **`FileNotFoundError: no phenopackets under …`** — the store path is wrong or
  empty. Set `SYNTHVAE_STORE_DIR` (or `SYNTHVAE_BENCH_REPO`) to your
  `exomiser-rerank-benchmark` checkout.
- **VAE is slow / no GPU** — fine; it trains on CPU in ~10–30s. Use
  `--epochs 100` for a faster run, or `--generator marginal` for an instant
  baseline.
