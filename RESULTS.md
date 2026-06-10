# VAE for realistic synthetic phenopackets — first results

*A one-page summary. The full, runnable story (code + plots) is in
[`synthvae_writeup.ipynb`](synthvae_writeup.ipynb). Method/theory details in [`DESIGN.md`](DESIGN.md).*

## The problem
We want synthetic patient cases to benchmark Exomiser reranking that are **realistic** and
**novel** (so an AI can't just look up the answer). The standard tool (phenotype2phenopacket)
picks symptoms **independently by frequency**, which breaks the way real symptoms *co-occur*.
We checked: real cases really do have co-occurrence structure — symptom pairs cluster **~3.5×
more than chance** (permutation test). So there's something a better generator can capture.

## What we did
Trained a small **conditional VAE** on real cases (Phenopacket Store: 99 single-gene diseases,
5,888 cases, 80/20 train/holdout). Compared it to two baselines and judged realism with a
detector aimed at co-occurrence (it scores how "real" a case's symptom pairs look; **AUC ~0.5 =
indistinguishable from real, higher = easy to spot as fake**).

## Result — the VAE looks most real

| Generator | Realism (AUC vs real; lower = better) |
|---|---|
| Floor (real vs real) | **0.53** |
| Marginal / independent draws (p2p-style) | 0.75 |
| Chow-Liu (simple pairwise model) | 0.61 |
| **VAE** | **0.57** |

The VAE is hardest to tell apart from real cases and lands closest to the floor — beating even
the pairwise model. It's **stable** across seeds (sd ≤ 0.01) and corpus sizes (N=20/30/50).

## Result — and it isn't memorizing
Generated cases are **less** similar to the training data than real held-out cases are
(mean overlap 0.55 vs 0.64; exact copies 9% vs 22%). So the VAE makes **novel** combinations —
good for the "can't look it up" goal.

## What this does NOT yet show (please push on these)
- **Pairwise only** — realism is measured on symptom *pairs*; higher-order structure and mode
  coverage aren't checked yet.
- **One metric** — we should add an independent second ruler.
- **Benchmark utility untested** — we haven't shown these cases make Exomiser reranking
  *hard-but-fair*. This is the most important next experiment.
- **Anti-cheat is indirect** — lower training overlap is encouraging, but we haven't tested
  whether an LLM can still retrieve the diagnosis.

## To run
```bash
uv run jupyter notebook synthvae_writeup.ipynb   # needs the Phenopacket Store bundle; see cell 0
```
