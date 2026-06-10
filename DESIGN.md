# Synthetic Phenopacket Generator — Design (v1, brainstorm)

## Goal
Generate **realistic** synthetic patient phenopackets to benchmark **OpenScientist's
reranking of Exomiser output**. Requirements:
1. **Known ground truth** — each case anchored to a causal gene/disease so we can score
   rank-recovery.
2. **Phenotypically realistic** — captures the *joint* structure of real cases (which HPO
   terms co-occur), not just marginal term frequencies.
3. **Uncheatable** — novel enough that an LLM reranker can't retrieve a matching published
   case report from PubMed and read off the diagnosis.

## Problem with the current approach (phenotype2phenopacket / pheval)
`exomiser-rerank-benchmark/benchmark/synthesize.py` draws HPO terms **independently**,
frequency-weighted from HPOA aggregate annotations, plus uniform-random distractor "noise"
terms. This gets *marginal* term frequencies right but breaks *co-occurrence* structure,
producing clinically incoherent symptom combinations.

## Scope (v1)
- **Phenotypes only.** Generator outputs an HPO term set; the existing variant-spiking
  pipeline (`add-genes` + VCF spiking) attaches the causal variant. Variant synthesis is a
  separate future module.
- **Demographics (sex, age-of-onset) NOT modeled in the VAE.** Attached downstream by
  empirical conditional sampling from real cases of the same disease. (Realism eval runs on
  HPO terms only, so modeling demographics adds complexity without contributing to the claim.)

## Data
- Source: **phenopacket-store v0.1.26** (9,588 packets, 694 OMIM diseases, 622 genes,
  4,031-term HPO vocabulary; profile size median 7, mean 8.7, max 66).
- Per-disease density is heavily skewed: 83 diseases have 1 case; only 104 have >=20 cases
  (6,116 packets), 71 have >=30, 39 have >=50.
- **Filter to diseases with >=N cases AND a single causal gene** (single gene in store *and*
  a single `genes_to_disease` mapping — required so rank-scoring has unambiguous ground truth).
  Empirically the single-gene rule is nearly free: **99.3% of store cases are single-gene**.
  Primary **N=20** → **99 eligible diseases / 5,890 cases** (single-store only: 103 / 6,047).
  Treat N as a sensitivity knob (N=30 → 68 diseases). 
- **Split: 80% train / 20% held-out real, STRATIFIED PER DISEASE** (every included disease
  appears in both train and holdout). All empirical distributions used by generation
  (profile-size, marginals) must be computed **train-only** to avoid holdout leakage.

## Empirical findings (de-risking experiments, run on the eligible corpus)
- **Premise validated:** real cases carry co-occurrence structure well beyond per-disease term
  marginals. Within-disease term-pair strong-correlation rate is **11.9% observed vs 3.3%
  under a permutation independence null (3.6x excess, 95/98 diseases)**. This is exactly the
  structure p2p's independent draws discard → a joint generative model has real signal to learn.
- **Profile size (most-specific terms):** median 7, mean 8.0, p90 15, max 66.
- **Exact-duplicate profiles: 16.7%** — many identical term sets (small profiles + recurrent
  presentations). Matters for the novelty/memorization guard and for interpreting any
  similarity-to-nearest metric.
- NOT yet shown: that a VAE beats a simple co-occurrence model at capturing this structure —
  that head-to-head is the next experiment.

## Model
- **One global Conditional VAE (CVAE)**, NOT per-disease — shares statistical strength across
  diseases so it tolerates N=20 and uses the long tail. (Per-disease VAE would need N>=50 and
  discard ~60% of data.)
- **Input representation:** binary multi-hot over HPO vocab, **ancestor-propagated** (turn on
  all is-a ancestors of each annotated term). Encodes ontology structure and densifies input
  from ~7 to ~30-50 active bits. Collapse to most-specific terms at generation.
- **Conditioning:** learned **disease embedding** (not one-hot) fed to encoder and decoder.
- **Network:** MLP encoder/decoder, latent dim ~16-32. Loss = binary cross-entropy + KL.
- **Profile-size control:** sample target size from the disease's empirical size distribution,
  then draw that many terms from decoder probabilities (keeps size realistic AND variable).
- **Coherence cleanup:** enforce ontology consistency post-sampling (drop redundant ancestors,
  resolve contradictions) so output is a valid phenopacket.
- **Also build a simple learned baseline** (e.g. co-occurrence-corrected sampler) sitting
  between p2p and the VAE, to test whether the VAE's complexity actually buys realism.

## Evaluation — realism
Three arms, all conditioned on the **same diseases**: (1) **synthetic** (CVAE), (2) **real
holdout** (the 20% never seen), (3) **p2p baseline**.

**LESSON (empirical, do not repeat the mistake):** a generic classifier (HistGradientBoosting)
on raw multi-hot term vectors is a WEAK realism detector — it separates an independent-draw
baseline from real cases at only **AUC ~0.57** (barely above 0.5), because it must rediscover
which symptom-pairs matter from ~12 cases/disease. Disease-aware and within-disease variants
were *worse* (0.55, 0.52). This is NOT evidence that fakes look real; it is the wrong
instrument. A detector aimed directly at co-occurrence (per-disease pairwise PMI, learned from
TRAIN real cases, scoring a case by the mean PMI of its present symptom-pairs) separates the
SAME baseline from real at **AUC 0.745**, with a calibrated **real-vs-real floor of 0.53**.
=> Use the **co-occurrence (PMI) detector** (`eval.CooccurrenceDetector`) as the primary
realism metric. ~0.21 AUC of headroom (0.745 -> 0.53) for the VAE to demonstrate improvement.

**Primary metric: co-occurrence (PMI) detector.** Report its AUC for each arm vs real holdout.
- Calibrate with a **floor** (real-vs-real, split holdout in half -> AUC ~0.5) and a
  **ceiling** (p2p-vs-real). Win = synthetic AUC near the floor and well below the p2p ceiling.
- The classifier doubles as a **diagnostic**: inspect what it uses to separate (set size, an
  unused term, an over-frequent pair) and fix the generator, re-test, watch AUC drop.
- **Control confounds:** match set-size distribution and disease mix across all three arms so
  the classifier wins on realism, not bookkeeping.
- Chose C2ST over embedding-distance tests because pooling per-term embeddings attenuates the
  co-occurrence signal (lossy linear projection); raw vectors + a nonlinear classifier keep it
  explicit. Embedding kept only for UMAP visualization, not as the measurement.

**Novelty guard (anti-cheat):** synthetic cases must NOT be near-duplicates of training cases
(e.g. semantic similarity to nearest *training* case must not pile up near 1.0). This is the
direct "can't just look it up" evidence.

## Integration
Replaces the "draw terms from HPOA" step in `synthesize.py`; downstream `add-genes` + VCF
spiking and the scoring harness are unchanged. New project lives in `synthetic_vae_generator/`.

## RESULTS (first end-to-end run, single seed, N=20 corpus)
Realism = PMI co-occurrence detector AUC vs real holdout (floor 0.532; lower = more realistic):
- marginal / independent draws (p2p-style):  0.745   (0% of gap closed)
- Chow-Liu simple pairwise model:            0.609   (~64% closed)
- **CVAE:                                    0.568   (~83% closed)** — beats the simple model.
Novelty (nearest TRAIN-case Jaccard, same disease; lower = more novel):
- real holdout: mean 0.642, 21.9% exact copies of a train case (real cases are repetitive).
- **CVAE: mean 0.545, 9.2% exact copies** — MORE novel than real cases are to each other.
=> The VAE is both more realistic than the simple baselines AND not memorizing. It earns its
complexity on the pairwise metric. Caveats: pairwise metric only (higher-order / mode-coverage
checks pending); downstream Exomiser utility not yet tested.

Robustness sweep (scripts/robustness.py; PMI-AUC vs real, lower=better; VAE mean+/-sd over 3 seeds):
  N    diseases cases | floor  marginal ChowLiu | VAE         gap%
  20   99       5888  | 0.532  0.745    0.609    | 0.575+/-.007  80
  30   68       5142  | 0.509  0.704    0.596    | 0.583+/-.007  62
  50   37       3992  | 0.460  0.686    0.583    | 0.561+/-.012  55
- VAE seed-stable (sd <= 0.012) and beats BOTH baselines at every N. Ordering is consistent.
- gap% falls at high N mainly because the real-vs-real FLOOR estimate gets noisy with few
  diseases (N=50 floor 0.46 < 0.5). VAE absolute AUC is best at N=50 (0.561). N=20 = sweet spot
  (most diseases, cleanest floor). The floor-noise is a metric limitation at small disease
  counts, not a VAE problem.

## Revised phasing (Codex review)
1. ~~Eligibility audit (single-gene + N)~~ DONE — 99 diseases / 5,890 cases viable.
2. ~~Co-occurrence structure check~~ DONE — premise validated (3.6x over independence).
3. Data prep: build the eligible corpus + stratified train/holdout split; the C2ST "ruler"
   + target-distribution stats (the reusable eval harness).
4. Simple baselines: p2p (existing) + a train-only disease-conditioned co-occurrence sampler.
5. VAE: build it, evaluate head-to-head vs the simple baseline on the ruler.
6. Downstream-utility eval: run generated cases through the existing Exomiser harness; confirm
   hard-but-fair (true gene present, baseline rank not trivial/hopeless, reranker headroom),
   and that difficulty resembles real holdout more than p2p.

## Open questions / risks to probe
- Is a CVAE the right tool given small, sparse binary data, or is a simpler co-occurrence model
  competitive? (Hence the baseline.)
- Posterior collapse / memorization risk on a ~6k-case corpus.
- Does ancestor-propagation help or just inflate dimensionality?
- Is the per-disease holdout (~4 cases at N=20) enough for per-disease analysis, or only pooled?
- Does C2ST with a powerful classifier always win, making the "gap to floor" the only honest
  readout? (We accept comparative-to-baseline framing.)
