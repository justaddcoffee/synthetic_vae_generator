"""Robustness sweep: is the VAE's realism advantage stable across seeds and corpus size N?

For each N in {20,30,50}: build the eligible corpus, stratified split, then report the PMI
detector AUC (vs real holdout; floor = real-vs-real) for the marginal and Chow-Liu baselines
and for the VAE across several training seeds (mean +/- sd). Lower AUC = more realistic.
"""

from __future__ import annotations

import collections
import random
import statistics as st

import numpy as np

from synthvae import data, eval as ev
from synthvae.baselines import MarginalSampler, CooccurrenceSampler
from synthvae.vae import VAEGenerator


def floor_auc(detector, holdout, seed=0):
    rng = random.Random(seed)
    bd = collections.defaultdict(list)
    for c in holdout:
        bd[c.omim].append(c)
    a, b = [], []
    for g in bd.values():
        gg = g[:]; rng.shuffle(gg)
        a += gg[: len(gg) // 2]; b += gg[len(gg) // 2 :]
    return detector.auc(a, b)


def gap_closed(auc, marg, floor):
    return 100 * (marg - auc) / (marg - floor) if marg > floor else float("nan")


VAE_SEEDS = [0, 1, 2]

print(f"{'N':>3} {'diseases':>8} {'cases':>6} | {'floor':>6} {'marginal':>8} {'ChowLiu':>8} "
      f"| {'VAE mean':>9} {'VAE sd':>6} {'gap%':>5}")
for N in [20, 30, 50]:
    cases = data.build_corpus(min_cases=N)
    train, holdout = data.stratified_split(cases)
    det = ev.CooccurrenceDetector().fit(train)
    counts = ev.disease_counts(holdout)
    n_dis = len({c.omim for c in cases})

    floor = floor_auc(det, holdout)
    marg = det.auc(holdout, MarginalSampler(seed=0).fit(train).sample_cohort(counts))
    cooc = det.auc(holdout, CooccurrenceSampler(seed=0).fit(train).sample_cohort(counts))

    vae_aucs = []
    for s in VAE_SEEDS:
        gen = VAEGenerator(seed=s).fit(train, epochs=300)
        vae_aucs.append(det.auc(holdout, gen.sample_cohort(counts)))
    vmean = st.mean(vae_aucs)
    vsd = st.pstdev(vae_aucs)
    print(f"{N:>3} {n_dis:>8} {len(cases):>6} | {floor:>6.3f} {marg:>8.3f} {cooc:>8.3f} "
          f"| {vmean:>9.3f} {vsd:>6.3f} {gap_closed(vmean, marg, floor):>5.0f}")
