"""Realism evaluation: the classifier two-sample test (C2ST).

A classifier is trained to separate one arm of cases ("A", e.g. synthetic) from another
("B", e.g. real holdout) using their multi-hot HPO vectors. We report cross-validated ROC
AUC. Interpretation (DESIGN.md):

    AUC ~ 0.5  -> arms indistinguishable (good — looks real)
    AUC -> 1.0 -> easily separable (unrealistic)

Always read an arm's AUC against two references computed the same way:
    floor   = real vs real      (split holdout in half)  -> the achievable noise floor
    ceiling = p2p/marginal vs real                        -> the baseline to beat

To stop the classifier winning on bookkeeping rather than realism, the compared arms must
share their per-disease case counts (match the real arm's counts when generating), and the
multi-hot uses a shared vocabulary.
"""

from __future__ import annotations

import collections
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .data import Case, multihot, vocabulary


def disease_counts(cases: list[Case]) -> dict[str, int]:
    return dict(collections.Counter(c.omim for c in cases))


def c2st_auc(
    arm_a: list[Case],
    arm_b: list[Case],
    vocab: list[str] | None = None,
    *,
    seed: int = 0,
    n_splits: int = 5,
) -> float:
    """Cross-validated ROC AUC for separating arm_a (label 1) from arm_b (label 0).
    A shared vocabulary is built from both arms if not supplied."""
    vocab = vocab or vocabulary(arm_a + arm_b)
    xa, xb = multihot(arm_a, vocab), multihot(arm_b, vocab)
    X = np.vstack([xa, xb])
    y = np.concatenate([np.ones(len(arm_a)), np.zeros(len(arm_b))])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    # ROC AUC from out-of-fold scores
    return _auc(y, oof)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, score))


class CooccurrenceDetector:
    """Co-occurrence-aware realism detector (the instrument that actually has resolution).

    A plain classifier on raw symptom lists barely separates an independent-draw baseline
    from real cases (AUC ~0.57) because it must rediscover which symptom-pairs matter from
    little data. This detector instead looks DIRECTLY at co-occurrence: from real TRAIN cases
    it learns each disease's pairwise pointwise-mutual-information (PMI, high when two symptoms
    co-occur more than chance), then scores a case by the mean PMI of its present symptom-pairs.
    Real cases contain genuine high-PMI pairs; independent-draw fakes contain fewer.

    Calibrated: real-vs-real ~0.53 (floor), independent-draw-vs-real ~0.745 (ceiling).
    A good generator should pull its AUC down toward the floor.
    """

    def __init__(self, smoothing: float = 1e-3) -> None:
        self.smoothing = smoothing
        self.tab: dict[str, tuple[int, collections.Counter, collections.Counter]] = {}

    def fit(self, train_cases: list[Case]) -> "CooccurrenceDetector":
        by_disease: dict[str, list[set]] = collections.defaultdict(list)
        for c in train_cases:
            by_disease[c.omim].append(set(c.terms))
        for omim, sets in by_disease.items():
            uni: collections.Counter = collections.Counter()
            pair: collections.Counter = collections.Counter()
            for s in sets:
                for t in s:
                    uni[t] += 1
                sl = sorted(s)
                for i in range(len(sl)):
                    for j in range(i + 1, len(sl)):
                        pair[(sl[i], sl[j])] += 1
            self.tab[omim] = (len(sets), uni, pair)
        return self

    def score(self, c: Case) -> float:
        if c.omim not in self.tab:
            return 0.0
        n, uni, pair = self.tab[c.omim]
        sl = sorted(set(c.terms))
        eps = self.smoothing
        vals = []
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                a, b = sl[i], sl[j]
                pa, pb = uni.get(a, 0) / n, uni.get(b, 0) / n
                if pa > 0 and pb > 0:
                    pab = pair.get((a, b), 0) / n
                    vals.append(math.log((pab + eps) / (pa * pb + eps)))
        return float(np.mean(vals)) if vals else 0.0

    def auc(self, real_arm: list[Case], other_arm: list[Case]) -> float:
        """Pooled, rank-within-disease AUC separating real_arm (1) from other_arm (0).
        Rank-normalizing within disease makes per-disease PMI scales comparable when pooled."""
        from scipy.stats import rankdata

        bd_r: dict[str, list[float]] = collections.defaultdict(list)
        bd_o: dict[str, list[float]] = collections.defaultdict(list)
        for c in real_arm:
            bd_r[c.omim].append(self.score(c))
        for c in other_arm:
            bd_o[c.omim].append(self.score(c))
        ys, ss = [], []
        for omim in bd_r:
            r, o = bd_r[omim], bd_o.get(omim, [])
            if len(r) < 4 or len(o) < 4:
                continue
            ss.append(rankdata(np.r_[r, o]) / (len(r) + len(o)))
            ys.append(np.r_[np.ones(len(r)), np.zeros(len(o))])
        return float(roc_auc_score(np.concatenate(ys), np.concatenate(ss)))


def real_vs_real_floor(holdout: list[Case], *, seed: int = 0, n_splits: int = 5) -> float:
    """Split the real holdout in half (stratified by disease) and run C2ST — the noise floor."""
    import random

    rng = random.Random(seed)
    by_disease: dict[str, list[Case]] = collections.defaultdict(list)
    for c in holdout:
        by_disease[c.omim].append(c)
    a: list[Case] = []
    b: list[Case] = []
    for group in by_disease.values():
        g = group[:]
        rng.shuffle(g)
        a.extend(g[: len(g) // 2])
        b.extend(g[len(g) // 2 :])
    return c2st_auc(a, b, seed=seed, n_splits=n_splits)
