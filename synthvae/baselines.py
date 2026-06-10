"""Baseline generators.

MarginalSampler is the phenotype2phenopacket mechanism made in-process: per disease it
draws a profile size from the (train-only) empirical size distribution, then samples that
many terms *independently*, weighted by each term's per-disease frequency. It deliberately
ignores co-occurrence — it is the "ceiling" arm the VAE must beat and the resolution probe
for the C2ST ruler.

All statistics are fit on TRAIN cases only (no holdout leakage).
"""

from __future__ import annotations

import collections
import random

import numpy as np

from .data import Case


class MarginalSampler:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.sizes: dict[str, list[int]] = {}
        self.terms: dict[str, list[str]] = {}
        self.weights: dict[str, list[float]] = {}
        self.gene: dict[str, str] = {}

    def fit(self, train_cases: list[Case]) -> "MarginalSampler":
        by_disease: dict[str, list[Case]] = collections.defaultdict(list)
        for c in train_cases:
            by_disease[c.omim].append(c)
        for omim, group in by_disease.items():
            self.sizes[omim] = [len(c.terms) for c in group]
            freq = collections.Counter(t for c in group for t in c.terms)
            self.terms[omim] = list(freq)
            self.weights[omim] = [freq[t] for t in self.terms[omim]]
            self.gene[omim] = group[0].gene
        return self

    def sample_one(self, omim: str, idx: int) -> Case:
        size = self.rng.choice(self.sizes[omim])
        pool, wts = self.terms[omim], self.weights[omim]
        size = min(size, len(pool))
        # weighted sampling without replacement
        chosen: set[str] = set()
        pw = list(zip(pool, wts))
        while len(chosen) < size and pw:
            terms, weights = zip(*pw)
            pick = self.rng.choices(range(len(pw)), weights=weights, k=1)[0]
            chosen.add(pw[pick][0])
            pw.pop(pick)
        return Case(f"marginal_{omim}_{idx}", omim, self.gene[omim], tuple(sorted(chosen)))

    def sample_cohort(self, counts: dict[str, int]) -> list[Case]:
        """Generate a cohort with the given per-disease case counts."""
        out: list[Case] = []
        for omim, n in counts.items():
            if omim not in self.sizes:
                continue
            for i in range(n):
                out.append(self.sample_one(omim, i))
        return out


class CooccurrenceSampler:
    """Per-disease Chow-Liu tree model: the simplest generator that captures *pairwise*
    symptom structure (vs MarginalSampler's independent draws). It builds the maximum-mutual-
    information spanning tree over symptoms, then samples each symptom conditional on its
    parent. By construction it reproduces marginals AND tree-edge pairwise statistics, so it
    should drive the (pairwise) PMI detector toward the floor — the test of whether a VAE's
    extra complexity is even needed for pairwise realism.

    Fit on TRAIN cases only. Laplace-smoothed; terms below min_count are dropped (too noisy).
    """

    def __init__(self, seed: int = 0, min_count: int = 2, alpha: float = 1.0) -> None:
        self.rng = random.Random(seed)
        self.min_count = min_count
        self.alpha = alpha
        self.models: dict[str, dict] = {}
        self.gene: dict[str, str] = {}

    def fit(self, train_cases: list[Case]) -> "CooccurrenceSampler":
        by_disease: dict[str, list[set]] = collections.defaultdict(list)
        for c in train_cases:
            by_disease[c.omim].append(set(c.terms))
            self.gene[c.omim] = c.gene
        for omim, sets in by_disease.items():
            self.models[omim] = self._fit_tree(sets)
        return self

    def _fit_tree(self, sets: list[set]) -> dict:
        n = len(sets)
        counts = collections.Counter(t for s in sets for t in s)
        terms = sorted(t for t, k in counts.items() if k >= self.min_count)
        m = len(terms)
        if m == 0:
            return {"terms": [], "sizes": [len(s) for s in sets]}
        idx = {t: i for i, t in enumerate(terms)}
        X = np.zeros((n, m), dtype=np.float64)
        for r, s in enumerate(sets):
            for t in s:
                if t in idx:
                    X[r, idx[t]] = 1.0
        a = self.alpha
        p1 = (X.sum(0) + a) / (n + 2 * a)  # smoothed marginal P(term=1)
        # pairwise mutual information over the complete graph
        mi = np.zeros((m, m))
        joint11 = (X.T @ X)  # co-occurrence counts
        for i in range(m):
            for j in range(i + 1, m):
                c11 = joint11[i, j]
                c10 = X[:, i].sum() - c11
                c01 = X[:, j].sum() - c11
                c00 = n - c11 - c10 - c01
                jt = (np.array([c00, c01, c10, c11]) + a) / (n + 4 * a)
                pi = np.array([1 - p1[i], p1[i]])
                pj = np.array([1 - p1[j], p1[j]])
                outer = np.array([pi[0]*pj[0], pi[0]*pj[1], pi[1]*pj[0], pi[1]*pj[1]])
                val = float((jt * np.log(jt / outer)).sum())
                mi[i, j] = mi[j, i] = val
        # maximum spanning tree (Prim) over MI -> parent of each node
        parent = [-1] * m
        in_tree = [False] * m
        best = mi[0].copy()
        best_from = [0] * m
        in_tree[0] = True
        order = [0]
        for _ in range(m - 1):
            cand = -1
            cbest = -1.0
            for v in range(m):
                if not in_tree[v] and best[v] > cbest:
                    cbest = best[v]
                    cand = v
            if cand == -1:
                break
            in_tree[cand] = True
            parent[cand] = best_from[cand]
            order.append(cand)
            for v in range(m):
                if not in_tree[v] and mi[cand, v] > best[v]:
                    best[v] = mi[cand, v]
                    best_from[v] = cand
        # conditional P(child=1 | parent=v) along tree edges
        cond = {}
        for child in range(m):
            par = parent[child]
            if par == -1:
                continue
            c11 = float(joint11[child, par])
            par_n = float(X[:, par].sum())
            p_c1_given_p1 = (c11 + a) / (par_n + 2 * a)
            c_child_no_par = float(X[:, child].sum()) - c11
            p_c1_given_p0 = (c_child_no_par + a) / (n - par_n + 2 * a)
            cond[child] = (p_c1_given_p0, p_c1_given_p1)
        return {
            "terms": terms, "order": order, "parent": parent,
            "p_root": float(p1[order[0]]), "cond": cond,
        }

    def sample_one(self, omim: str, idx: int) -> Case:
        mdl = self.models[omim]
        terms = mdl["terms"]
        if not terms:
            return Case(f"cooc_{omim}_{idx}", omim, self.gene[omim], ())
        val = [0] * len(terms)
        for node in mdl["order"]:
            par = mdl["parent"][node]
            if par == -1:
                p = mdl["p_root"]
            else:
                p = mdl["cond"][node][val[par]]
            val[node] = 1 if self.rng.random() < p else 0
        chosen = tuple(sorted(terms[i] for i in range(len(terms)) if val[i]))
        return Case(f"cooc_{omim}_{idx}", omim, self.gene[omim], chosen)

    def sample_cohort(self, counts: dict[str, int]) -> list[Case]:
        out: list[Case] = []
        for omim, n in counts.items():
            if omim not in self.models:
                continue
            for i in range(n):
                out.append(self.sample_one(omim, i))
        return out
