"""Conditional VAE over HPO symptom sets.

One global model conditioned on disease (a learned disease embedding), so all diseases
share the encoder/decoder and pool statistical strength — necessary given ~12-90 cases per
disease (DESIGN.md). The latent variable is meant to capture a disease's coherent
presentation "flavors"; sampling a fresh latent yields a novel-but-coherent case.

Anti-collapse measures (the real risk here): KL warmup + free-bits floor, so the latent
isn't ignored in favour of the highly-informative disease label.

Generation matches profile size to the disease's empirical (train) size distribution, then
draws that many terms weighted by the decoder probabilities for a sampled latent — keeping
size realistic while letting the latent pick a correlated subset.
"""

from __future__ import annotations

import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Case, multihot, vocabulary, disease_labels


class CVAE(nn.Module):
    def __init__(self, n_terms: int, n_diseases: int, latent: int = 16,
                 hidden: int = 256, dis_emb: int = 32) -> None:
        super().__init__()
        self.disease_emb = nn.Embedding(n_diseases, dis_emb)
        self.enc = nn.Sequential(
            nn.Linear(n_terms + dis_emb, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent + dis_emb, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_terms),
        )

    def encode(self, x, d):
        h = self.enc(torch.cat([x, self.disease_emb(d)], dim=1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, d):
        return self.dec(torch.cat([z, self.disease_emb(d)], dim=1))  # logits

    def forward(self, x, d):
        mu, logvar = self.encode(x, d)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z, d), mu, logvar


class VAEGenerator:
    """Train wrapper + sampler that returns Case objects matched to per-disease counts."""

    def __init__(self, latent: int = 16, hidden: int = 256, dis_emb: int = 32,
                 seed: int = 0) -> None:
        self.latent, self.hidden, self.dis_emb, self.seed = latent, hidden, dis_emb, seed
        self.model: CVAE | None = None

    def fit(self, train_cases: list[Case], *, epochs: int = 300, lr: float = 1e-3,
            beta: float = 0.5, warmup: int = 50, free_bits: float = 0.1,
            verbose: bool = False) -> "VAEGenerator":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.vocab = vocabulary(train_cases)
        self.diseases, self.didx = disease_labels(train_cases)
        X = torch.tensor(multihot(train_cases, self.vocab))
        d = torch.tensor([self.didx[c.omim] for c in train_cases], dtype=torch.long)
        # train-only empirical size distribution + gene per disease
        self.sizes: dict[str, list[int]] = collections.defaultdict(list)
        self.gene: dict[str, str] = {}
        for c in train_cases:
            self.sizes[c.omim].append(len(c.terms))
            self.gene[c.omim] = c.gene

        self.model = CVAE(len(self.vocab), len(self.diseases), self.latent,
                          self.hidden, self.dis_emb)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        n = len(train_cases)
        bs = 256
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            kl_w = beta * min(1.0, ep / max(1, warmup))
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                logits, mu, logvar = self.model(X[idx], d[idx])
                recon = F.binary_cross_entropy_with_logits(
                    logits, X[idx], reduction="none").sum(1).mean()
                # per-dim KL with a free-bits floor, then summed
                kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kld = torch.clamp(kld, min=free_bits).sum(1).mean()
                loss = recon + kl_w * kld
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach())
            if verbose and (ep % 50 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  loss={tot/(n//bs+1):.2f}  kl_w={kl_w:.2f}")
        return self

    @torch.no_grad()
    def sample_cohort(self, counts: dict[str, int]) -> list[Case]:
        assert self.model is not None
        self.model.eval()
        rng = random.Random(self.seed)
        out: list[Case] = []
        for omim, k in counts.items():
            if omim not in self.didx or not self.sizes.get(omim):
                continue
            d = torch.tensor([self.didx[omim]] * k, dtype=torch.long)
            z = torch.randn(k, self.latent)
            probs = torch.sigmoid(self.model.decode(z, d)).numpy()
            for i in range(k):
                size = rng.choice(self.sizes[omim])
                p = probs[i]
                size = min(size, int((p > 0).sum()))
                if size <= 0:
                    out.append(Case(f"vae_{omim}_{i}", omim, self.gene[omim], ()))
                    continue
                # sample `size` terms without replacement, weighted by decoder probability
                chosen = np.random.choice(len(self.vocab), size=size, replace=False,
                                          p=p / p.sum())
                terms = tuple(sorted(self.vocab[j] for j in chosen))
                out.append(Case(f"vae_{omim}_{i}", omim, self.gene[omim], terms))
        return out
