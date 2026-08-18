"""Neural-network ("deep") Koopman discovery method.

Companion arm to `src/discovery_koopman.py`'s EDMD (fixed monomial
dictionary + linear least-squares). This module tests whether the EDMD
"chaos reversal" finding (DECISION_LOG.md 2026-08-17: EDMD one-step error
is WORSE in chaotic regimes than matched non-chaotic controls, contrary to
the Gallo et al. arXiv:2607.18490 prediction) is an artifact specific to
EDMD's fixed polynomial dictionary, or a more general property of
linear-in-some-representation Koopman surrogates.

Architecture: encoder MLP (state -> learned latent z), a LINEAR latent
dynamics matrix K_latent (z_t -> z_{t+1} = K_latent @ z_t, no bias, no
nonlinearity -- this is the actual "Koopman" part, mirroring EDMD's linear
K acting on dictionary space), and a decoder MLP (latent -> state).
Trained end-to-end on:
  1. one-step state prediction loss:  ||decode(K_latent @ encode(x_t)) - x_{t+1}||^2
  2. latent consistency loss:         ||K_latent @ encode(x_t) - encode(x_{t+1})||^2
  3. reconstruction loss:             ||decode(encode(x_t)) - x_t||^2
combined as a weighted sum (see DeepKoopmanModel.loss). This is the
standard "linearly-recurrent autoencoder" / deep Koopman recipe (e.g.
Lusch, Kutz & Brunton 2018; Takeishi et al. 2017) -- not degenerate to
EDMD, because encode/decode are genuinely learned nonlinear maps rather
than a fixed dictionary, and K_latent acts in a *learned* coordinate
system rather than dictionary-of-monomials space.

CPU-only (torch CPU wheel), small networks, short training -- this is a
local-only-compute laptop project (PROJECT_CHARTER.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _mlp(in_dim: int, out_dim: int, hidden: int, n_hidden_layers: int = 1) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class _KoopmanNet(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden: int, n_hidden_layers: int = 1):
        super().__init__()
        self.encoder = _mlp(state_dim, latent_dim, hidden, n_hidden_layers)
        self.decoder = _mlp(latent_dim, state_dim, hidden, n_hidden_layers)
        # Linear latent dynamics, no bias, no nonlinearity: the "Koopman" part.
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, x0: torch.Tensor):
        z0 = self.encoder(x0)
        z1_pred = self.K(z0)
        x1_pred = self.decoder(z1_pred)
        x0_recon = self.decoder(z0)
        return x1_pred, z1_pred, x0_recon, z0


@dataclass
class DeepKoopmanModel:
    """Fitted NN Koopman model. Mirrors EDMDModel's public surface
    (predict_state, simulate) so downstream scripts can compute the
    identical one-step-error metric used for EDMD.
    """

    net: _KoopmanNet
    state_dim: int
    latent_dim: int
    dt: float
    train_loss_history: list
    final_train_loss: float
    n_snapshot_pairs: int
    x_mean: np.ndarray
    x_std: np.ndarray

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def _denormalize(self, X: np.ndarray) -> np.ndarray:
        return X * self.x_std + self.x_mean

    def predict_state(self, x0: np.ndarray) -> np.ndarray:
        """One-step state predictor, matching EDMDModel.predict_state's
        calling convention: accepts a single state (1D) or a batch (2D),
        returns the corresponding shape.
        """
        x0_arr = np.atleast_2d(x0).astype(np.float64)
        xn = self._normalize(x0_arr)
        self.net.eval()
        with torch.no_grad():
            x1_pred, *_ = self.net(torch.as_tensor(xn, dtype=torch.float32))
        x1_pred = self._denormalize(x1_pred.numpy().astype(np.float64))
        return x1_pred[0] if np.atleast_2d(x0).shape[0] == 1 else x1_pred

    def simulate(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Roll out n_steps one-step predictions from x0, re-encoding the
        predicted (decoded) state at each step -- same convention as
        EDMDModel.simulate, so rollout error is directly comparable.
        """
        traj = np.empty((n_steps + 1, len(x0)))
        traj[0] = x0
        x = np.array(x0, dtype=float)
        for i in range(n_steps):
            x = self.predict_state(x)
            traj[i + 1] = x
        return traj


def fit_deep_koopman(
    states: np.ndarray,
    dt: float,
    latent_dim: int = 12,
    hidden: int = 32,
    n_hidden_layers: int = 1,
    n_epochs: int = 300,
    lr: float = 1e-3,
    lambda_recon: float = 1.0,
    lambda_latent: float = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> DeepKoopmanModel:
    """Fit a deep-Koopman (autoencoder + linear latent dynamics) model on a
    single trajectory of states sampled at fixed interval dt, via full-batch
    Adam gradient descent on the combined one-step-prediction +
    latent-consistency + reconstruction loss.

    states: (n_samples, n_vars) array, assumed evenly spaced in time
        (identical calling convention to `discovery_koopman.fit_edmd`).
    """
    torch.manual_seed(seed)
    states = np.asarray(states, dtype=float)
    n_vars = states.shape[1]

    x_mean = states.mean(axis=0)
    x_std = states.std(axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)

    Xn = (states - x_mean) / x_std
    X0 = torch.as_tensor(Xn[:-1], dtype=torch.float32)
    X1 = torch.as_tensor(Xn[1:], dtype=torch.float32)

    net = _KoopmanNet(state_dim=n_vars, latent_dim=latent_dim, hidden=hidden,
                       n_hidden_layers=n_hidden_layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    loss_history = []
    net.train()
    for epoch in range(n_epochs):
        opt.zero_grad()
        x1_pred, z1_pred, x0_recon, z0 = net(X0)
        with torch.no_grad():
            z1_target = net.encoder(X1)
        loss_pred = torch.mean((x1_pred - X1) ** 2)
        loss_latent = torch.mean((z1_pred - z1_target) ** 2)
        loss_recon = torch.mean((x0_recon - X0) ** 2)
        loss = loss_pred + lambda_latent * loss_latent + lambda_recon * loss_recon
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))
        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch}: loss={loss.item():.6g} "
                  f"(pred={loss_pred.item():.4g} latent={loss_latent.item():.4g} recon={loss_recon.item():.4g})")

    return DeepKoopmanModel(
        net=net, state_dim=n_vars, latent_dim=latent_dim, dt=dt,
        train_loss_history=loss_history, final_train_loss=loss_history[-1],
        n_snapshot_pairs=X0.shape[0], x_mean=x_mean, x_std=x_std,
    )


def fit(states: np.ndarray, dt: float, **kwargs) -> DeepKoopmanModel:
    """Alias matching discovery_koopman.fit's entry-point convention."""
    return fit_deep_koopman(states, dt=dt, **kwargs)
