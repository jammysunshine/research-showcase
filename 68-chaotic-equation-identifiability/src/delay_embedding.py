"""Delay-coordinate (Takens) embedding of a single observed scalar series."""
import numpy as np


def choose_delay_by_autocorrelation(x, max_lag=200):
    """First lag at which the autocorrelation of x drops to <= 1/e (standard heuristic).

    Falls back to lag=1 if the series never drops below the threshold within
    max_lag (e.g. a near-constant or very slowly decorrelating series).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return 1
    max_lag = min(max_lag, len(x) - 2)
    threshold = 1.0 / np.e
    for lag in range(1, max_lag + 1):
        acf = np.dot(x[:-lag], x[lag:]) / denom
        if acf <= threshold:
            return lag
    return 1


def delay_embed(x, dim, tau):
    """Build an m=dim delay-coordinate embedding Y[t] = [x[t], x[t-tau], ..., x[t-(dim-1)*tau]].

    Returns an array of shape (len(x) - (dim-1)*tau, dim), row-ordered
    forward in time so Y[i+1] is the embedded state one sample after Y[i].
    """
    x = np.asarray(x, dtype=float)
    if dim == 1:
        return x.reshape(-1, 1)
    n = len(x) - (dim - 1) * tau
    if n <= 1:
        raise ValueError(f"series too short for dim={dim}, tau={tau}, len={len(x)}")
    cols = [x[(dim - 1 - k) * tau: (dim - 1 - k) * tau + n] for k in range(dim)]
    return np.column_stack(cols)
