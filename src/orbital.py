"""Vectorized two-body conversions (heliocentric, elliptic orbits). Angles in radians."""
from __future__ import annotations

import numpy as np


def solve_kepler(M: np.ndarray, e: np.ndarray, tol: float = 1e-14, max_iter: int = 60) -> np.ndarray:
    """Eccentric anomaly E from mean anomaly M (elliptic, e < 1). Robust to e -> 1."""
    M = np.mod(np.asarray(M, float) + np.pi, 2 * np.pi) - np.pi
    e = np.asarray(e, float)
    E = M + 0.85 * e * np.sign(np.sin(M))                  # Danby starter
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        fpp = e * np.sin(E)
        dE = -f / fp
        dE = -f / (fp + 0.5 * dE * fpp)                    # Halley step
        E = E + dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def elements_to_cartesian(a, e, inc, Omega, omega, M, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (r, v) arrays of shape (n, 3)."""
    a, e, inc, Omega, omega, M = (np.atleast_1d(np.asarray(x, float)) for x in (a, e, inc, Omega, omega, M))
    E = solve_kepler(M, e)
    cE, sE = np.cos(E), np.sin(E)
    b = np.sqrt(1 - e ** 2)
    x_p, y_p = a * (cE - e), a * b * sE
    rn = a * (1 - e * cE)
    fac = np.sqrt(mu * a) / rn
    vx_p, vy_p = -fac * sE, fac * b * cE
    cO, sO, co, so, ci, si = np.cos(Omega), np.sin(Omega), np.cos(omega), np.sin(omega), np.cos(inc), np.sin(inc)
    P = np.stack([cO * co - sO * so * ci, sO * co + cO * so * ci, so * si], axis=-1)
    Q = np.stack([-cO * so - sO * co * ci, -sO * so + cO * co * ci, co * si], axis=-1)
    r = x_p[:, None] * P + y_p[:, None] * Q
    v = vx_p[:, None] * P + vy_p[:, None] * Q
    return r, v


def cartesian_to_aei(r: np.ndarray, v: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semi-major axis, eccentricity, inclination [rad] from heliocentric state (n, 3)."""
    rn = np.linalg.norm(r, axis=-1)
    v2 = np.einsum("ij,ij->i", v, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = 1.0 / (2.0 / rn - v2 / mu)
        h = np.cross(r, v)
        hn = np.linalg.norm(h, axis=-1)
        evec = np.cross(v, h) / mu - r / rn[:, None]
        e = np.linalg.norm(evec, axis=-1)
        inc = np.arccos(np.clip(h[:, 2] / hn, -1, 1))
    return a, e, inc


def periapsis_distance(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    """Two-body periapsis distance q = h^2 / (mu (1+e)); valid for elliptic and hyperbolic."""
    h = np.cross(r, v)
    rn = np.linalg.norm(r, axis=-1)
    e = np.linalg.norm(np.cross(v, h) / mu - r / rn[:, None], axis=-1)
    return np.einsum("ij,ij->i", h, h) / (mu * (1 + e))


def cartesian_to_elements(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    """Osculating elliptic elements (n, 6): a, e, inc, Omega, omega, M [rad] from states (n, 3).

    Inverse of ``elements_to_cartesian`` for bound orbits.  Degenerate angles are fixed by
    convention: for inc ~ 0 the node is set to 0 and omega is measured from +x.
    """
    r, v = np.atleast_2d(r).astype(float), np.atleast_2d(v).astype(float)
    rn = np.linalg.norm(r, axis=1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    evec = np.cross(v, h) / mu - r / rn[:, None]
    e = np.linalg.norm(evec, axis=1)
    a = 1.0 / (2.0 / rn - np.einsum("ij,ij->i", v, v) / mu)
    inc = np.arccos(np.clip(h[:, 2] / hn, -1, 1))
    nvec = np.stack([-h[:, 1], h[:, 0], np.zeros(len(h))], axis=1)       # z x h
    nn = np.linalg.norm(nvec, axis=1)
    flat = nn < 1e-12 * hn
    Omega = np.where(flat, 0.0, np.arctan2(nvec[:, 1], nvec[:, 0]))
    # argument of perihelion: angle from the node to e, in the orbital plane
    hhat = h / hn[:, None]
    nhat = np.where(flat[:, None], np.array([1.0, 0.0, 0.0]), nvec / np.where(nn == 0, 1, nn)[:, None])
    omega = np.arctan2(np.einsum("ij,ij->i", np.cross(nhat, evec), hhat), np.einsum("ij,ij->i", nhat, evec))
    E = np.arctan2(np.einsum("ij,ij->i", r, v) / np.sqrt(mu * a), 1.0 - rn / a)
    M = E - e * np.sin(E)
    wrap = lambda x: np.mod(x, 2 * np.pi)
    return np.column_stack([a, e, inc, wrap(Omega), wrap(omega), wrap(M)])
