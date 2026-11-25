"""Offline planetary initial conditions.

Source: E.M. Standish, "Keplerian Elements for Approximate Positions of the Major Planets"
(JPL SSD), Table 1, valid 1800-2050 AD, heliocentric ecliptic & equinox J2000.
Position error is ~1e-4 au for the inner planets (tens of arcsec), adequate for
multi-decade *feature* extraction (not for impact prediction).  JPL Horizons is not reachable
from this environment; a Horizons/DE440 source can be swapped in via ``planet_elements``.

Masses: DE440 GM ratios (Sun / body).
"""
from __future__ import annotations

import numpy as np

GAUSS_K = 0.01720209895
G_AU_DAY = GAUSS_K ** 2            # au^3 / (Msun day^2)
J2000_JD = 2451545.0

# Sun/body mass ratios (DE440). EMB = Earth + Moon.
MASS_MSUN: dict[str, float] = {
    "Venus": 1.0 / 408523.719,
    "EMB": 1.0 / 328900.559,
    "Mars": 1.0 / 3098703.59,
    "Jupiter": 1.0 / 1047.348644,
    "Saturn": 1.0 / 3497.9018,
}
PLANETS: tuple[str, ...] = tuple(MASS_MSUN)

# a [au], e, I [deg], L [deg], long.peri [deg], long.node [deg]  ; then rates per Julian century
_STANDISH: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "Venus": ((0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
              (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418)),
    "EMB": ((1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0),
            (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0)),
    "Mars": ((1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
             (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343)),
    "Jupiter": ((5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
                (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106)),
    "Saturn": ((9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
               (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794)),
}


def planet_elements(name: str, jd: float) -> dict[str, float]:
    """Heliocentric osculating-like elements (a, e, inc, Omega, omega, M) in au / radians."""
    if not 2378496.5 <= jd <= 2469807.5:      # 1800-01-01 .. 2050-01-01
        raise ValueError(f"Standish Table 1 is valid 1800-2050; got JD {jd}")
    el0, rate = _STANDISH[name]
    T = (jd - J2000_JD) / 36525.0
    a, e, inc, L, varpi, node = (x0 + r * T for x0, r in zip(el0, rate))
    if inc < 0:                       # EMB inclination is ~ -1.5e-5 deg: flip to inc>0
        inc, node = -inc, node + 180.0
    omega = varpi - node
    M = L - varpi
    wrap = lambda d: np.deg2rad(np.mod(d, 360.0))
    return {"a": a, "e": e, "inc": np.deg2rad(inc), "Omega": wrap(node),
            "omega": wrap(omega), "M": wrap(M)}
