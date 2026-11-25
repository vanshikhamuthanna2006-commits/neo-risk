"""Central configuration and leakage taxonomy for the NEO risk pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

SBDB_QUERY_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
SBDB_OBJECT_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
SENTRY_URL = "https://ssd-api.jpl.nasa.gov/sentry.api"

# PHA definition (CNEOS): MOID <= 0.05 au AND H <= 22.0
PHA_MOID_AU = 0.05
PHA_H_MAX = 22.0


class TargetMode(str, Enum):
    PHA = "pha"                        # Mode A: non-circular PHA (no MOID / H in X)
    SENTRY_BINARY = "sentry_binary"    # Mode B: on the Sentry risk list?
    LOG10_IP = "log10_ip"              # Mode B: log10 cumulative impact probability
    PS_CUM = "ps_cum"                  # Mode B: cumulative Palermo scale


class PhysicalPolicy(str, Enum):
    """How much size information is allowed into X.

    H = 5 log10(1329 km / (D sqrt(p_v)))  ->  (diameter, albedo) jointly reconstruct H.
    """
    STRICT = "strict"                            # no diameter, no albedo, no size masks
    NO_H_RECONSTRUCTION = "no_h_reconstruction"  # diameter allowed, albedo -> audit (default)
    FULL = "full"                                # everything (documented leakage; ablation only)


# Columns that *define* the label. Never allowed in X.
LABEL_DEFINING: frozenset[str] = frozenset({"moid", "moid_ld", "H", "pha"})
# Sentry outputs (targets for Mode B, or derived from H/MOID). Never allowed in X.
SENTRY_OUTPUTS: frozenset[str] = frozenset(
    {"ip", "ps_cum", "ps_max", "ts_max", "n_imp", "v_inf", "h", "sentry_diameter",
     "range", "in_sentry"}
    # NB: Sentry `last_obs_jd` is deliberately excluded: it duplicates SBDB `last_obs`
    # (bookkeeping, not risk information) and caused a false-positive gate trip.
)
# Regex guard against renamed copies (applied to X column names, case-insensitive).
FORBIDDEN_NAME_REGEX = r"(moid|^h$|^h_|abs_?mag|pha|sentry|^ip$|impact_prob|palermo|^ps_|torino|^ts_|albedo)"
# Explicit, reviewed exceptions to the NAME guard (the numeric value guard still applies).
# Class-level albedo priors are a deterministic function of the taxonomy one-hots already in X;
# they are not measured albedo and cannot reconstruct H with diameter beyond what taxonomy does.
GATE_NAME_ALLOWLIST: frozenset[str] = frozenset({"phys_albedo_prior_median", "phys_albedo_prior_logsd"})
# Numeric value guard: any X column this correlated with an audit column is rejected.
MAX_ABS_CORR_WITH_AUDIT = 0.995

SBDB_FIELDS: tuple[str, ...] = (
    "spkid", "pdes", "full_name", "class",
    # orbit (heliocentric ecliptic J2000, osculating at `epoch`, TDB)
    "epoch", "a", "e", "i", "om", "w", "ma", "q", "ad", "tp", "t_jup",
    # 1-sigma diagonal uncertainties
    "sigma_a", "sigma_e", "sigma_i", "sigma_om", "sigma_w", "sigma_ma", "sigma_q", "sigma_tp",
    # observation metadata
    "data_arc", "n_obs_used", "n_del_obs_used", "n_dop_obs_used", "condition_code", "rms",
    "first_obs", "last_obs",
    # sparse physical
    "diameter", "diameter_sigma", "albedo", "spec_B", "spec_T", "rot_per",
    # leakage / audit only
    "moid", "moid_ld", "H", "pha",
)

ORBIT_CORE: tuple[str, ...] = ("epoch", "a", "e", "i", "om", "w", "ma")
# Cometary covariance labels as returned by sbdb.api (order matters).
COV_LABELS: tuple[str, ...] = ("e", "q", "tp", "node", "peri", "i")


@dataclass
class PipelineConfig:
    cache_dir: Path = Path("data/raw")
    target_mode: TargetMode = TargetMode.PHA
    physical_policy: PhysicalPolicy = PhysicalPolicy.NO_H_RECONSTRUCTION
    sbdb_page_size: int = 20_000
    max_objects: int | None = None          # cap for debugging
    request_timeout_s: float = 120.0
    max_retries: int = 4
    cov_n_jobs: int = 4                     # be polite to JPL
    log10_ip_floor: float = -12.0           # log10(ip) assigned to non-Sentry objects
    ps_cum_floor: float = -12.0             # ps_cum assigned to non-Sentry objects
    snapshot_jd: float | None = None        # reference epoch for staleness; default: max(epoch)
    extra_query_params: dict[str, str] = field(default_factory=dict)
