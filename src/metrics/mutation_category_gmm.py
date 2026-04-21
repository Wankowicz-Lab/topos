"""
2-component Gaussian mixture EM and helpers for mutation_category.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# Equal-tail central mass for LOF/GOF vs neutral (fixed policy; not user-configurable)
MUTATION_CATEGORY_CENTRAL_INTERVAL = 0.90

# Reference sanity (empirically reasonable defaults; revisit with large DMS panels if needed)
MIN_REFERENCE_SAMPLE_COUNT = 5
MIN_REFERENCE_UNIQUE_VALUES = 3
MIN_SYNONYMOUS_STD_FRACTION = 0.05

# Floor for Gaussian scales in EM / PDFs only (not a reference rejection rule)
_EPS_SIGMA = 1e-12

SEPARATION_RATIO_THRESHOLD = 1.5
MIN_COMPONENT_WEIGHT = 0.1

N_MIXTURE_EM_ITER = 200
N_MIXTURE_SAMPLES = 20_000
MIXTURE_SAMPLE_SEED = 314_159_265


def tail_quantile_probs(central_interval: float) -> tuple[float, float]:
    """Equal-tail outer mass: return (lower_q, upper_q) for np.quantile."""
    p = (1.0 - central_interval) / 2.0
    return p, 1.0 - p


def fit_gaussian_mixture(
    values: np.ndarray,
    sigma_floor: float = _EPS_SIGMA,
    n_iter: int = N_MIXTURE_EM_ITER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-component diagonal Gaussian EM."""
    values = np.asarray(values, dtype=float)
    sorted_values = np.sort(values)
    means = np.array(
        [float(np.quantile(sorted_values, 0.25)), float(np.quantile(sorted_values, 0.75))],
        dtype=float,
    )
    std = max(float(np.std(values, ddof=1)), sigma_floor)
    sigmas = np.array([std, std], dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)

    for _ in range(n_iter):
        pdfs = np.vstack(
            [
                weights[k] * norm.pdf(values, loc=means[k], scale=max(sigmas[k], sigma_floor))
                for k in range(2)
            ]
        ).T
        denom = np.clip(pdfs.sum(axis=1, keepdims=True), 1e-12, None)
        resp = pdfs / denom

        nk = resp.sum(axis=0)
        weights = nk / len(values)
        means = (resp * values[:, None]).sum(axis=0) / np.clip(nk, 1e-12, None)
        variances = (resp * (values[:, None] - means) ** 2).sum(axis=0) / np.clip(nk, 1e-12, None)
        sigmas = np.sqrt(np.clip(variances, sigma_floor**2, None))

    return weights, means, sigmas


def mixture_sample_interval(
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    central_interval: float,
    rng: np.random.Generator,
    sigma_floor: float = _EPS_SIGMA,
) -> tuple[float, float]:
    """Equal-tail interval under the mixture (sample-based)."""
    p_lo, p_hi = tail_quantile_probs(central_interval)
    draws = rng.multinomial(N_MIXTURE_SAMPLES, weights)
    chunks: list[np.ndarray] = []
    for k in range(2):
        if draws[k] == 0:
            continue
        chunks.append(
            rng.normal(loc=means[k], scale=max(sigmas[k], sigma_floor), size=int(draws[k]))
        )
    samples = np.concatenate(chunks) if chunks else np.array([], dtype=float)
    if samples.size < 2:
        samples = rng.normal(loc=float(means[0]), scale=max(float(sigmas[0]), sigma_floor), size=1000)
    lower = float(np.quantile(samples, p_lo))
    upper = float(np.quantile(samples, p_hi))
    return lower, upper


def separation_ratio(means: np.ndarray, sigmas: np.ndarray, sigma_floor: float = _EPS_SIGMA) -> float:
    return float(abs(means[0] - means[1]) / max(sigmas[0], sigmas[1], sigma_floor))


def is_two_component(
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    sigma_floor: float = _EPS_SIGMA,
) -> bool:
    sr = separation_ratio(means, sigmas, sigma_floor)
    return bool(sr >= SEPARATION_RATIO_THRESHOLD and float(np.min(weights)) >= MIN_COMPONENT_WEIGHT)


def single_gaussian_interval_from_component(
    mean: float,
    sigma: float,
    central_interval: float,
    sigma_floor: float = _EPS_SIGMA,
) -> tuple[float, float]:
    p_lo, p_hi = tail_quantile_probs(central_interval)
    scale = max(sigma, sigma_floor)
    return (
        float(norm.ppf(p_lo, loc=mean, scale=scale)),
        float(norm.ppf(p_hi, loc=mean, scale=scale)),
    )


def _finite_effects_by_type(seq_data: pd.DataFrame, reference_type: str) -> np.ndarray:
    s = seq_data.loc[seq_data["type"] == reference_type, "effect"]
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s)].astype(float)
    return s.to_numpy()


def _precheck_reference_array(values: np.ndarray, reference_type: str) -> Optional[str]:
    if values.size == 0:
        return f"no finite {reference_type} effects"
    if values.size < MIN_REFERENCE_SAMPLE_COUNT:
        return (
            f"found only {values.size} finite {reference_type} effects; "
            f"need at least {MIN_REFERENCE_SAMPLE_COUNT}"
        )
    if len(np.unique(values)) < MIN_REFERENCE_UNIQUE_VALUES:
        return (
            f"found only {len(np.unique(values))} unique {reference_type} effect values; "
            f"need at least {MIN_REFERENCE_UNIQUE_VALUES}"
        )
    return None


def _warn_reference_issue(reference_type: str, reason: str) -> None:
    warnings.warn(
        f"Mutation category metric rejected the {reference_type} reference distribution: {reason}",
        UserWarning,
        stacklevel=3,
    )


def _narrow_synonymous_rejects(syn_std: float, overall_std: float) -> bool:
    if not (np.isfinite(overall_std) and overall_std > 0):
        return False
    return (syn_std / overall_std) < MIN_SYNONYMOUS_STD_FRACTION


@dataclass(frozen=True)
class MutationCategoryFit:
    """Fitted reference for mutation_category labeling and diagnostics."""

    reference_type: Literal["synonymous", "stop"]
    stop_mode: Literal["lower_component", "mixture"]
    lower_bound: float
    upper_bound: float
    weights: np.ndarray
    means: np.ndarray
    sigmas: np.ndarray
    ref_effects: np.ndarray
    warn_poor_separation: bool


def fit_mutation_category_reference(
    seq_data: pd.DataFrame,
    central_interval: float,
) -> Optional[MutationCategoryFit]:
    """
    Choose synonymous GMM first (with narrow-syn guard); else stop GMM with lower-component
    or mixture interval per separation quality.
    """
    overall = pd.to_numeric(seq_data["effect"], errors="coerce")
    overall = overall[np.isfinite(overall)].astype(float)
    overall_std = float(overall.std(ddof=1)) if overall.size > 1 else float("nan")

    syn = _finite_effects_by_type(seq_data, "synonymous")
    syn_err = _precheck_reference_array(syn, "synonymous")
    syn_ok = syn_err is None
    if syn_ok and _narrow_synonymous_rejects(float(np.std(syn, ddof=1)), overall_std):
        _warn_reference_issue(
            "synonymous",
            "the fitted spread is much narrower than the overall effect distribution, which suggests "
            "the dataset may already be normalized to synonymous mutations.",
        )
        syn_ok = False

    rng = np.random.default_rng(MIXTURE_SAMPLE_SEED)

    if syn_ok:
        weights, means, sigmas = fit_gaussian_mixture(syn)
        lower, upper = mixture_sample_interval(weights, means, sigmas, central_interval, rng)
        warn = not is_two_component(weights, means, sigmas)
        if warn:
            warnings.warn(
                "Mutation category: synonymous 2-component Gaussian mixture is poorly separated "
                f"(separation_ratio={separation_ratio(means, sigmas):.3f}; "
                f"threshold {SEPARATION_RATIO_THRESHOLD}). Labels are still computed from the mixture.",
                UserWarning,
                stacklevel=2,
            )
        return MutationCategoryFit(
            reference_type="synonymous",
            stop_mode="mixture",
            lower_bound=lower,
            upper_bound=upper,
            weights=weights,
            means=means,
            sigmas=sigmas,
            ref_effects=syn,
            warn_poor_separation=warn,
        )

    if syn_err is not None and syn.size > 0:
        _warn_reference_issue("synonymous", syn_err.replace("synonymous ", ""))

    stop = _finite_effects_by_type(seq_data, "stop")
    stop_err = _precheck_reference_array(stop, "stop")
    if stop_err is not None:
        if stop.size > 0:
            _warn_reference_issue("stop", stop_err.replace("stop ", ""))
        warnings.warn(
            "Mutation category metric could not identify a credible synonymous or stop reference "
            "distribution; mutation_category, total_lof, and total_gof will be left unset.",
            UserWarning,
            stacklevel=2,
        )
        return None

    weights, means, sigmas = fit_gaussian_mixture(stop)
    tc = is_two_component(weights, means, sigmas)
    if tc:
        k = int(np.argmin(means))
        lower, upper = single_gaussian_interval_from_component(
            float(means[k]), float(sigmas[k]), central_interval
        )
        mode: Literal["lower_component", "mixture"] = "lower_component"
    else:
        lower, upper = mixture_sample_interval(weights, means, sigmas, central_interval, rng)
        mode = "mixture"
        warnings.warn(
            "Mutation category: stop 2-component Gaussian mixture is poorly separated "
            f"(separation_ratio={separation_ratio(means, sigmas):.3f}; "
            f"threshold {SEPARATION_RATIO_THRESHOLD}). Using the combined mixture for the stop reference.",
            UserWarning,
            stacklevel=2,
        )

    if syn.size > 0:
        warnings.warn(
            "Mutation category metric is using stop mutations as the LOF reference after rejecting "
            "the synonymous reference fit.",
            UserWarning,
            stacklevel=2,
        )
    else:
        warnings.warn(
            "Mutation category metric found no synonymous mutations and is using stop mutations "
            "as the LOF reference.",
            UserWarning,
            stacklevel=2,
        )

    return MutationCategoryFit(
        reference_type="stop",
        stop_mode=mode,
        lower_bound=lower,
        upper_bound=upper,
        weights=weights,
        means=means,
        sigmas=sigmas,
        ref_effects=stop,
        warn_poor_separation=not tc,
    )


def classify_synonymous(lower: float, upper: float, effects: pd.Series) -> pd.Series:
    """LOF / neutral / GOF from mixture quantile bounds."""
    v = pd.to_numeric(effects, errors="coerce").astype(float)
    out = pd.Series(pd.NA, index=effects.index, dtype="object")
    fin = np.isfinite(v.to_numpy())
    vv = v.to_numpy()
    out.loc[fin & (vv < lower)] = "LOF"
    out.loc[fin & (vv > upper)] = "GOF"
    out.loc[fin & (vv >= lower) & (vv <= upper)] = "neutral"
    return out


def classify_stop(upper: float, effects: pd.Series) -> pd.Series:
    """LOF if effect <= upper bound of central mass; neutral above (no GOF)."""
    v = pd.to_numeric(effects, errors="coerce").astype(float)
    out = pd.Series(pd.NA, index=effects.index, dtype="object")
    fin = np.isfinite(v.to_numpy())
    vv = v.to_numpy()
    out.loc[fin & (vv <= upper)] = "LOF"
    out.loc[fin & (vv > upper)] = "neutral"
    return out


def save_mutation_category_diagnostic_png(
    fit: MutationCategoryFit,
    central_interval: float,
    path: Path,
) -> None:
    """Histogram + mixture components + vertical cutoffs; matplotlib imported lazily."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ref = fit.ref_effects
    w, m, s = fit.weights, fit.means, fit.sigmas
    xmin = float(np.min(ref))
    xmax = float(np.max(ref))
    pad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
    grid = np.linspace(xmin - pad, xmax + pad, 400)
    density_mix = sum(w[k] * norm.pdf(grid, loc=m[k], scale=max(s[k], _EPS_SIGMA)) for k in range(2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ref, bins=min(40, max(10, len(ref) // 2)), density=True, color="#c0c0c0", alpha=0.85, label="data")
    ax.plot(grid, density_mix, color="#1f77b4", linewidth=2, label="mixture")
    for k in range(2):
        ax.plot(
            grid,
            w[k] * norm.pdf(grid, loc=m[k], scale=max(s[k], _EPS_SIGMA)),
            linestyle="--",
            linewidth=1.2,
            label=f"comp{k + 1}",
        )
    ax.axvline(fit.lower_bound, color="black", linestyle=":", linewidth=1.5)
    ax.axvline(fit.upper_bound, color="black", linestyle=":", linewidth=1.5)
    mode = f"{fit.reference_type}"
    if fit.reference_type == "stop":
        mode += f" ({fit.stop_mode})"
    ax.set_title(f"mutation_category reference: {mode}, central_interval={central_interval}")
    ax.set_xlabel("effect")
    ax.set_ylabel("density")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
