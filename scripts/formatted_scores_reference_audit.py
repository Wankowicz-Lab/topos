#!/usr/bin/env python3
"""One-off audit of synonymous/stop reference stats vs mutation_category_gmm thresholds."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics.mutation_category_gmm import (  # noqa: E402
    MIN_COMPONENT_WEIGHT,
    MIN_REFERENCE_SAMPLE_COUNT,
    MIN_REFERENCE_UNIQUE_VALUES,
    MIN_SYNONYMOUS_STD_FRACTION,
    SEPARATION_RATIO_THRESHOLD,
    fit_gaussian_mixture,
    is_two_component,
    separation_ratio,
)


def finite_effects(df: pd.DataFrame, mut_type: str) -> np.ndarray:
    s = df.loc[df["type"] == mut_type, "effect"]
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s)].astype(float)
    return s.to_numpy(dtype=float)


def precheck_flags(values: np.ndarray) -> list[str]:
    flags: list[str] = []
    if values.size == 0:
        return ["empty"]
    if values.size < MIN_REFERENCE_SAMPLE_COUNT:
        flags.append(f"n<{MIN_REFERENCE_SAMPLE_COUNT}")
    if len(np.unique(values)) < MIN_REFERENCE_UNIQUE_VALUES:
        flags.append(f"uniq<{MIN_REFERENCE_UNIQUE_VALUES}")
    return flags


def main() -> None:
    inputs = Path(
        "/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/biogenesis/inputs"
    )
    paths = sorted(inputs.glob("*_formatted_scores.csv"))
    rows = []
    for csv_path in paths:
        df = pd.read_csv(csv_path)
        if "effect" not in df.columns or "type" not in df.columns:
            rows.append(
                {
                    "dataset": csv_path.name,
                    "error": "missing effect or type column",
                }
            )
            continue
        overall = pd.to_numeric(df["effect"], errors="coerce")
        overall = overall[np.isfinite(overall)].astype(float)
        overall_std = float(overall.std(ddof=1)) if overall.size > 1 else float("nan")

        syn = finite_effects(df, "synonymous")
        stop = finite_effects(df, "stop")
        syn_std = float(np.std(syn, ddof=1)) if syn.size > 1 else float("nan")
        narrow_syn = (
            np.isfinite(overall_std)
            and overall_std > 0
            and syn.size > 0
            and (syn_std / overall_std) < MIN_SYNONYMOUS_STD_FRACTION
        )

        syn_flags = precheck_flags(syn)
        stop_flags = precheck_flags(stop)

        rec: dict = {
            "dataset": csv_path.stem.replace("_formatted_scores", ""),
            "n_overall": int(overall.size),
            "overall_std": round(overall_std, 6) if np.isfinite(overall_std) else None,
            "n_syn": int(syn.size),
            "syn_uniq": int(len(np.unique(syn))) if syn.size else 0,
            "syn_std": round(syn_std, 8) if syn.size > 1 else None,
            "syn_precheck_fail": "|".join(syn_flags) if syn_flags else "",
            "narrow_syn": narrow_syn,
            "n_stop": int(stop.size),
            "stop_uniq": int(len(np.unique(stop))) if stop.size else 0,
            "stop_std": round(float(np.std(stop, ddof=1)), 8) if stop.size > 1 else None,
            "stop_precheck_fail": "|".join(stop_flags) if stop_flags else "",
        }

        if syn.size >= MIN_REFERENCE_SAMPLE_COUNT and len(np.unique(syn)) >= MIN_REFERENCE_UNIQUE_VALUES:
            w, m, s = fit_gaussian_mixture(syn)
            sr = separation_ratio(m, s)
            tc = is_two_component(w, m, s)
            rec["syn_gmm_separation"] = round(sr, 4)
            rec["syn_gmm_two_component"] = tc
            rec["syn_gmm_wmin"] = round(float(np.min(w)), 4)

        if stop.size >= MIN_REFERENCE_SAMPLE_COUNT and len(np.unique(stop)) >= MIN_REFERENCE_UNIQUE_VALUES:
            w, m, s = fit_gaussian_mixture(stop)
            sr = separation_ratio(m, s)
            tc = is_two_component(w, m, s)
            rec["stop_gmm_separation"] = round(sr, 4)
            rec["stop_gmm_two_component"] = tc
            rec["stop_gmm_wmin"] = round(float(np.min(w)), 4)

        rows.append(rec)

    out = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("Thresholds: "
          f"n>={MIN_REFERENCE_SAMPLE_COUNT}, uniq>={MIN_REFERENCE_UNIQUE_VALUES}, "
          f"narrow_syn: syn_std/overall_std < {MIN_SYNONYMOUS_STD_FRACTION}, "
          f"two_component: sep>={SEPARATION_RATIO_THRESHOLD} and wmin>={MIN_COMPONENT_WEIGHT}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
