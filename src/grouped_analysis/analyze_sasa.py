#!/usr/bin/env python3

#THIS SCRIPT IS OLD BUT A HOLDER TO BE UPDATED WHEN WE GET OUTPUT CSVs. 

"""
SASA comparison with per-PDB + per-cluster summaries.

Usage
-----
python sasa_compare_with_cluster_summary.py \
  --sasa_dir results_freesasa \
  --clusters_csv HBDScan_clusters.csv \
"""
import argparse
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Set font and figure params
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 18
})

boxprops = {'edgecolor': 'k', 'linewidth': 2}
lineprops = {'color': 'k', 'linewidth': 2}

boxplot_kwargs = dict({'boxprops': boxprops, 'medianprops': lineprops,
                       'whiskerprops': lineprops, 'capprops': lineprops,
                       'width': 0.75})

def is_atom(line: str) -> bool:
    return line.startswith("ATOM  ") or line.startswith("HETATM")

def ffloat(s: str) -> float:
    try:
        return float(s.strip())
    except Exception:
        return 0.0

def elem_from_fields(atom_name: str, element_field: str) -> str:
    e = element_field.strip().upper()
    if e:
        return e
    m = re.match(r"\s*([A-Za-z])", atom_name)
    return m.group(1).upper() if m else "X"

def is_polar(elem: str) -> bool:
    # Simple rule: C = non-polar; N/O/S/P = polar; others -> polar.
    return False if elem == "C" else True

def res_key(chain: str, resi: int, icode: str, resn: str):
    return (chain or "_", int(resi), (icode or " ").strip() or " ", resn.strip())

def res_label(key) -> str:
    chain, resi, icode, resn = key
    icode = "" if icode == " " else icode
    return f"{resn}{resi}{icode}:{chain}"

def extract_id_from_name(name: str):
    m = re.search(r'(^|[^A-Za-z0-9])(x\d{4})([^A-Za-z0-9]|$)', name, flags=re.I)
    if m: return m.group(2).lower()
    m = re.search(r'(^|[^A-Za-z0-9])([0-9][A-Za-z0-9]{3})([^A-Za-z0-9]|$)', name)
    if m: return m.group(2).upper()
    return None

# ---------- parsing ----------
def parse_residue_sasa_from_pdb(pdb_path: Path) -> dict:
    """
    Sum per-atom SASA (from B-factor) to residue totals, also polar/nonpolar.
    Returns { (chain, resi, icode, resn): {total, polar, nonpolar} }
    """
    out = defaultdict(lambda: {"total":0.0, "polar":0.0, "nonpolar":0.0})
    with pdb_path.open("r", errors="ignore") as fh:
        for line in fh:
            if not is_atom(line): continue
            atom  = line[12:16]
            resn  = line[17:20]
            chain = line[21].strip() or "_"
            resi  = ffloat(line[22:26])
            icode = (line[26].strip() or " ")
            bfac  = ffloat(line[60:66])
            elem  = elem_from_fields(atom, line[76:78] if len(line) >= 78 else "")
            k = res_key(chain, int(resi), icode, resn)
            out[k]["total"] += bfac
            if is_polar(elem): out[k]["polar"] += bfac
            else:              out[k]["nonpolar"] += bfac
    return out

# ---------- core ----------
def load_cluster_map(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df["ID"] = df["ID"].astype(str).str.strip().str.lower()
    return {row.ID: str(row.Cluster) for _, row in df.iterrows()}

def collect_files(sasa_dir: Path) -> list:
    return sorted([p for p in sasa_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".pdb",".ent",".txt"}])


def summarize_clusters(byclu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-residue cluster deltas to cluster-level exposure scores.
    Positive deltas -> increased solvent exposure upon ligand binding.
    """
    if byclu_df.empty:
        return pd.DataFrame(columns=[
            "cluster","n_residues",
            "mean_delta_total","net_delta_total",
            "mean_positive_total","net_positive_total",
            "mean_delta_polar","net_delta_polar",
            "mean_positive_polar","net_positive_polar",
            "mean_delta_nonpolar","net_delta_nonpolar",
            "mean_positive_nonpolar","net_positive_nonpolar",
        ])
    def pos(x): return x.clip(lower=0.0)
    g = byclu_df.groupby("cluster", as_index=False)
    summary = g.apply(lambda df: pd.Series({
        "n_residues": len(df),
        # total
        "mean_delta_total": df["mean_delta_total"].mean(),
        "net_delta_total": df["mean_delta_total"].sum(),
        "mean_positive_total": pos(df["mean_delta_total"]).mean(),
        "net_positive_total": pos(df["mean_delta_total"]).sum(),
        # polar
        "mean_delta_polar": df["mean_delta_polar"].mean(),
        "net_delta_polar": df["mean_delta_polar"].sum(),
        "mean_positive_polar": pos(df["mean_delta_polar"]).mean(),
        "net_positive_polar": pos(df["mean_delta_polar"]).sum(),
        # nonpolar
        "mean_delta_nonpolar": df["mean_delta_nonpolar"].mean(),
        "net_delta_nonpolar": df["mean_delta_nonpolar"].sum(),
        "mean_positive_nonpolar": pos(df["mean_delta_nonpolar"]).mean(),
        "net_positive_nonpolar": pos(df["mean_delta_nonpolar"]).sum(),
    })).reset_index(drop=True)
    return summary.sort_values("net_positive_total", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sasa_dir", required=True, help="SASA information")
    ap.add_argument("--clusters_csv", required=True, help="CSV with columns: PDB, Cluster")
    args = ap.parse_args()

    sasa_dir = Path(args.sasa_dir); outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    files = collect_files(sasa_dir)
    if not files: raise SystemExit(f"No PDB-like files in {sasa_dir}")
    id2cluster = load_cluster_map(Path(args.clusters_csv))
    overall_df, byclu_df, pdb_summary = compare_vs_ref(files, ref_file, id2cluster)

    # Save per-residue tables
    overall_csv = outdir/"residue_sasa_changes_overall.csv"
    byclu_csv   = outdir/"residue_sasa_changes_by_cluster.csv"
    overall_df.to_csv(overall_csv, index=False)
    byclu_df.to_csv(byclu_csv, index=False)

    # Cluster and PDB summaries
    cluster_summary = summarize_clusters(byclu_df)
    cluster_csv = "cluster_sasa_summary.csv"
    cluster_summary.to_csv(cluster_csv, index=False)

    pdb_csv = "pdb_sasa_summary.csv"
    pdb_summary.to_csv(pdb_csv, index=False)

      # --- Create boxplot ---
      plt.figure(figsize=(12, 6))
      ax = sns.boxplot(
          data=pdb_summary,
          x="cluster",
          y="mean_delta_total",
          palette="magma",
          order=ordered_clusters,
          width=0.6,
          fliersize=2
      )
      ax.yaxis.grid(True, linestyle='-', which='major', color='gray', alpha=0.5)

      plt.xlabel("Cluster")
      plt.ylabel("Mean ΔSASA per PDB (Å²)")
      plt.xticks(rotation=45)
      plt.tight_layout()

      out_path = "cluster_mean_SASA_delta_total_boxplot.png"
      plt.savefig(out_path, dpi=300)


if __name__ == "__main__":
    main()
