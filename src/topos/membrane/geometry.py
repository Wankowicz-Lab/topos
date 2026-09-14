"""Estimate membrane normal and transform from transmembrane-helix Cα spans."""

from __future__ import annotations

import numpy as np
import pandas as pd


class InsufficientTransmembraneHelices(ValueError):
    """Raised when fewer than three usable TM helices are available for estimation."""


def helix_axis_unit(coords: np.ndarray) -> np.ndarray | None:
    """PCA first component of Cα coords as a unit helix axis (oriented N→C)."""
    if coords.shape[0] < 3:
        return None
    x = coords - coords.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    u = vt[0]
    u = u / (np.linalg.norm(u) + 1e-12)
    # Orient along N→C so opposing helices do not cancel when outer-product summed.
    if np.dot(coords[-1] - coords[0], u) < 0:
        u = -u
    return u


def tmatrix_from_normal_and_center(normal: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Build 4×4 matrix for x' = R x + t with normal→z and center→origin."""
    n = np.asarray(normal, dtype=float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    # Orthonormal basis with n as the new z axis.
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, n)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(n, x)
    R = np.vstack([x, y, n])
    c = np.asarray(center, dtype=float).reshape(3)
    t = -R @ c
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _ca_dataframe(atom_array) -> pd.DataFrame:
    """Collect Cα coordinates keyed by chain and residue id."""
    mask = (atom_array.atom_name == "CA")
    ca = atom_array[mask]
    return pd.DataFrame(
        {
            "chain": ca.chain_id.astype(str),
            "resi": ca.res_id.astype(int),
            "x": ca.coord[:, 0],
            "y": ca.coord[:, 1],
            "z": ca.coord[:, 2],
        }
    )


def estimate_membrane_tmatrix(atom_array, regions: pd.DataFrame) -> np.ndarray:
    """
    Estimate a PDBTM-compatible transform from ≥3 transmembrane helices.

    Per-helix Cα PCA axes are outer-product summed; the principal eigenvector is
    the membrane normal. The center is the median of helix midpoints along that
    normal. Raises InsufficientTransmembraneHelices when fewer than three usable
    helices are available.
    """
    tm = regions[regions["type"] == "transmembrane_helix"]
    if len(tm) < 3:
        raise InsufficientTransmembraneHelices(
            f"Need ≥3 transmembrane helices for membrane-frame estimation; found {len(tm)}"
        )

    ca = _ca_dataframe(atom_array)
    axes = []
    mids = []
    for _, row in tm.iterrows():
        mask = (
            (ca["chain"] == str(row["chain"]))
            & (ca["resi"] >= int(row["pdb_beg"]))
            & (ca["resi"] <= int(row["pdb_end"]))
        )
        pts = ca.loc[mask, ["x", "y", "z"]].to_numpy(dtype=float)
        if len(pts) < 3:
            continue
        u = helix_axis_unit(pts)
        if u is None:
            continue
        axes.append(u)
        mids.append(pts.mean(axis=0))

    if len(axes) < 3:
        raise InsufficientTransmembraneHelices(
            f"Need ≥3 usable TM helix axes for membrane-frame estimation; found {len(axes)}"
        )

    # Dominant direction of helix axes ≈ membrane normal.
    M = np.zeros((3, 3))
    for u in axes:
        M += np.outer(u, u)
    evals, evecs = np.linalg.eigh(M)
    n = evecs[:, int(np.argmax(evals))]
    n = n / (np.linalg.norm(n) + 1e-12)

    mid_proj = [float(np.dot(m, n)) for m in mids]
    center_s = float(np.median(mid_proj))
    center = center_s * n

    return tmatrix_from_normal_and_center(n, center)
