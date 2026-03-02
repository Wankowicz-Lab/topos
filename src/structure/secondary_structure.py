import subprocess
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, List

import numpy as np
import pandas as pd

from src.pipeline.context import Context
from src.structure.utils import get_metadata_cols
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile


def get_secondary_structure_annotations(context: Context) -> pd.DataFrame:
    """
    Get secondary structure annotations for an atom array.

    Parameters
    ----------
    context : Context
        Context object containing atom array.

    Returns
    -------
    ss_df : pd.DataFrame
        DataFrame containing secondary structure annotations.
    """
    backend = context.extras.get("ss_backend", "pydssp")
    if backend == "mkdssp":
        ss_df, dssp_df = _annotate_with_mkdssp(context)
        context.extras["dssp_output"] = dssp_df
    elif backend == "pydssp":
        ss_df = _annotate_with_pydssp(context)
        context.extras.pop("dssp_output", None)
    else:
        raise ValueError(f"Unknown secondary-structure backend: {backend}")

    ss_df['ss_group'] = make_contiguous_group_labels(ss_df['sse'].tolist())
    return ss_df


def _to_internal_sse(symbol: Any) -> str:
    """Map DSSP/pydssp style labels to legacy a/b/c labels."""
    if symbol is None:
        return "c"
    token = str(symbol).strip().upper()
    if token in {"H", "G", "I", "A"}:
        return "a"
    if token in {"E", "B"}:
        return "b"
    return "c"


def _parse_int(raw: str) -> int | float:
    token = raw.strip()
    if not token:
        return np.nan
    return int(token)


def _parse_float(raw: str) -> float:
    token = raw.strip()
    if not token:
        return np.nan
    return float(token)


def _write_temp_pdb(context: Context) -> Path:
    """Write current structure to temporary PDB for mkdssp input."""
    tmp = NamedTemporaryFile(delete=False, suffix=".pdb")
    tmp.close()
    pdb_file = PDBFile()
    pdb_file.set_structure(context.array)
    pdb_file.write(tmp.name)
    pdb_path = Path(tmp.name)

    # mkdssp expects a valid PDB header line for PDB inputs.
    pdb_text = pdb_path.read_text(encoding="utf-8")
    if not pdb_text.startswith("HEADER"):
        pdb_text = "HEADER    BIOGENESIS GENERATED\n" + pdb_text
        pdb_path.write_text(pdb_text, encoding="utf-8")
    return pdb_path


def _annotate_with_mkdssp(context: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run mkdssp and return normalized SSE annotations plus full DSSP fields."""
    pdb_path = _write_temp_pdb(context)
    dssp_tmp = NamedTemporaryFile(delete=False, suffix=".dssp")
    dssp_tmp.close()
    dssp_path = Path(dssp_tmp.name)
    rows: list[dict[str, Any]] = []
    cmd = ["mkdssp", str(pdb_path), str(dssp_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    in_table = False
    for line in dssp_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("  #"):
            in_table = True
            continue
        if not in_table or len(line) < 115:
            continue

        aa = line[13].strip()
        if aa == "!" or not aa:
            continue
        if len(aa) == 1 and aa.islower():
            aa = "C"

        rows.append({
            "chain": line[11].strip(),
            "resi": _parse_int(line[5:10]),
            "resn_dssp": aa,
            "dssp_sse8": line[16].strip() or "C",
            "dssp_acc": _parse_int(line[34:38]),
            "dssp_nh_o_1_relidx": _parse_int(line[38:45]),
            "dssp_nh_o_1_energy": _parse_float(line[46:50]),
            "dssp_o_nh_1_relidx": _parse_int(line[50:56]),
            "dssp_o_nh_1_energy": _parse_float(line[57:61]),
            "dssp_nh_o_2_relidx": _parse_int(line[61:67]),
            "dssp_nh_o_2_energy": _parse_float(line[68:72]),
            "dssp_o_nh_2_relidx": _parse_int(line[72:78]),
            "dssp_o_nh_2_energy": _parse_float(line[79:83]),
            "dssp_tco": _parse_float(line[85:91]),
            "dssp_kappa": _parse_float(line[91:97]),
            "dssp_alpha": _parse_float(line[97:103]),
            "dssp_phi": _parse_float(line[103:109]),
            "dssp_psi": _parse_float(line[109:115]),
        })

    dssp_df = pd.DataFrame(rows)
    dssp_df["sse"] = dssp_df["dssp_sse8"].map(_to_internal_sse)

    # Build canonical residue keys from the structure first, then merge DSSP onto them.
    ss_df = get_metadata_cols(context.aa)
    ss_df.rename(columns={"resi_struct": "resi"}, inplace=True)
    merge_cols = ["chain", "resi"]
    ss_df = ss_df.merge(dssp_df[merge_cols + ["sse"]], on=merge_cols, how="left")
    ss_df["sse"] = ss_df["sse"].fillna("c")

    return ss_df, dssp_df


def _extract_backbone_coords(context: Context) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract per-residue N/CA/C/O coordinates for pydssp assignment."""
    aa = context.aa
    starts = struc.get_residue_starts(aa)
    ends = np.append(starts[1:], aa.array_length())

    keys: list[dict[str, Any]] = []
    coords: list[np.ndarray] = []

    for start, end in zip(starts, ends):
        residue = aa[start:end]
        atom_index = {name: idx for idx, name in enumerate(residue.atom_name)}

        keys.append({
            "chain": str(residue.chain_id[0]).strip(),
            "resi": int(residue.res_id[0]),
            "resn_struct": str(residue.res_name[0]).strip(),
        })

        backbone_names = ("N", "CA", "C", "O")
        if not all(name in atom_index for name in backbone_names):
            coords.append(np.full((4, 3), np.nan, dtype=float))
            continue

        backbone_coords = np.vstack([
            residue.coord[atom_index["N"]],
            residue.coord[atom_index["CA"]],
            residue.coord[atom_index["C"]],
            residue.coord[atom_index["O"]],
        ])
        coords.append(backbone_coords)

    keys_df = pd.DataFrame(keys)
    if not coords:
        return keys_df, np.empty((0, 4, 3), dtype=float)
    return keys_df, np.asarray(coords, dtype=float)


def _annotate_with_pydssp(context: Context) -> pd.DataFrame:
    """Run pydssp and normalize output into the existing ss_df schema."""
    import pydssp  # type: ignore

    keys_df, coords = _extract_backbone_coords(context)
    if len(keys_df) == 0:
        keys_df["sse"] = pd.Series(dtype=str)
        return keys_df

    valid_mask = ~np.isnan(coords).any(axis=(1, 2))
    labels = np.array(["c"] * len(keys_df), dtype=object)
    if valid_mask.any():
        assign_fn = getattr(pydssp, "assign", None)
        if assign_fn is None and hasattr(pydssp, "pydssp"):
            assign_fn = getattr(pydssp.pydssp, "assign", None)
        if assign_fn is None:
            raise RuntimeError("Could not locate pydssp assign function.")

        raw = assign_fn(coords[valid_mask], out_type="c3")
        labels[valid_mask] = [_to_internal_sse(x) for x in np.asarray(raw)]

    keys_df["sse"] = labels
    return keys_df[["chain", "resi", "sse"]]



def make_contiguous_group_labels(lst: List[str]) -> List[str]:
    """
    Given a list of values, return a new list where contiguous identical values
    are labeled with a suffix indicating their group number.

    Parameters
    ----------

    lst : List[str]
        Input list of values.

    Returns
    -------

    result : List[str]
        List with contiguous group labels.
    """
    result = []
    counters = {}

    # Group by contiguous identical values
    for val, group in groupby(lst):
        counters[val] = counters.get(val, 0) + 1

        # Create label with group number
        label = f"{val}_{counters[val]}"
        result.extend([label] * len(list(group)))

    return result


def define_membrane_secondary_structure(residue_table: pd.DataFrame, ss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify discrete secondary structure domains and add to the residue_table.

    Parameters
    ----------
    residue_table : pd.DataFrame
        DataFrame containing residue metadata
    ss_df : pd.DataFrame
        DataFrame containing secondary structure assignments for each residue

    Returns
    -------
    annotated_df : pd.DataFrame
        Input residue_table augmented with 'ss_group' and 'ss_domains' columns
    """

    residue_table = residue_table.copy()
    residue_table['ss_domains'] = pd.NA
    residue_table = residue_table.merge(ss_df[['chain', 'resi', 'ss_group']], on=['chain', 'resi'], how='left')

    membrane_spanning = residue_table.loc[residue_table['pdbtm_region'] == 'membrane_spanning', 'pdbtm_region_detailed'].unique()

    # Loop through each membrane spanning region
    for region in membrane_spanning:
        # Get all secondary structure elements that overlap with this region
        region_count = region.split('membrane_spanning_')[-1]
        mask = residue_table['pdbtm_region_detailed'] == region
        ss_in_region = residue_table.loc[mask, 'ss_group'].unique()

        for ss in ss_in_region:
            # helices that overlap at all with the membrane region are part of TMD
            if ss.startswith('a'):
                residue_table.loc[residue_table['ss_group'] == ss, 'ss_domains'] = 'TMD_' + region_count

            # loops or beta sheets that are completely contained within the membrane are part of TMD
            else:
                ss_mask = residue_table['ss_group'] == ss
                ss_indices = np.where(ss_mask)[0]
                mask_indices = np.where(mask)[0]

                # check if ss is fully contained within the membrane region, ends inclusive
                if ss_indices[0] >= mask_indices[0] and ss_indices[-1] <= mask_indices[-1]:
                    residue_table.loc[ss_mask, 'ss_domains'] = 'TMD_' + region_count

    non_membrane_mask = residue_table['pdbtm_region'].isin(['cytoplasmic', 'extracellular'])
    non_membrane = residue_table.loc[non_membrane_mask, 'pdbtm_region_detailed'].unique()

    # loop through each non-membrane region
    for region in non_membrane:
        # Get all secondary structure elements that overlap with this region
        region_name, region_count = region.split('_')
        mask = residue_table['pdbtm_region_detailed'] == region
        ss_in_region = residue_table.loc[mask, 'ss_group'].unique()

        # loop through each secondary structure element in this region
        for ss in ss_in_region:
            # Get parts of this element that haven't been assigned to a TMD
            ss_mask = residue_table['ss_group'] == ss
            unassigned_mask = residue_table.loc[ss_mask, 'ss_domains'].isna()
            ss_mask = ss_mask & unassigned_mask

            # unassigned regions are part of the loop
            if np.sum(ss_mask) > 0:
                residue_table.loc[ss_mask, 'ss_domains'] = region_name + '_loop_' + region_count

    return residue_table


def define_soluble_secondary_structure(residue_table: pd.DataFrame, ss_df: pd.DataFrame, min_ss_length: int = 2) -> pd.DataFrame:
    """
    Identify discrete secondary structure domains and add to the residue_table.

    Parameters
    ----------
    residue_table : pd.DataFrame
        DataFrame containing residue metadata
    ss_df : pd.DataFrame
        DataFrame containing secondary structure assignments for each residue
    min_ss_length : int
        Minimum length of a secondary structure domain to be considered a discrete domain. Domains less than this length 
        that are in between two domains of the same type will be merged into the adjacent domains.

    Returns
    -------
    annotated_df : pd.DataFrame
        Input residue_table augmented with 'ss_group' and 'ss_domains' columns
    """
    
    # Get secondary structure groups less than min_ss_length
    ss_group_counts = ss_df['ss_group'].value_counts()
    short_ss_groups = ss_group_counts[ss_group_counts < min_ss_length].index.tolist()

    # Merge short ss groups into adjacent domains
    for ss_group in short_ss_groups:
        ss_mask = ss_df['ss_group'] == ss_group
        ss_indices = np.where(ss_mask)[0]
        if len(ss_indices) == 0:
            continue

        # Merge into previous domain if not first in chain or last in chain
        if ss_indices[0] > 0 and ss_indices[0] < len(ss_df) - 1:
            # Get adjacent ss groups
            previous_ss_group = ss_df.iloc[ss_indices[0] - 1]['ss_group']
            subsequent_ss_group = ss_df.iloc[ss_indices[0] + 1]['ss_group']
            if previous_ss_group.split('_')[0] == subsequent_ss_group.split('_')[0]:
                ss_df.loc[ss_mask, 'ss_group'] = previous_ss_group
                ss_df.loc[ss_df['ss_group'] == subsequent_ss_group, 'ss_group'] = previous_ss_group
    
    # ss_domains column has the full name of each group
    ss_df['ss_domains'] = ss_df['ss_group']
    ss_df['ss_domains'] = ss_df['ss_domains'].str.replace('a_', 'alpha-helix_')
    ss_df['ss_domains'] = ss_df['ss_domains'].str.replace('b_', 'beta-sheet_')
    ss_df['ss_domains'] = ss_df['ss_domains'].str.replace('c_', 'coil_')

    residue_table = pd.merge(residue_table, ss_df[['chain', 'resi', 'ss_group', 'ss_domains']], on=['chain', 'resi'], how='left')
    return residue_table    