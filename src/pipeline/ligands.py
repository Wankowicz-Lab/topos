import logging
from typing import List, Tuple

import biotite.structure as struc
import numpy as np
import pandas as pd

from src.pipeline.context import Context
from src.structure.utils import res_key

logger = logging.getLogger(__name__)

PROTEIN_MODS = {
    "MSE", "SEP", "TPO", "PTR", "HYP", "CSO", "MHO", "KCX", "CSD", "CME", "CSX",
    "TRY", "LYS", "ALY", "CMH", "CAF",
}
SOLVENT = {"HOH", "WAT", "H2O", "DOD", "HOD"}
COMMON_BUFFER = {"SO4", "PO4", "GOL", "MPD", "EDO", "PEG"}
IONS = frozenset({
    "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CD", "CO",
    "SO4", "PO4", "ACT", "F", "BR", "IOD", "AU", "AG", "BA", "SR", "LI", "RB",
    "CS", "AL", "CR", "MO", "W", "V", "PT", "PD", "IR", "RH", "RU", "OS", "RE",
    "AZI", "IUM", "MMC", "NO3",
})
KNOWN_LIGANDS = frozenset({
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "NAD", "NAH", "NAI", "NAP", "NDP",
    "FAD", "FMN", "HEM", "NAG", "NDG", "MAN", "GAL", "GLC", "BMA", "FUC", "SIA",
    "DMS", "PG4", "LI1", "SQU", "PLP", "TPP", "COA", "ACP", "SAM", "SAH",
    "STU", "BOG", "DDQ", "LDA", "MLA", "OLA", "RET", "CHL", "CLR", "LMT", "LPP",
})


def _sanitize_column_name(value: str) -> str:
    """Normalize ligand-derived strings for safe column names."""
    return str(value).strip().replace(" ", "_")


def format_ligand_id(chain: str, res_id: int, res_name: str) -> str:
    """Build a canonical ligand identifier for joins against partner residue keys."""
    res_name_str = "" if res_name is None else str(res_name).strip()
    return f"{str(chain).strip()}:{int(res_id)}:{res_name_str}"


def find_ligands(
    array: struc.AtomArray | struc.AtomArrayStack,
    exclude_protein_mods: bool = True,
    exclude_solvent: bool = True,
    exclude_ions: bool = True,
    exclude_common_buffer: bool = True,
    exclude_cholesterol: bool = True,
    warn_unknown_ligands: bool = True,
) -> List[Tuple[str, int, str]]:
    """Identify ligand residues from hetero atoms after applying exclusion filters."""
    if isinstance(array, struc.AtomArrayStack):
        array = array[0]
    if "hetero" not in array.get_annotation_categories():
        return []

    hetero_mask = array.hetero
    if not np.any(hetero_mask):
        return []
    hetero_atoms = array[hetero_mask]

    chains = hetero_atoms.chain_id
    res_ids = hetero_atoms.res_id
    res_names = hetero_atoms.res_name
    res_starts = struc.get_residue_starts(hetero_atoms)

    unique_tuples = set()
    for start in res_starts:
        ch = chains[start]
        rid = int(res_ids[start])
        rn_raw = str(res_names[start]).strip() if res_names[start] is not None else ""
        rn_upper = rn_raw.upper()

        if exclude_protein_mods and rn_upper in PROTEIN_MODS:
            continue
        if exclude_solvent and rn_upper in SOLVENT:
            continue
        if exclude_ions and rn_upper in IONS:
            continue
        if exclude_common_buffer and rn_upper in COMMON_BUFFER:
            continue
        if exclude_cholesterol and rn_upper == "CLR":
            continue
        unique_tuples.add((str(ch).strip(), rid, rn_raw))

    result = sorted(unique_tuples, key=lambda item: (item[0], item[1], item[2]))

    if warn_unknown_ligands:
        for ch, res_id, rn_raw in result:
            rn_upper = rn_raw.upper()
            if rn_upper not in KNOWN_LIGANDS:
                logger.warning(
                    "Ligand not in KNOWN_LIGANDS: %s (chain=%s, res_id=%s)",
                    rn_raw or "(empty)",
                    ch,
                    res_id,
                )

    return result


def calculate_protein_ligand_interactions(
    context: Context,
    contacting_residues_df: pd.DataFrame,
    ligand_radius: float = 4.5,
    second_shell_cutoff: float = 5.0,
) -> pd.DataFrame:
    """Label residues as contact, binding site, or second shell for each ligand."""
    merge_cols = ["chain", "resi_struct", "resn_struct"]
    out = context.residue_table.loc[
        context.residue_table.resi_struct.notna(), merge_cols
    ].drop_duplicates(subset=merge_cols).reset_index(drop=True)

    arr = context.array
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    ligands = find_ligands(arr)
    if len(ligands) == 0:
        logger.warning(
            "No ligands found (hetero atoms absent or all filtered); skipping protein-ligand interaction analysis."
        )
        return out

    protein = context.aa
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(protein.chain_id, context.config.structural_feature_chains)
        protein = protein[chain_mask]

    res_starts = struc.get_residue_starts(protein)
    protein_chains = protein.chain_id[res_starts]
    protein_res_ids = protein.res_id[res_starts]
    protein_res_names = protein.res_name[res_starts]
    n_protein_res = len(res_starts)

    atom_to_res_idx = np.repeat(
        np.arange(n_protein_res),
        np.diff(list(res_starts) + [protein.array_length()]),
    )

    cell_size = max(ligand_radius, second_shell_cutoff) + 0.01
    protein_cell = struc.CellList(protein, cell_size=cell_size)

    for lig_chain, lig_res_id, lig_res_name in ligands:
        ligand_id = format_ligand_id(lig_chain, lig_res_id, lig_res_name)
        ligand_contacting_df = contacting_residues_df[
            contacting_residues_df["partner_residue_key"] == ligand_id
        ]

        contact_keys = set()
        for _, row in ligand_contacting_df[
            ["chain", "resi_struct", "resn_struct"]
        ].drop_duplicates().iterrows():
            contact_keys.add(res_key(row["chain"], row["resi_struct"], row["resn_struct"]))

        ligand_mask = (
            (arr.chain_id == lig_chain)
            & (arr.res_id == lig_res_id)
            & (arr.res_name == lig_res_name)
        )
        ligand_atoms = arr[ligand_mask]
        ligand_coords = ligand_atoms.coord

        protein_atom_indices = set()
        for i in range(ligand_coords.shape[0]):
            near = protein_cell.get_atoms(ligand_coords[i], radius=ligand_radius)
            protein_atom_indices.update(near.tolist())
        binding_res_indices = set(atom_to_res_idx[list(protein_atom_indices)])

        binding_keys = set()
        for ri in binding_res_indices:
            binding_keys.add(
                res_key(protein_chains[ri], protein_res_ids[ri], protein_res_names[ri])
            )

        contact_labels = binding_keys & contact_keys
        binding_only = binding_keys - contact_keys

        binding_atom_mask = np.zeros(protein.array_length(), dtype=bool)
        for ri in binding_res_indices:
            start = res_starts[ri]
            end = res_starts[ri + 1] if ri + 1 < n_protein_res else protein.array_length()
            binding_atom_mask[start:end] = True
        binding_coords = protein.coord[binding_atom_mask]

        second_shell_res_indices = set()
        for i in range(binding_coords.shape[0]):
            near = protein_cell.get_atoms(binding_coords[i], radius=second_shell_cutoff)
            for ai in near:
                ri = atom_to_res_idx[ai]
                if ri not in binding_res_indices:
                    second_shell_res_indices.add(ri)

        second_shell_keys = set()
        for ri in second_shell_res_indices:
            second_shell_keys.add(
                res_key(protein_chains[ri], protein_res_ids[ri], protein_res_names[ri])
            )

        label_map = {key: "contact" for key in contact_labels}
        for key in binding_only:
            label_map[key] = "binding site"
        for key in second_shell_keys:
            label_map[key] = "second shell"

        def lookup(row: pd.Series) -> str | float:
            key = res_key(row["chain"], row["resi_struct"], row["resn_struct"])
            return label_map.get(key, np.nan)

        col = (
            f"ligand_{_sanitize_column_name(lig_chain)}_{lig_res_id}_"
            f"{_sanitize_column_name(lig_res_name)}_interactions"
        )
        out[col] = out.apply(lookup, axis=1)

    return out
