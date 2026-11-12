"""
Sequence alignment functions for protein analysis pipeline.

This module provides functions for loading mutation scores, performing sequence
alignment, and merging mutation data with structural context.
"""

import logging
import warnings
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner

from topos.sequence.utils import (
    VALID_1_CODES,
    VALID_RESM_3_CODES,
    VALID_RESN_3_CODES,
    convert_amino_acid_1to3,
    convert_amino_acid_3to1,
    invalid_codes,
)
from topos.structure.construct_coverage import UNMODELED_SS_LABEL

logger = logging.getLogger(__name__)

# User-facing labels for alignment warnings (merged columns use resn_df1 / resn_df2 before rename).
_LABEL_MUTATION = "mutation sequence"
_LABEL_CONSTRUCT = "construct sequence"
_MUTATION_INPUT_README_SECTION = "README.md#mutation-input-requirements"
VALID_MUTATION_TYPES = frozenset({"missense", "synonymous", "stop", "deletion", "insertion"})
_ANNOTATION_COLS = ("ss_domains", "ss_category", "ss_group", "pdbtm_region", "pdbtm_region_detailed", "membrane_side")


def _assign_coverage_status(merged: pd.DataFrame) -> pd.Series:
    """Classify DMS-facing alignment rows into coverage_status labels."""
    status = pd.Series(pd.NA, index=merged.index, dtype=object)
    has_mut = merged["resn_mut"].notna()
    has_construct = merged["resn_struct"].notna()
    aa_match = has_mut & has_construct & (merged["resn_mut"] == merged["resn_struct"])
    modeled = merged["modeled"].fillna(False).astype(bool)

    status.loc[has_mut & ~has_construct] = "missing_from_construct"
    status.loc[has_mut & has_construct & (merged["resn_mut"] != merged["resn_struct"])] = "construct_mismatch"
    status.loc[aa_match & modeled] = "modeled"
    status.loc[aa_match & ~modeled] = "unmodeled"
    return status


def _format_residue_ranges(positions: pd.Series) -> str:
    """
    Format residue indices as compact ranges, e.g. ``1-3, 5, 10-12``.

    Non-finite values are dropped. Empty input returns an em dash placeholder.
    """
    vals = pd.to_numeric(positions, errors="coerce").dropna()
    if vals.empty:
        return "—"
    ints = sorted({int(x) for x in vals})
    if not ints:
        return "—"
    ranges: list[tuple[int, int]] = []
    start = prev = ints[0]
    for x in ints[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    parts: list[str] = []
    for a, b in ranges:
        parts.append(str(a) if a == b else f"{a}-{b}")
    return ", ".join(parts)


def load_mutation_scores(
    path: Union[str, Path],
    residue_col_name: str,
    residue_idx_name: str,
    mutation_col_name: str,
    mutation_type_col_name: str,
    score_col_name: str,
) -> pd.DataFrame:
    """
    Load mutation scores from a CSV file and standardize column names.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing mutation scores.
    residue_col_name : str
        Name of the column containing wildtype residues.
    residue_idx_name : str
        Name of the column containing residue positions.
    mutation_col_name : str
        Name of the column containing mutant residues.
    mutation_type_col_name : str
        Name of the column containing mutation types.
    score_col_name : str
        Name of the column containing mutation effect scores.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names: 'resn', 'resi', 'resm',
        'type', and 'effect'.

    Raises
    ------
    ValueError
        If required columns are missing or if the residue column contains
        codes that are neither 1-letter nor 3-letter amino acid codes.

    """
    logger.info("Loading mutation scores")
    df = pd.read_csv(path)

    required_cols = [residue_col_name, residue_idx_name, mutation_col_name, mutation_type_col_name, score_col_name]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns {missing_cols} not found in mutation scores file at {path}.\n"
            f"Expected columns based on config settings:\n"
            f"  - Wildtype residue column: '{residue_col_name}'\n"
            f"  - Residue position column: '{residue_idx_name}'\n"
            f"  - Mutant residue column: '{mutation_col_name}'\n"
            f"  - Mutation type column: '{mutation_type_col_name}'\n"
            f"  - Mutation score column: '{score_col_name}'\n"
            f"Available columns in file: {list(df.columns)}"
        )
    df = df.rename(columns={
        residue_col_name: "resn",
        residue_idx_name: "resi",
        mutation_col_name: "resm",
        mutation_type_col_name: "type",
        score_col_name: "effect"
    })

    df["resn"] = df["resn"].astype(str).str.strip().str.upper()
    df["resm"] = df["resm"].astype(str).str.strip().str.upper()
    df["type"] = df["type"].astype(str).str.strip().str.lower()

    # Wildtype residues must use a single code system because they define the sequence.
    residue_lens = set(df["resn"].str.len().unique())
    if residue_lens == {1}:
        invalid_resn = invalid_codes(set(df["resn"].unique()), VALID_1_CODES)
        if invalid_resn:
            raise ValueError(
                f"Wildtype residue column contains invalid 1-letter codes: {sorted(invalid_resn)}. "
                f"See {_MUTATION_INPUT_README_SECTION}."
            )
        df["resn"] = df["resn"].map(convert_amino_acid_1to3)
    elif residue_lens == {3}:
        invalid_resn = invalid_codes(set(df["resn"].unique()), VALID_RESN_3_CODES)
        if invalid_resn:
            raise ValueError(
                f"Wildtype residue column contains invalid 3-letter codes: {sorted(invalid_resn)}. "
                f"See {_MUTATION_INPUT_README_SECTION}."
            )
    else:
        raise ValueError(
            "Wildtype residue column must contain only valid 1-letter or only valid 3-letter amino acid codes. "
            f"See {_MUTATION_INPUT_README_SECTION}."
        )

    mutant_lens = set(df["resm"].str.len().unique())
    if not mutant_lens.issubset({1, 3, 4}):
        invalid_resm = sorted(code for code in df["resm"].unique() if len(code) not in {1, 3, 4})
        raise ValueError(
            f"Mutant residue column contains unsupported tokens: {invalid_resm}. "
            f"See {_MUTATION_INPUT_README_SECTION}."
        )

    invalid_resm_1 = invalid_codes({code for code in df["resm"].unique() if len(code) == 1}, VALID_1_CODES)
    invalid_resm_3 = invalid_codes({code for code in df["resm"].unique() if len(code) == 3}, VALID_RESM_3_CODES)
    invalid_resm_4 = invalid_codes({code for code in df["resm"].unique() if len(code) == 4}, VALID_RESM_3_CODES)
    invalid_resm = sorted(invalid_resm_1 | invalid_resm_3 | invalid_resm_4)
    if invalid_resm:
        raise ValueError(
            f"Mutant residue column contains invalid codes: {invalid_resm}. "
            f"See {_MUTATION_INPUT_README_SECTION}."
        )

    df["resm"] = df["resm"].map(
        lambda code: convert_amino_acid_1to3(code) if len(code) == 1 else code
    )

    found_types = set(df["type"].unique())
    invalid_types = found_types - VALID_MUTATION_TYPES
    if invalid_types:
        raise ValueError(
            f"Mutation type column contains invalid values: {sorted(invalid_types)}. "
            f"Expected one of {sorted(VALID_MUTATION_TYPES)}. "
            f"See {_MUTATION_INPUT_README_SECTION}."
        )

    return df


def alignment_to_index_map(alignment):
    """
    Convert alignment.coordinates into explicit per-residue index mapping to allow indexing into pandas df.

    Parameters
    ----------
    alignment : Alignment object
        Alignment object containing coordinates attribute.

    Returns:
    -------
    list of tuples
        list of (align_pos, idx1, idx2) where either idx may be None for gaps
    """

    coords = alignment.coordinates  # shape (2, n_segments+1)
    map_list = []
    align_pos = 0

    for col in range(coords.shape[1] - 1):
        start1, end1 = coords[0, col], coords[0, col + 1]
        start2, end2 = coords[1, col], coords[1, col + 1]

        len1 = end1 - start1
        len2 = end2 - start2

        if len1 == len2:
            # Match/substitution block
            for i in range(len1):
                map_list.append((align_pos, start1 + i, start2 + i))
                align_pos += 1
        elif len1 > len2:
            # Deletion in seq2 (seq1 has extra)
            for i in range(len1):
                # seq2 gap for positions beyond its end
                map_list.append((align_pos, start1 + i, start2 + i if i < len2 else None))
                align_pos += 1
        else:
            # Insertion in seq2 (seq2 has extra)
            for i in range(len2):
                map_list.append((align_pos, start1 + i if i < len1 else None, start2 + i))
                align_pos += 1

    return map_list


def merge_sequence_dfs(df1: pd.DataFrame, df2: pd.DataFrame, mapping: list) -> pd.DataFrame:
    """
    Merge two sequence DataFrames based on a provided index mapping.

    Parameters
    ----------
    df1: pd.DataFrame
        First DataFrame containing sequence information.
    df2: pd.DataFrame
        Second DataFrame containing sequence information.
    mapping: list of tuples
        List of (align_pos, idx1, idx2) tuples mapping indices from df1 to df2. Either idx may be None for gaps.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing combined sequence information from both DataFrames.
    """

    map_df = pd.DataFrame(mapping, columns=["align_pos", "i1", "i2"])

    # copy dfs to avoid modifying originals
    df1 = df1.copy()
    df2 = df2.copy()

    # add sequential index to each df for merging
    df1['seq_idx'] = range(len(df1))
    df2['seq_idx'] = range(len(df2))

    merged = (
        map_df
        .merge(df1, how="left", left_on="i1", right_on="seq_idx", suffixes=("", "_df1"))
        .merge(df2, how="left", left_on="i2", right_on="seq_idx", suffixes=("", "_df2"))
    )

    merged.drop(columns=["i1", "i2", "seq_idx", "seq_idx_df2"], inplace=True)
    merged.rename(columns={'resi': 'resi_df1', 'resn': 'resn_df1'}, inplace=True)

    # guarantee stable alignment ordering
    merged = merged.sort_values("align_pos", kind="mergesort").reset_index(drop=True)
    return merged


def evaluate_sequence_alignment(merged: pd.DataFrame, alignment_cutoff: float) -> None:
    """
    Evaluate alignment quality and report construct vs coordinate coverage.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged alignment with ``resn_df1``/``resi_df1`` (mutation), ``resn_df2``/
        ``resi_df2`` (construct), and ``modeled`` (coordinates present).
    alignment_cutoff : float
        Quality cutoff for the alignment. If the proportion of alignment is below this cutoff,
        a warning is issued.
    """
    total_residues = len(merged)
    if total_residues == 0:
        return

    has_mut = merged["resn_df1"].notna()
    has_construct = merged["resn_df2"].notna()
    mismatch_mask = has_mut & has_construct & (merged["resn_df1"] != merged["resn_df2"])
    # Unmodeled construct matches are not alignment errors (coordinates missing, sequence OK)
    unmodeled_match = (
        has_mut
        & has_construct
        & (merged["resn_df1"] == merged["resn_df2"])
        & ~merged["modeled"].fillna(False).astype(bool)
    )
    indel_mask = ((merged["resn_df1"].isna()) | (merged["resn_df2"].isna())).to_numpy()
    termini_mask = np.array([False] * total_residues)

    # Check for contiguous blocks of indels at beginning or end
    if indel_mask[0] or indel_mask[-1]:
        for i in range(total_residues):
            if indel_mask[i]:
                termini_mask[i] = True
            else:
                break
        for i in range(total_residues - 1, -1, -1):
            if indel_mask[i]:
                termini_mask[i] = True
            else:
                break

        # Exclude terminal gaps from indel count
        indel_mask = indel_mask & (~termini_mask)

    # Error rate excludes terminal gaps and unmodeled-but-matched construct residues
    exclude_from_errors = termini_mask | unmodeled_match.to_numpy()
    error_mask = (mismatch_mask.to_numpy() | indel_mask) & (~exclude_from_errors)
    scored = ~exclude_from_errors
    n_scored = int(scored.sum())

    readme_hint = (
        " See the README section \"Sequence alignment\" for how warnings map to "
        "runner.context.extras['sequence_alignment_merged']."
    )

    if n_scored and (error_mask.sum() / n_scored) > 1 - alignment_cutoff:
        warnings.warn(
            f"Alignment quality below cutoff of {alignment_cutoff:.2f}. "
            f"Found {(error_mask.sum() / n_scored) * 100:.2f}% errors "
            f"({error_mask.sum()} out of {n_scored} alignment positions) "
            f"excluding terminal gaps and unmodeled construct matches.{readme_hint}"
        )

    mut_only = has_mut
    n_mut = int(mut_only.sum())
    if n_mut:
        aa_match = has_mut & has_construct & (merged["resn_df1"] == merged["resn_df2"])
        has_coords = has_mut & has_construct & merged["modeled"].fillna(False).astype(bool)
        construct_cov = float(aa_match.sum() / n_mut)
        coord_cov = float(has_coords.sum() / n_mut)
        warnings.warn(
            f"Construct coverage: {construct_cov * 100:.2f}% of mutation wildtype positions "
            f"match the construct sequence ({int(aa_match.sum())}/{n_mut}). "
            f"Coordinate coverage: {coord_cov * 100:.2f}% align to a construct residue with "
            f"deposited coordinates ({int(has_coords.sum())}/{n_mut})."
        )

    if mismatches := int(mismatch_mask.sum()):
        mut_pos = _format_residue_ranges(merged.loc[mismatch_mask, "resi_df1"])
        construct_pos = _format_residue_ranges(merged.loc[mismatch_mask, "resi_df2"])
        warnings.warn(
            f"Found {mismatches} construct mismatches out of {total_residues} alignment positions "
            f"({(mismatches / total_residues) * 100:.2f}%).\n"
            f"  {_LABEL_MUTATION.capitalize()} residue positions: {mut_pos}\n"
            f"  {_LABEL_CONSTRUCT.capitalize()} residue positions: {construct_pos}"
        )

    if unmodeled := int(unmodeled_match.sum()):
        mut_pos = _format_residue_ranges(merged.loc[unmodeled_match, "resi_df1"])
        construct_pos = _format_residue_ranges(merged.loc[unmodeled_match, "resi_df2"])
        warnings.warn(
            f"Found {unmodeled} unmodeled construct residues out of {total_residues} "
            f"alignment positions ({(unmodeled / total_residues) * 100:.2f}%). "
            f"These match the deposited polymer but lack coordinates.\n"
            f"  {_LABEL_MUTATION.capitalize()} residue positions: {mut_pos}\n"
            f"  {_LABEL_CONSTRUCT.capitalize()} residue positions: {construct_pos}"
        )

    if indels := int(indel_mask.sum()):
        mut_pos = _format_residue_ranges(merged.loc[indel_mask, "resi_df1"])
        construct_pos = _format_residue_ranges(merged.loc[indel_mask, "resi_df2"])
        warnings.warn(
            f"Found {indels} alignment positions with internal indels out of {total_residues} "
            f"({(indels / total_residues) * 100:.2f}%). "
            f"Mutation-only gaps are missing_from_construct.\n"
            f"  {_LABEL_MUTATION.capitalize()} residue positions: {mut_pos}\n"
            f"  {_LABEL_CONSTRUCT.capitalize()} residue positions: {construct_pos}"
        )

    if termini_mask.any():
        tm = pd.Series(termini_mask)
        mut_pos = _format_residue_ranges(merged.loc[tm, "resi_df1"])
        construct_pos = _format_residue_ranges(merged.loc[tm, "resi_df2"])
        warnings.warn(
            f"Found gaps at the termini of the sequence alignment.\n"
            f"  {_LABEL_MUTATION.capitalize()} residue positions: {mut_pos}\n"
            f"  {_LABEL_CONSTRUCT.capitalize()} residue positions: {construct_pos}"
        )


def merge_mutation_scores(
    mutation_scores: pd.DataFrame,
    residue_table: pd.DataFrame,
    construct_table: pd.DataFrame,
    chain: str,
    alignment_cutoff: float,
    construct_source: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align mutation wildtype sequence to the construct residue table and merge.

    Parameters
    ----------
    mutation_scores : pd.DataFrame
        Mutation scores with ``resi`` / ``resn``.
    residue_table : pd.DataFrame
        Coordinate residue table (annotations such as secondary structure).
    construct_table : pd.DataFrame
        Construct residues (``chain``, ``seq_id``, ``resi``, ``resn``, ``modeled``).
    chain : str
        Chain used for DMS alignment.
    alignment_cutoff : float
        Alignment quality warning threshold.
    construct_source : str
        ``polymer_scheme`` or ``coordinates`` (from construct coverage builder).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(residue_table, alignment_merged)`` for the pipeline and per-column mapping.
    """
    aligner = PairwiseAligner()

    # Create copies to avoid modifying original DataFrames
    residue_table = residue_table.copy()
    mutation_scores = mutation_scores.copy()
    construct_table = construct_table.copy()

    # Subset construct table to the specified chain (N-to-C polymer / seq_id order)
    construct_chain = (
        construct_table[construct_table["chain"] == chain]
        .sort_values(["seq_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if construct_chain.empty:
        available = sorted(construct_table["chain"].unique().tolist())
        if construct_source == "polymer_scheme":
            raise ValueError(
                f"mutation_data_chain '{chain}' not found in mmCIF polymer scheme chains "
                f"{available}. The chain may be absent from the deposited construct."
            )
        raise ValueError(
            f"mutation_data_chain '{chain}' not found in construct residue table chains {available}."
        )

    # Subset mutation scores to only the wildtype sequence (N-to-C order, not CSV row order)
    mutation_scores_subset = (
        mutation_scores[["resi", "resn"]]
        .drop_duplicates()
        .sort_values(["resi", "resn"], kind="mergesort")
        .reset_index(drop=True)
    )

    # Prepare sequences for alignment, a single string of single-letter amino acids
    mut_seq_short = mutation_scores_subset["resn"].apply(
        lambda aa: convert_amino_acid_3to1(aa, force_convert=True)
    )
    mut_seq = "".join(mut_seq_short.tolist())
    construct_seq_short = construct_chain["resn"].apply(
        lambda aa: convert_amino_acid_3to1(aa, force_convert=True)
    )
    construct_seq = "".join(construct_seq_short.tolist())

    # Perform alignment (mutation wildtype → construct polymer / coordinate-fallback sequence)
    logger.info("Performing sequence alignment to construct (source=%s)", construct_source)
    alignment = aligner.align(mut_seq, construct_seq)[0]

    # Create mapping to link dataframes based on alignment
    index_map = alignment_to_index_map(alignment)

    # Merge mutation scores and construct table based on alignment mapping.
    # Align against construct columns only; join coordinate annotations afterward.
    construct_for_align = construct_chain[["resi", "resn", "seq_id", "modeled", "ins_code"]].copy()
    merged_df = merge_sequence_dfs(
        df1=mutation_scores_subset,
        df2=construct_for_align,
        mapping=index_map,
    )

    # Evaluate alignment quality
    evaluate_sequence_alignment(merged=merged_df, alignment_cutoff=alignment_cutoff)

    # Add chain information and rename columns
    merged_df["chain"] = chain
    merged_df.rename(
        columns={
            "resn_df1": "resn_mut",
            "resi_df1": "resi_mut",
            "resn_df2": "resn_struct",
            "resi_df2": "resi_struct",
        },
        inplace=True,
    )
    # merge_sequence_dfs leaves df2 non-key columns unprefixed (seq_id, modeled, ins_code)
    merged_df["coverage_status"] = _assign_coverage_status(merged_df)

    # Join SS / PDBTM annotations from coordinate residue table onto modeled rows
    annot_cols = [c for c in _ANNOTATION_COLS if c in residue_table.columns]
    if annot_cols:
        coord_annot = (
            residue_table.loc[residue_table["chain"] == chain, ["resi", "resn", *annot_cols]]
            .drop_duplicates(subset=["resi", "resn"])
            .rename(columns={"resi": "resi_struct", "resn": "resn_struct"})
        )
        merged_df = merged_df.merge(coord_annot, how="left", on=["resi_struct", "resn_struct"])

    # Unmodeled construct residues have no DSSP assignment; label SS columns explicitly.
    # Aggregation metrics exclude this label (see secondary_structure_features).
    unmodeled_mask = merged_df["modeled"].eq(False)
    for col in ("ss_domains", "ss_category", "ss_group"):
        if col not in merged_df.columns:
            merged_df[col] = pd.NA
        merged_df.loc[unmodeled_mask, col] = UNMODELED_SS_LABEL

    alignment_merged = merged_df.copy()

    # Add mutation information into merged_df
    merged_df = merged_df.merge(
        mutation_scores,
        how="left",
        left_on=["resi_mut", "resn_mut"],
        right_on=["resi", "resn"],
    )
    # Drop duplicate columns from the merge (resi and resn are duplicates of resi_mut and resn_mut)
    merged_df.drop(columns=["resi", "resn"], inplace=True, errors="ignore")

    # Remove rows from mutation chain from residue table, update with merged construct-aligned rows.
    # Non-target chains stay coordinate-only; target chain is replaced by the construct alignment.
    other_chains = residue_table[residue_table["chain"] != chain].copy()
    other_chains.rename(columns={"resn": "resn_struct", "resi": "resi_struct"}, inplace=True)
    other_chains["modeled"] = True
    other_chains["coverage_status"] = pd.NA
    residue_table = pd.concat([other_chains, merged_df], axis=0).reset_index(drop=True)

    # Determine which rows have mutation and structure (coordinate) info
    residue_table["mut_info"] = ~residue_table["resn_mut"].isna()
    # struct_info means coordinates present (not merely in the construct polymer)
    residue_table["struct_info"] = residue_table["modeled"].fillna(False).astype(bool)

    # drop extra columns if present
    keep_cols = [
        "chain",
        "resi_mut",
        "resn_mut",
        "resm",
        "resi_struct",
        "resn_struct",
        "ss_domains",
        "ss_category",
        "ss_group",
        "type",
        "effect",
        "mut_info",
        "struct_info",
        "modeled",
        "coverage_status",
        "align_pos",
        "seq_id",
    ]
    keep_cols.extend([c for c in annot_cols if c in residue_table.columns and c not in keep_cols])
    residue_table = residue_table[[c for c in keep_cols if c in residue_table.columns]]

    return residue_table, alignment_merged
