"""
Secondary structure domain metrics.

Provides functions to compute domain-level metrics from residue-level data
aggregated by ss_domains (e.g. ss_length, log2 aa-group proportion ratios).
"""
import numpy as np
import pandas as pd

# AA groups for log2 proportion ratio columns (keys match sequence.aa_groupings / aa_groups_3letter)
AA_GROUPS = {
    "Nonpolar_Aliphatic": ["ALA", "VAL", "LEU", "ILE", "MET"],
    "Aromatic": ["PHE", "TRP", "TYR"],
    "Polar_Uncharged": ["SER", "THR", "ASN", "GLN", "CYS"],
    "Positively_Charged": ["LYS", "ARG", "HIS"],
    "Negatively_Charged": ["ASP", "GLU"],
    "Special": ["PRO", "GLY"],
}

AA_TO_GROUP = {}
for group_name, aa_list in AA_GROUPS.items():
    for aa in aa_list:
        AA_TO_GROUP[aa] = group_name


def ss_domain_lengths(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute residue count per ss_domain.

    Parameters
    ----------
    merged : pd.DataFrame
        DataFrame with column ss_domains (no NA).

    Returns
    -------
    pd.DataFrame
        Columns: ss_domains, ss_length.
    """
    return merged.groupby(['chain', 'ss_domains'], as_index=False).size().rename(columns={'size': 'ss_length'})


def ss_domain_log2_aa_group_ratios(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute log2(prop_in_domain / prop_global) per ss_domain and aa group.

    Uses residue-level counts (one row per residue). Derives aa_group from
    resn_struct via AA_TO_GROUP. Returns a DataFrame with ss_domains and
    log2_ratio_<group> for each group in AA_GROUPS.

    Parameters
    ----------
    merged : pd.DataFrame
        DataFrame with columns chain, resi_struct, resn_struct, ss_domains.

    Returns
    -------
    pd.DataFrame
        Columns: ss_domains, log2_ratio_<g> for each g in AA_GROUPS.
    """
    out = merged[['chain', 'ss_domains']].drop_duplicates().copy()

    res_key = ['chain', 'resi_struct']
    residues = merged[res_key + ['ss_domains', 'resn_struct']].copy()
    residues['aa_group'] = residues['resn_struct'].map(AA_TO_GROUP)
    residues = residues[res_key + ['ss_domains', 'aa_group']].drop_duplicates(res_key)

    n_total = len(residues)
    global_counts = residues['aa_group'].value_counts()

    for g in AA_GROUPS.keys():
        prop_global = global_counts.get(g, 0) / n_total
        if prop_global == 0 or np.isnan(prop_global):
            out[f'log2_ratio_{g}'] = np.nan
            continue

        domain_counts = residues.groupby(['chain', 'ss_domains'])['aa_group'].agg(
            total='count',
            in_group=lambda s: (s == g).sum(),
        ).reset_index()

        domain_counts['prop_domain'] = np.where(
            domain_counts['total'] > 0,
            domain_counts['in_group'] / domain_counts['total'],
            np.nan,
        )

        domain_counts['log2_ratio'] = np.nan
        valid = domain_counts['prop_domain'].notna() & (domain_counts['prop_domain'] > 0)
        domain_counts.loc[valid, 'log2_ratio'] = np.log2(
            domain_counts.loc[valid, 'prop_domain'] / prop_global
        )
        domain_counts.loc[domain_counts['prop_domain'] == 0, 'log2_ratio'] = -np.inf

        out = out.merge(
            domain_counts[['chain', 'ss_domains', 'log2_ratio']].rename(columns={'log2_ratio': f'log2_ratio_{g}'}),
            on=['chain', 'ss_domains'],
            how='left',
        )

    return out
