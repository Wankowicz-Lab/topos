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
        Columns: chain, ss_domains, ss_domain_length.
    """
    return merged.groupby(['chain', 'ss_domains'], as_index=False).size().rename(columns={'size': 'ss_domain_length'})


def ss_domain_log2_aa_group_ratios(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute log2(prop_in_domain / prop_global) per (chain, ss_domain) and aa group.

    Prop_global is computed per chain (chain-wide proportion of each aa group).
    Uses residue-level counts (one row per residue). Derives aa_group from
    resn_struct via AA_TO_GROUP. Returns a DataFrame with chain, ss_domains and
    log2_ratio_<group> for each group in AA_GROUPS.

    Parameters
    ----------
    merged : pd.DataFrame
        DataFrame with columns chain, resi_struct, resn_struct, ss_domains.

    Returns
    -------
    pd.DataFrame
        Columns: chain, ss_domains, ss_domain_log2_aa_group_ratio_<g> for each g in AA_GROUPS.
    """
    res_key = ['chain', 'resi_struct']
    residues = merged[res_key + ['ss_domains', 'resn_struct']].copy()
    residues['aa_group'] = residues['resn_struct'].map(AA_TO_GROUP)
    residues = residues[res_key + ['ss_domains', 'aa_group']].drop_duplicates(res_key)

    # Per-chain global proportions for each aa group
    chain_totals = residues.groupby('chain').size()
    chain_group_counts = residues.groupby(['chain', 'aa_group']).size().unstack(fill_value=0)

    out = merged[['chain', 'ss_domains']].drop_duplicates().copy()

    for g in AA_GROUPS.keys():
        # Chain-specific prop_global
        group_in_chain = chain_group_counts[g] if g in chain_group_counts.columns else 0
        chain_prop_global = group_in_chain / chain_totals
        chain_prop_global = chain_prop_global.reindex(chain_totals.index).fillna(0)

        domain_counts = residues.groupby(['chain', 'ss_domains'])['aa_group'].agg(
            total='count',
            in_group=lambda s: (s == g).sum(),
        ).reset_index()

        domain_counts['prop_domain'] = np.where(
            domain_counts['total'] > 0,
            domain_counts['in_group'] / domain_counts['total'],
            np.nan,
        )
        domain_counts['prop_global'] = domain_counts['chain'].map(chain_prop_global)

        domain_counts['log2_ratio'] = np.nan
        valid = (
            domain_counts['prop_domain'].notna()
            & (domain_counts['prop_domain'] > 0)
            & (domain_counts['prop_global'] > 0)
        )
        domain_counts.loc[valid, 'log2_ratio'] = np.log2(
            domain_counts.loc[valid, 'prop_domain'] / domain_counts.loc[valid, 'prop_global']
        )
        domain_counts.loc[domain_counts['prop_domain'] == 0, 'log2_ratio'] = -np.inf
        # When prop_global is 0, ratio is undefined
        domain_counts.loc[domain_counts['prop_global'] == 0, 'log2_ratio'] = np.nan

        out = out.merge(
            domain_counts[['chain', 'ss_domains', 'log2_ratio']].rename(
                columns={'log2_ratio': f'ss_domain_log2_aa_group_ratio_{g}'}
            ),
            on=['chain', 'ss_domains'],
            how='left',
        )

    return out
