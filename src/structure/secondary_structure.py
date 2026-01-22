import numpy as np
import pandas as pd
from typing import List
from itertools import groupby

from src.pipeline.context import Context
from src.structure.utils import get_metadata_cols
import biotite.structure as struc


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
    sse_vals = struc.annotate_sse(context.aa)
    ss_df = get_metadata_cols(context.aa)
    ss_df.rename(columns={'resi_struct': 'resi'}, inplace=True)
    ss_df["sse"] = sse_vals
    ss_df['ss_group'] = make_contiguous_group_labels(ss_df['sse'].tolist())
    return ss_df



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