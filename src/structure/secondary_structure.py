import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from itertools import groupby


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


def define_secondary_structure(residue_table: pd.DataFrame, ss_df: pd.DataFrame) -> pd.DataFrame:
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
        Input residue_table augmented with 'secondary_structure' column
    """

    residue_table = residue_table.copy()

    ss_df['ss_group'] = make_contiguous_group_labels(ss_df['sse'].tolist())

    residue_table['ss_domains'] = pd.NA
    residue_table = residue_table.merge(ss_df[['chain', 'resi_struct', 'ss_group']], on=['chain', 'resi_struct'], how='left')

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

    return residue_table[['chain', 'resi_struct', 'resn_struct', 'ss_group', 'ss_domains']].drop_duplicates()
