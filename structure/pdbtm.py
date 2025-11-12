import requests
import pandas as pd
from lxml import etree
from typing import Tuple, Dict, List

API_BASE = "https://pdbtm.unitmp.org/api/v1/entry"

def _parse_pdbtm_xml(xml_bytes: bytes) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """
    Parse PDBTM XML content into a list of regions and a chain→regions dictionary.
    """
    parser = etree.XMLParser(ns_clean=True, recover=True)
    root = etree.fromstring(xml_bytes, parser=parser)
    regions = []
    chain_map = {}

    # Loop through <CHAIN> elements
    for chain_elem in root.xpath("//*[local-name() = 'CHAIN']"):
        # Chain ID can appear in several possible attributes
        cid = (chain_elem.get("CHAINID") or chain_elem.get("CHAIN_ID") or
               chain_elem.get("chainId") or chain_elem.get("id") or "").strip()
        if not cid:
            continue

        chain_regions = []
        # Loop through <REGION> elements inside each chain
        for region in chain_elem.xpath(".//*[local-name() = 'REGION']"):
            def get_int(attr_names):
                for name in attr_names:
                    val = region.get(name)
                    if val:
                        try:
                            return int(val)
                        except ValueError:
                            pass
                    el = region.find(f".//*[local-name() = '{name}']")
                    if el is not None and el.text:
                        try:
                            return int(el.text)
                        except ValueError:
                            pass
                return 0

            # Extract region attributes, accounting for potentially different namings
            seq_beg = get_int(["seq_beg", "seqBeg", "SEQ_BEG"])
            seq_end = get_int(["seq_end", "seqEnd", "SEQ_END"])
            pdb_beg = get_int(["pdb_beg", "pdbBeg", "PDB_BEG"])
            pdb_end = get_int(["pdb_end", "pdbEnd", "PDB_END"])
            rtype = (region.get("type") or
                     region.findtext(".//*[local-name() = 'type']") or "").strip()

            rec = dict(
                chain=cid,
                type=rtype,
                seq_beg=seq_beg,
                seq_end=seq_end,
                pdb_beg=pdb_beg,
                pdb_end=pdb_end,
            )
            regions.append(rec)
            chain_regions.append(rec)

        chain_map[cid] = chain_regions

    return regions, chain_map


def describe_pdbtm_region(region_code: str) -> str:
    """
    Convert a single-letter PDBTM region type into a descriptive string.

    Parameters
    ----------
    region_code : str
        Single-letter code for the region (e.g., 'H', '1', '2', 'U')

    Returns
    -------
    description : str
        Full-word description of the region.
    """
    mapping = {
        "H": "transmembrane_domain",
        "1": "cytoplasmic_loop",
        "2": "extracellular_loop",
        "U": "unknown_or_unresolved"
    }
    # Return mapping if found, otherwise the initial code
    return mapping.get(region_code.upper(), region_code)


def fetch_pdbtm_annotation(pdb_id: str, timeout: int = 15) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    """
    Fetch PDBTM annotation in XML format from pdbtm.unitmp.org.

    Parameters:
    -----------
        pdb_id : str
            4-character PDB identifier (case-insensitive)
        timeout : int
            Request timeout in seconds (default: 15)

    Returns:
    ---------
        regions_df : pd.DataFrame
            Dataframe with information on regions in each chain
        chain_map : dict
            mapping chain ID -> list of region dicts
    """

    pdb = pdb_id.lower()
    xml_url = f"{API_BASE}/{pdb}.xml"
    headers = {"Accept": "application/xml, */*"}

    try:
        r = requests.get(xml_url, timeout=timeout, headers=headers)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch PDBTM entry for {pdb_id}: {e}")

    xml_bytes = r.content
    regions, chain_map = _parse_pdbtm_xml(xml_bytes)

    if not regions:
        raise RuntimeError(f"No regions found in XML for {pdb_id}")

    regions_df = pd.DataFrame(regions, columns=['chain', 'type', 'seq_beg', 'seq_end', 'pdb_beg', 'pdb_end'])
    regions_df.type = regions_df.type.apply(describe_pdbtm_region)
    return regions_df, chain_map


def annotate_pdbtm_detailed(pdbtm_regions: pd.DataFrame) -> pd.DataFrame:
    """
    Add detailed PDBTM region annotations to the pdbtm_regions DataFrame.

    Parameters
    ----------
    pdbtm_regions : pd.DataFrame
        DataFrame with PDBTM region annotations extracted from .xml file

    Returns
    -------
    annotated_df : pd.DataFrame
        Input pdbtm_regions augmented with a 'detailed_type' column indicating detailed region type.
    """

    # Add a counter for each type within each chain
    pdbtm_regions = pdbtm_regions.copy()
    pdbtm_regions['type_idx'] = pdbtm_regions.groupby(['chain', 'type']).cumcount() + 1

    # Create a default detailed label with sequential numbering
    pdbtm_regions['detailed_type'] = pdbtm_regions['type'] + '_' + pdbtm_regions['type_idx'].astype(str)

    # Label first and last instance of unknown regions in each chain
    for chain_id, group in pdbtm_regions.groupby('chain'):

            # Annotate first and last instance of unknown regions as protein_start and protein_end
            unknown_regions = group[group['type'] == 'unknown_or_unresolved']
            if len(unknown_regions) > 1:
                first_idx = unknown_regions['type_idx'].idxmin()
                last_idx = unknown_regions['type_idx'].idxmax()
                pdbtm_regions.at[first_idx, 'detailed_type'] = 'protein_start'
                pdbtm_regions.at[last_idx, 'detailed_type'] = 'protein_end'

            # Label sequential transmembrane and loop regions
            other_regions = group[group['type'] != 'unknown_or_unresolved']
            if len(other_regions) > 0:
                for idx in other_regions.index:
                    pdbtm_regions.at[idx, 'detailed_type'] = f"{pdbtm_regions.at[idx, 'type']}_{pdbtm_regions.at[idx, 'type_idx']}"

    return pdbtm_regions


def merge_pdbtm_regions(residue_table : pd.DataFrame, pdbtm_regions : pd.DataFrame) -> pd.DataFrame:
    """
    Merge PDBTM region annotations with residue_table.

    Parameters
    ----------
    residue_table : pd.DataFrame
        DataFrame containing residue metadata
    pdbtm_regions : pd.DataFrame
        DataFrame with PDBTM region annotations extracted from .xml file

    Returns
    -------
    merged_df : pd.DataFrame
        Input residue_table augmented with a 'pdbtm_region' column indicating the region type.
    """

    # Annotate detailed region types
    pdbtm_regions = annotate_pdbtm_detailed(pdbtm_regions)

    # Initialize the new column with NAs
    residue_table = residue_table.copy()
    residue_table['pdbtm_region'] = pd.NA
    residue_table['pdbtm_region_detailed'] = pd.NA

    # Iterate over each region and assign the region type to matching residues
    for _, region in pdbtm_regions.iterrows():
        mask = (
            (residue_table['chain'] == region['chain']) &
            (residue_table['resi'] >= region['pdb_beg']) &
            (residue_table['resi'] <= region['pdb_end'])
        )

        # Assign descriptive region type instead of 1 letter code
        residue_table.loc[mask, 'pdbtm_region'] = region['type']
        residue_table.loc[mask, 'pdbtm_region_detailed'] = region['detailed_type']

    return residue_table