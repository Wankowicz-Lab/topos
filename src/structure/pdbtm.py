import numpy as np
import requests
import pandas as pd
import logging
from lxml import etree
from typing import Tuple, Dict, List
from itertools import groupby

logger = logging.getLogger(__name__)

API_BASE = "https://pdbtm.unitmp.org/api/v1/entry"

def _parse_pdbtm_xml(xml_bytes: bytes) -> Tuple[np.ndarray, List[dict]]:
    """
    Parse PDBTM XML content into a list of regions and transformation matrix.
    """
    parser = etree.XMLParser(ns_clean=True, recover=True)
    root = etree.fromstring(xml_bytes, parser=parser)
    regions = []

    # Loop through <CHAIN> elements
    for chain_elem in root.xpath("//*[local-name() = 'CHAIN']"):
        # Chain ID can appear in several possible attributes
        cid = (chain_elem.get("CHAINID") or chain_elem.get("CHAIN_ID") or
               chain_elem.get("chainId") or chain_elem.get("id") or "").strip()
        if not cid:
            continue

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

    # find MEMBRANE node
    membrane_node = root.xpath("//*[local-name() = 'MEMBRANE']")
    if not membrane_node:
        raise RuntimeError("No <MEMBRANE> element found in XML")

    # use the first MEMBRANE block
    mem = membrane_node[0]

    # Extract TMATRIX rows
    tmatrix_node = mem.xpath(".//*[local-name() = 'TMATRIX']")
    if not tmatrix_node:
        raise RuntimeError("No <TMATRIX> element found in <MEMBRANE>")

    tn = tmatrix_node[0]

    # Parse numeric values from each row
    mat = np.eye(4, dtype=float)
    for i, key in enumerate(["ROWX", "ROWY", "ROWZ"]):
        row = tn.xpath(f".//*[local-name() = '{key}']")[0]

        mat[i, 0] = float(row.get("X"))
        mat[i, 1] = float(row.get("Y"))
        mat[i, 2] = float(row.get("Z"))
        mat[i, 3] = float(row.get("T"))

    return mat, regions


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
        "H": "membrane_spanning",
        "1": "cytoplasmic",
        "2": "extracellular",
        "U": "unknown"
    }
    # Return mapping if found, otherwise the initial code
    return mapping.get(region_code.upper(), region_code)


def fetch_pdbtm_annotation(pdb_id: str, timeout: int = 15) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    """
    Fetch PDBTM annotation in XML format from pdbtm.unitmp.org.

    Parameters:
    -----------
        pdb_id: str
            4-character PDB identifier (case-insensitive)
        timeout: int
            Request timeout in seconds (default: 15)

    Returns:
    ---------
        regions_df: pd.DataFrame
            DataFrame with PDBTM region annotations
        mat: np.ndarray
            4x4 transformation matrix from PDBTM
    """

    pdb = pdb_id.lower()
    xml_url = f"{API_BASE}/{pdb}.xml"
    headers = {"Accept": "application/xml, */*"}

    logger.info("Initiating PDBTM API request")
    try:
        r = requests.get(xml_url, timeout=timeout, headers=headers)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch PDBTM entry for {pdb_id}: {e}")

    xml_bytes = r.content
    mat, regions = _parse_pdbtm_xml(xml_bytes)

    if not regions:
        raise RuntimeError(f"No regions found in XML for {pdb_id}")

    regions_df = pd.DataFrame(regions, columns=['chain', 'type', 'seq_beg', 'seq_end', 'pdb_beg', 'pdb_end'])
    regions_df.type = regions_df.type.apply(describe_pdbtm_region)

    return regions_df, mat


def transform_coordinates(coords: np.ndarray, tmatrix: np.ndarray) -> np.ndarray:
    """
    Apply the PDBTM transformation matrix to a set of 3D coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of 3D coordinates
    tmatrix : np.ndarray
        4x4 transformation matrix from PDBTM with 3x3 rotation matrix and 4th column translation vector

    Returns
    -------
    transformed_coords : np.ndarray
        Nx3 array of transformed 3D coordinates
    """

    if len(coords.shape) !=2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be of shape Nx3")

    if tmatrix.shape != (4, 4):
        raise ValueError("Transformation matrix must be of shape 4x4")

    # Convert to homogeneous coordinates by adding a column of ones
    num_coords = coords.shape[0]
    homogeneous_coords = np.hstack([coords, np.ones((num_coords, 1))])

    # Apply the transformation matrix
    transformed_homogeneous = homogeneous_coords @ tmatrix.T

    # Convert back to 3D coordinates by dropping the homogeneous coordinate
    transformed_coords = transformed_homogeneous[:, :3]

    return transformed_coords


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
            unknown_regions = group[group['type'] == 'unknown']
            if len(unknown_regions) > 1:
                first_idx = unknown_regions['type_idx'].idxmin()
                last_idx = unknown_regions['type_idx'].idxmax()
                pdbtm_regions.at[first_idx, 'detailed_type'] = 'protein_start'
                pdbtm_regions.at[last_idx, 'detailed_type'] = 'protein_end'

            # Label sequential transmembrane and loop regions
            other_regions = group[group['type'] != 'unknown']
            if len(other_regions) > 0:
                for idx in other_regions.index:
                    pdbtm_regions.at[idx, 'detailed_type'] = f"{pdbtm_regions.at[idx, 'type']}_{pdbtm_regions.at[idx, 'type_idx']}"

    return pdbtm_regions


def add_pdbtm_regions(residue_table: pd.DataFrame, pdbtm_regions: pd.DataFrame) -> pd.DataFrame:
    """
    Add PDBTM region annotations to the residue_table.

    Parameters
    ----------
    residue_table : pd.DataFrame
        DataFrame containing residue metadata
    pdbtm_regions : pd.DataFrame
        DataFrame with PDBTM region annotations extracted from .xml file

    Returns
    -------
    merged_df : pd.DataFrame
        Input residue_table augmented with 'pdbtm_region' and 'pdbtm_region_detailed' columns
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
