"""
Graph-based residue descriptors for protein analysis.

This module provides functions for computing network metrics on residue-bond graphs,
including centrality measures and community detection.
"""
from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

from topos.structure.utils import res_key

logger = logging.getLogger(__name__)


def calculate_graph_metrics(
    bonds_df: pd.DataFrame,
    residue_table: pd.DataFrame,
    connected_components: bool = True,
) -> pd.DataFrame:
    """
    Compute graph-based metrics on a residue-bond graph and map them to residues.

    Constructs an undirected, unweighted graph from residue-residue bonds,
    restricts to the largest connected component (LCC), computes centrality and
    community metrics, then maps results back to the residue table.

    Parameters
    ----------
    bonds_df : pd.DataFrame
        DataFrame with columns chain_1, resi_1, chain_2, resi_2. Each row represents a bond between two residues.
    residue_table : pd.DataFrame
        DataFrame with at least chain and resi_struct columns to identify residues.
    connected_components: bool = True,
        Whether to restrict to the largest connected component.

    Returns
    -------
    pd.DataFrame
        residue_table with added columns:
        - graph_betweenness_centrality
        - graph_closeness_centrality
        - graph_eigenvector_centrality
        - graph_core_number
        - graph_community_id
        - graph_in_lcc

        Residues not in the largest connected component receive NaN for centrality
        metrics; graph_in_lcc is False for them.
    """
    merge_cols = ['chain', 'resi_struct', 'resn_struct']
    result = residue_table.copy()

    # Remove residues without structural information
    result = result[result['struct_info']]

    # Subset to one row per structural residue
    result = result[merge_cols].drop_duplicates(subset=merge_cols).reset_index(drop=True)

    # Initialize metric columns with NaN
    metric_cols = [
        "graph_betweenness_centrality",
        "graph_closeness_centrality",
        "graph_eigenvector_centrality",
        "graph_core_number",
        "graph_community_id",
        "graph_in_lcc",
    ]
    for col in metric_cols:
        result[col] = pd.NA if col != "graph_in_lcc" else False

    # Normalize column names
    c1, r1, c2, r2 = 'chain', 'resi_struct', 'partner_chain', 'partner_resi'

    # Build undirected graph
    G = nx.Graph()
    for _, row in bonds_df.iterrows():
        key1 = res_key(row[c1], row[r1], row["resn_struct"])
        key2 = res_key(row[c2], row[r2], row["partner_resn"])
        if key1 != key2:
            G.add_edge(key1, key2)

    if G.number_of_nodes() == 0:
        logger.warning("No edges in graph; returning residue_table with NaN for all graph metrics")
        return result

    # Find largest connected component
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_lcc = G.subgraph(largest_cc).copy()
    
    # Report excluded residues
    excluded = set(G.nodes()) - largest_cc
    if excluded:
        logger.info(
            "Residues not in largest connected component (%d excluded): %s",
            len(excluded),
            sorted(excluded)[:20],
        )
        if len(excluded) > 20:
            logger.info("... and %d more", len(excluded) - 20)

    if connected_components:
        G = G_lcc        
    
    # Compute metrics on LCC
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        logger.warning("eigenvector_centrality failed to converge; using zeros")
        eigenvector = {n: 0.0 for n in G.nodes()}
    core_num = nx.core_number(G)
    comms = list(nx.community.greedy_modularity_communities(G))

    # Map community id (0-indexed)
    node_to_community = {}
    for i, comm in enumerate(comms):
        for node in comm:
            node_to_community[node] = i

    # Build residue keys for residue_table
    result_keys = [
        res_key(row["chain"], row["resi_struct"], row["resn_struct"])
        for _, row in result.iterrows()
    ]

    # Map metrics onto residue_table
    result["graph_betweenness_centrality"] = [
        betweenness.get(k, float("nan")) for k in result_keys
    ]
    result["graph_closeness_centrality"] = [
        closeness.get(k, float("nan")) for k in result_keys
    ]
    result["graph_eigenvector_centrality"] = [
        eigenvector.get(k, float("nan")) for k in result_keys
    ]
    result["graph_core_number"] = [core_num.get(k, float("nan")) for k in result_keys]
    result["graph_community_id"] = [
        node_to_community.get(k, float("nan")) for k in result_keys
    ]
    result["graph_in_lcc"] = [k in largest_cc for k in result_keys]

    return result
