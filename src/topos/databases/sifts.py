"""PDBe SIFTS helpers: map PDB structures to UniProt accessions."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

SIFTS_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"

UNIPROT_CONFIG_HINT = (
    'Set uniprot_id in your config to provide the accession explicitly '
    '(e.g. uniprot_id = "P02699"), or set structural_feature_chains so UniProt '
    'can be inferred via SIFTS from pdb_id.'
)


class UniprotLookupError(LookupError):
    """Raised when a UniProt accession cannot be resolved for a PDB id."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def fetch_sifts_uniprot_mappings(pdb_id: str) -> Dict[str, Any]:
    """Fetch SIFTS UniProt mappings JSON for a PDB id."""
    url = SIFTS_URL.format(pdb=pdb_id.lower())
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise UniprotLookupError(f"SIFTS request failed for PDB {pdb_id}: {e}") from e

    # Decode/validate at this I/O boundary so callers can soft-fail on bad bodies.
    try:
        payload = response.json()
    except ValueError as e:
        raise UniprotLookupError(
            f"SIFTS returned invalid JSON for PDB {pdb_id}: {e}"
        ) from e
    if not isinstance(payload, dict):
        raise UniprotLookupError(
            f"SIFTS returned non-object JSON for PDB {pdb_id}: {type(payload).__name__}"
        )
    return payload


def _root_for_pdb(sifts_json: Mapping[str, Any], pdb_id: str) -> Mapping[str, Any]:
    return (
        sifts_json.get(pdb_id.lower())
        or sifts_json.get(pdb_id.upper())
        or next(iter(sifts_json.values()), {})
    )


def pick_uniprot_for_chains(
    sifts_json: Mapping[str, Any],
    pdb_id: str,
    chains: Sequence[str] | None = None,
) -> Tuple[str, List[dict]]:
    """
    Choose a UniProt accession for the given PDB chains.

    Prefers accessions covering ``chains`` when provided; otherwise all chains.
    Among candidates, picks the accession with the largest mapped residue coverage.
    """
    root = _root_for_pdb(sifts_json, pdb_id)
    uni = root.get("UniProt") or {}
    chain_filter = {str(c) for c in chains} if chains else None

    by_acc: Dict[str, List[dict]] = {}
    for acc, payload in uni.items():
        for mapping in payload.get("mappings", []):
            chain_id = str(mapping.get("chain_id"))
            if chain_filter is not None and chain_id not in chain_filter:
                continue
            by_acc.setdefault(acc, []).append(
                {
                    "accession": acc,
                    "chain_id": chain_id,
                    "unp_start": mapping["unp_start"],
                    "unp_end": mapping["unp_end"],
                    "pdb_start": mapping["start"]["residue_number"],
                    "pdb_end": mapping["end"]["residue_number"],
                    "coverage": abs(mapping["unp_end"] - mapping["unp_start"]) + 1,
                }
            )

    if not by_acc:
        scope = (
            f"chains {sorted(chain_filter)}"
            if chain_filter is not None
            else "any chain"
        )
        raise UniprotLookupError(
            f"SIFTS returned no UniProt mapping for PDB {pdb_id} ({scope})"
        )

    def coverage(acc: str) -> int:
        return sum(seg["coverage"] for seg in by_acc[acc])

    best_acc = max(by_acc.keys(), key=coverage)
    return best_acc, by_acc[best_acc]


def resolve_uniprot_accession(
    *,
    pdb_id: Optional[str],
    config_uniprot_id: Optional[str] = None,
    chains: Sequence[str] | None = None,
) -> Tuple[str, str]:
    """
    Resolve UniProt accession from config or SIFTS.

    SIFTS inference requires non-empty ``chains`` (typically
    ``structural_feature_chains``) so the accession is scoped to the
    membrane chain(s) of interest.

    Returns
    -------
    tuple
        ``(accession, source)`` where source is ``"config"`` or ``"sifts"``.
    """
    if config_uniprot_id:
        return str(config_uniprot_id).strip(), "config"

    if not pdb_id:
        raise UniprotLookupError(
            "No uniprot_id in config and no pdb_id available for SIFTS lookup"
        )

    if not chains:
        raise UniprotLookupError(
            "No uniprot_id in config and structural_feature_chains is unset; "
            "SIFTS UniProt inference requires structural_feature_chains"
        )

    sifts_json = fetch_sifts_uniprot_mappings(pdb_id)
    accession, _segments = pick_uniprot_for_chains(sifts_json, pdb_id, chains=chains)
    logger.info(
        "Resolved UniProt %s for PDB %s via SIFTS (chains=%s)",
        accession,
        pdb_id,
        list(chains),
    )
    return accession, "sifts"
