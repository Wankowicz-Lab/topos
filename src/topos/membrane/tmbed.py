"""TMbed adapter: structure sequences → transmembrane-helix regions."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from topos.sequence.utils import convert_amino_acid_3to1

logger = logging.getLogger(__name__)

# Format-1 label alphabet from TMbed (H/B strands, i/o sides, signal, coil).
_TMBED_LABEL_CHARS = frozenset("HhBbSs.io")


def _label_char_to_region_type(char: str) -> str:
    """Map one TMbed format-1 label character to a PDBTM-compatible region type."""
    if char in "Hh":
        return "transmembrane_helix"
    if char in "Bb":
        return "transmembrane_beta_strand"
    if char == "i":
        return "inside"
    if char == "o":
        return "outside"
    if char in ".Ss":
        return "unknown"
    raise ValueError(f"Unexpected TMbed label character: {char!r}")


class TmbedNotAvailable(RuntimeError):
    """Raised when the optional TMbed dependency/CLI is not installed."""


def extract_chain_sequences(
    residue_table: pd.DataFrame,
    chains: Sequence[str] | None = None,
) -> Dict[str, Tuple[str, List[int]]]:
    """
    Build per-chain sequences from structure residues sorted by residue number.

    Returns
    -------
    dict
        Mapping chain_id → (one-letter sequence, residue numbers aligned 1:1).
    """
    rt = residue_table
    if chains is not None:
        rt = rt[rt["chain"].isin(list(chains))]

    out: Dict[str, Tuple[str, List[int]]] = {}
    for chain_id, group in rt.groupby("chain", sort=True):
        ordered = group.sort_values("resi")
        resis = ordered["resi"].astype(int).tolist()
        # force_convert keeps one char per residue (e.g. MSE → X) so TMbed
        # FASTA stays aligned with resis / returned labels.
        letters = [
            convert_amino_acid_3to1(str(r), force_convert=True)
            for r in ordered["resn"].tolist()
        ]
        out[str(chain_id)] = ("".join(letters), resis)
    return out


def labels_to_regions(labels: str, chain: str, resis: Sequence[int]) -> pd.DataFrame:
    """
    Collapse contiguous TMbed label runs into region rows (TM, sides, unknown).

    Label and residue lists must be the same length (1:1 with the extracted sequence).
    H/h → transmembrane_helix, i/o → inside/outside, ./S/s → unknown, B/b → beta strand.
    """
    if len(labels) != len(resis):
        raise ValueError(
            f"TMbed label length ({len(labels)}) does not match residue count "
            f"({len(resis)}) for chain {chain}"
        )

    rows = []
    i = 0
    n = len(labels)
    while i < n:
        region_type = _label_char_to_region_type(labels[i])
        j = i + 1
        while j < n and _label_char_to_region_type(labels[j]) == region_type:
            j += 1
        rows.append(
            {
                "chain": chain,
                "type": region_type,
                "seq_beg": i + 1,
                "seq_end": j,
                "pdb_beg": int(resis[i]),
                "pdb_end": int(resis[j - 1]),
            }
        )
        i = j

    return pd.DataFrame(
        rows, columns=["chain", "type", "seq_beg", "seq_end", "pdb_beg", "pdb_end"]
    )


def parse_tmbed_format1(text: str) -> Dict[str, str]:
    """Parse TMbed format-1 (header / sequence / labels) into {id: label_string}."""
    lines = text.splitlines()
    out: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(">"):
            i += 1
            continue

        sid = line[1:].strip().split()[0]
        i += 1
        if i >= len(lines):
            break
        seq = lines[i].strip()
        i += 1
        if i >= len(lines) or lines[i].startswith(">"):
            raise ValueError(f"TMbed format-1 missing label line for {sid}")
        labels = lines[i].strip()
        i += 1

        if len(labels) != len(seq):
            raise ValueError(
                f"TMbed format-1 sequence/label length mismatch for {sid}: "
                f"{len(seq)} vs {len(labels)}"
            )
        if not set(labels) <= _TMBED_LABEL_CHARS:
            raise ValueError(f"Unexpected TMbed label characters for {sid}: {labels!r}")
        out[sid] = labels

    return out


def _resolve_tmbed_executable() -> str:
    path = shutil.which("tmbed")
    if path is None:
        raise TmbedNotAvailable(
            "TMbed is required to estimate membrane parameters but was not found. "
            "Install it (e.g. `pip install torch` then "
            "`pip install git+https://github.com/BernhoferM/TMbed.git`), ensure the "
            "`tmbed` CLI is on PATH, and run `tmbed download` once for ProtT5 models."
        )
    return path


def predict_tmbed_labels(
    sequences: Mapping[str, str],
    *,
    use_gpu: bool = False,
) -> Dict[str, str]:
    """
    Run TMbed predict on named sequences and return format-1 label strings.

    Uses the `tmbed` CLI (optional dependency). Raises TmbedNotAvailable if missing.
    """
    if not sequences:
        return {}

    tmbed_bin = _resolve_tmbed_executable()

    with tempfile.TemporaryDirectory(prefix="topos_tmbed_") as tmp:
        tmp_path = Path(tmp)
        fasta_path = tmp_path / "input.fasta"
        pred_path = tmp_path / "out.pred"

        fasta_lines = []
        for sid, seq in sequences.items():
            fasta_lines.append(f">{sid}")
            fasta_lines.append(seq)
        fasta_path.write_text("\n".join(fasta_lines) + "\n")

        cmd = [
            tmbed_bin,
            "predict",
            "-f",
            str(fasta_path),
            "-p",
            str(pred_path),
            "--out-format",
            "1",
            "--batch-size",
            "1000",
        ]
        if not use_gpu:
            cmd.append("--no-use-gpu")

        logger.info("Running TMbed predict for %d sequence(s)", len(sequences))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            raise RuntimeError(
                f"TMbed predict failed (exit {e.returncode}): {stderr or e}"
            ) from e

        return parse_tmbed_format1(pred_path.read_text())


def predict_tm_regions(
    residue_table: pd.DataFrame,
    chains: Sequence[str] | None = None,
    *,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """Extract sequences, run TMbed, and return region rows for all label runs."""
    chain_seqs = extract_chain_sequences(residue_table, chains=chains)
    labels_by_id = predict_tmbed_labels(
        {cid: seq for cid, (seq, _) in chain_seqs.items()},
        use_gpu=use_gpu,
    )

    frames = []
    for chain_id, (seq, resis) in chain_seqs.items():
        labels = labels_by_id.get(chain_id)
        if labels is None:
            raise RuntimeError(f"TMbed returned no prediction for chain {chain_id}")
        if len(labels) != len(seq):
            raise ValueError(
                f"TMbed label length ({len(labels)}) does not match sequence length "
                f"({len(seq)}) for chain {chain_id}"
            )
        frames.append(labels_to_regions(labels, chain_id, resis))

    if not frames:
        return pd.DataFrame(
            columns=["chain", "type", "seq_beg", "seq_end", "pdb_beg", "pdb_end"]
        )
    return pd.concat(frames, ignore_index=True)
