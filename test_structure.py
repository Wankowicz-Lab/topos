#!/usr/bin/env python3
"""
Load a PDB file and output the baseline DataFrame with residue information,
SASA, and secondary structure.
"""

import sys
import argparse
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from structure.structure_context import Context, load_structure_with_id


def main():
    parser = argparse.ArgumentParser(
        description="Load a PDB file and output the baseline DataFrame"
    )
    parser.add_argument(
        "pdb_file",
        type=str,
        help="Path to the PDB file to process"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (CSV). If not specified, prints to stdout"
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        help="Model number to load (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    pdb_path = Path(args.pdb_file)
    if not pdb_path.exists():
        print(f"Error: PDB file not found: {pdb_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load and process structure
        print(f"Loading PDB file: {pdb_path}", file=sys.stderr)
        arr, pdb_id = load_structure_with_id(pdb_path, model=args.model)
        print(f"Extracted PDB ID: {pdb_id}", file=sys.stderr)
        ctx = Context(arr, pdb_id=pdb_id)
        
        # Output baseline DataFrame
        if args.output:
            output_path = Path(args.output)
            ctx.baseline_df.to_csv(output_path, index=False)
            print(f"Baseline DataFrame saved to: {output_path}", file=sys.stderr)
            print(f"Shape: {ctx.baseline_df.shape[0]} residues × {ctx.baseline_df.shape[1]} columns", file=sys.stderr)
        else:
            # Print to stdout
            print(ctx.baseline_df.to_string())
        
    except Exception as e:
        print(f"Error processing PDB file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
