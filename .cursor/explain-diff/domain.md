# Domain contract (this repository)

Computes per-residue structural and sequence metrics from a PDB/mmCIF structure, optionally joined to deep mutational scanning (DMS) data as one row per mutation.

Canonical schemas and column lists: [README.md](../../README.md). Worked examples and fixtures: `examples/`, `tests/`. Do not restate those dictionaries here.

## Objects

- **Structure**: one PDB ID (RCSB) or local PDB/mmCIF; one or more chains. `structural_feature_chains` may restrict which chains get metrics.
- **Experimental / DMS WT sequence**: reconstructed from the mutation table’s wildtype letters at `position`; aligned to `mutation_data_chain`.
- **Mutation**: one input row per mutation (`position`, `wildtype`, `mutation`, `type`, `effect`).
- **Membrane vs soluble**: `membrane_protein` fetches PDBTM, orients the structure, and enables membrane-frame metrics.

Do not assume the deposited structure and the assay construct are the same isoform, numbering, or length — alignment exists because they often differ.

## Identifiers and numbering

These are different systems. Equal integers are not the same residue without a documented mapping.

| System | Where it lives | Notes |
| --- | --- | --- |
| Chain ID | PDB `chain` | DMS alignment uses `mutation_data_chain` only |
| `resi_mut` / input `position` | experimental WT numbering | Not PDB numbering |
| `resi_struct` | Biotite `AtomArray.res_id` | Typically PDB author residue numbers, not mmCIF `label_seq_id` |
| `align_pos` | pairwise alignment column | 0-based; a gap is NaN on one side |
| `resn_mut` / `resn_struct` | 3-letter residue names | A letter mismatch after mapping is a scientific event, not a join bug |

Insertion codes exist in mmCIF (`pdbx_PDB_ins_code`) but output tables key residues by `(chain, resi_struct)`. Alternate locations: `altloc_policy` `"highest"` (default, one conformer) vs `"all"` (one row per conformer).

If a change transforms positions, diagram:

experimental `position` → alignment column (`align_pos`) → `resi_struct` on `mutation_data_chain`

## Cardinality

- Structure-only features: **one row per residue** (or per residue×altloc if `"all"`).
- DMS-mode features: **one row per mutation**.
- Metadata: residue-level; `struct_info` / `mut_info` mark which side is present.
- Alignment gaps: NaN on the missing side. Mutations without a structural partner should keep the experimental row and leave structural features missing — not a shifted mapping.

Joins on `(chain, resi_struct)` or `(resi_mut, resn_mut)` can drop, duplicate, or silently realign rows if keys are non-unique (insertion codes, altlocs, duplicate mutation positions).

## Missingness

Do not equate `NaN`, `None`, `0`, and “not in the structure.”

Typical meanings here: residue unresolved or absent from the chain; terminal vs internal alignment gap; mutation with no structural partner; metric not computed (wrong mode, missing DSSP, non-membrane run); hydrogens or altlocs stripped.

## Units and versions

- Distances and SASA: Å / Å². `membrane_thickness` is half-thickness in Å.
- Secondary structure: `mkdssp` if on `PATH`, else `pydssp` — engine can change SS labels.
- Structure source: RCSB `pdb_id` vs local `pdb_path`; the assembly/version is whatever file was loaded.
- Sequence features: checked-in AAIndex / Kidera tables under `data/`.

## Silent failures (check when implicated)

- Joining mutation `position` to `resi_struct` without going through the alignment.
- Mixing mmCIF `label_seq_id` with author `res_id`.
- Two residues sharing `resi_struct` (insertion code) collapsed to one row.
- `altloc_policy` changing row counts or occupancy-weighted metrics.
- Alignment gaps shifting which mutation inherits which structural features.
- `resn_mut` ≠ `resn_struct` after mapping and being ignored.
- Membrane metrics without PDBTM orientation, or the reverse.
- Terminal gaps excluded from alignment-quality scoring but still present in the merged table.
