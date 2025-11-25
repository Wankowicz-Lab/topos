import numpy as np
import pandas as pd
import pytest

from tests.test_utils import _make_residue_table, _write_mmcif_file, _make_aaindex_data
from src.pipeline import runner

# import files containing metrics to register them in _REGISTRY
import src.sequence.metrics
import src.structure.metrics
from src.structure.structure_context import _REGISTRY


def test_runner_initialization(tmp_path):

    # Create a mock mutation dataset
    residue_table = _make_residue_table(
        num_chains=1,
        num_residues=10,
        start_resis=1,
        make_muts=True
    )
    mut_dataset = residue_table[['resn', 'resi', 'resm', 'effect', 'type']]
    mut_dataset = mut_dataset.rename(columns={
        'resn': 'wildtype',
        'resi': 'position',
        'resm': 'mutation',
        'effect': 'effect',
        'type': 'type'
    })

    mut_data_path = tmp_path / 'mut_data.csv'
    mut_dataset.to_csv(mut_data_path, index=False)

    pdb_id = '8smv'

    myrunner = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=None,
        membrane_protein=False,
        mutation_data_path=None,
        mutation_data_chain=None
    )

    assert myrunner.pdb_path is not None
    assert myrunner.pdb_ext == 'cif'
    assert myrunner.array is not None

    myrunner_membrane = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=None,
        membrane_protein=True,
        mutation_data_path=None,
        mutation_data_chain=None
    )

    assert 'pdbtm_region' in myrunner_membrane.context.residue_table.columns.tolist()
    assert 'pdbtm_region_detailed' in myrunner_membrane.context.residue_table.columns.tolist()

    # Create synthetic mmcif file to match mutation data
    residues = residue_table[['resn', 'resi']].drop_duplicates()['resn']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues.tolist()})

    myrunner_mut = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=mmcif_path,
        membrane_protein=True,
        mutation_data_path=mut_data_path,
        mutation_data_chain='A'
    )
    assert 'effect' in myrunner_mut.context.residue_table.columns.tolist()


def test_runner_run_metric_provides(tmp_path):
    # Create a mock mutation dataset
    residue_table = _make_residue_table(
        num_chains=1,
        num_residues=10,
        start_resis=1,
        make_muts=True
    )
    # add membrane region columns so that we don't need to integrate with PDBTM to test
    residue_table['pdbtm_region'] = 'membrane_spanning'
    residue_table['pdbtm_region_detailed'] = 'TM1'

    mut_dataset = residue_table[['resn', 'resi', 'resm', 'effect', 'type']]
    mut_dataset = mut_dataset.rename(columns={
        'resn': 'wildtype',
        'resi': 'position',
        'resm': 'mutation',
        'effect': 'effect',
        'type': 'type'
    })

    mut_data_path = tmp_path / 'mut_data.csv'
    mut_dataset.to_csv(mut_data_path, index=False)

    # Create synthetic mmcif file to match mutation data
    residues = residue_table[['resn', 'resi']].drop_duplicates()['resn']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues.tolist()})

    myrunner = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        membrane_protein=False,
        mutation_data_path=mut_data_path,
        mutation_data_chain='A'
    )

    # modify arguments for downstream metrics
    myrunner.context.membrane_protein = True
    myrunner.context.residue_table = residue_table
    myrunner.context.aaindex_data = _make_aaindex_data(accessions=['AA1', 'AA2'])

    # get metrics that are registered
    metrics = _REGISTRY.keys()

    # run each metric individually to ensure 'provides' columns are present
    for metric in metrics:
        meta, func = _REGISTRY[metric]
        provides, requires = meta.provides, meta.requires
        result = myrunner.run(metrics=[metric])

        returned_cols = result.columns.tolist()
        if metric == 'aa_index_scores':
            # aaindex scores add columns for each index
            expected_cols = ['chain', 'resi', 'resn', 'resm']
            for acc in ['AA1', 'AA2']:
                expected_cols.extend([f'AAIndex_{acc}_wt', f'AAIndex_{acc}_mut', f'AAIndex_{acc}_diff'])
        else:
            expected_cols = provides + ['chain', 'resi', 'resn']
            if 'resm' in requires:
                expected_cols.append('resm')

        assert set(expected_cols) == set(returned_cols)

    # run all metrics
    all_result = myrunner.run(metrics=list(metrics))


def test_runner__merge_features():
    pdb_id = '8smv'

    myrunner = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=None,
        membrane_protein=False,
        mutation_data_path=None,
        mutation_data_chain=None
    )

    residue_table = _make_residue_table(num_residues=6, num_chains=2, start_resis=[1,8], make_muts=[True, False])
    residue_table_no_muts = residue_table[['chain', 'resi', 'resn']].drop_duplicates().reset_index(drop=True)

    # df1 has residue level features from a subset of the residues
    df1_keep_resis = np.random.choice(residue_table_no_muts['resi'], size=7, replace=False)
    df1 = residue_table_no_muts[residue_table_no_muts['resi'].isin(df1_keep_resis)].copy()
    df1 = df1[['chain', 'resi', 'resn']]
    df1['feature1'] = np.random.rand(len(df1))

    # df2 has mutation-level features for all mutations from a subset of positions in chain A
    df2_keep_resis = np.random.choice(residue_table[residue_table['chain']=='A']['resi'], size=3, replace=False)
    df2 = residue_table[(residue_table['chain']=='A') & (residue_table['resi'].isin(df2_keep_resis))].copy()
    df2 = df2[['chain', 'resi', 'resn', 'resm']]
    df2['feature2'] = np.random.rand(len(df2))

    # df3 has residue level features from residues in Chain B
    df3_keep_resis = np.random.choice(residue_table_no_muts.loc[residue_table_no_muts['chain']=='B']['resi'],
                                      size=5, replace=False)
    df3 = residue_table_no_muts[residue_table_no_muts['resi'].isin(df3_keep_resis)].copy()
    df3 = df3[['chain', 'resi', 'resn']]
    df3['feature3'] = np.random.rand(len(df3))

    # df4 has a subset of mutations at each position
    df4_keep_resis = np.random.choice(len(residue_table), size=40, replace=False)
    df4 = residue_table.iloc[df4_keep_resis].copy()
    df4 = df4[['chain', 'resi', 'resn', 'resm']]
    df4['feature4'] = np.random.rand(len(df4))

    result_frames = [df1, df2, df3, df4]

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
    myrunner.context = MockContext(residue_table=residue_table)

    merged_df = myrunner._merge_features(result_frames, mutations=True)

    assert len(merged_df) == len(residue_table)

    # Check that df values are correctly merged
    df1_mask = merged_df['resi'].isin(df1_keep_resis)
    assert not merged_df.loc[df1_mask, 'feature1'].isnull().all()
    assert merged_df.loc[~df1_mask, 'feature1'].isnull().all()

    df2_mask = (merged_df['chain']=='A') & (merged_df['resi'].isin(df2_keep_resis))
    assert not merged_df.loc[df2_mask, 'feature2'].isnull().all()
    assert merged_df.loc[~df2_mask, 'feature2'].isnull().all()

    df3_mask = merged_df['resi'].isin(df3_keep_resis)
    assert not merged_df.loc[df3_mask, 'feature3'].isnull().all()
    assert merged_df.loc[~df3_mask, 'feature3'].isnull().all()

    df4 = df4.set_index(['chain', 'resi', 'resn', 'resm'])
    merged_df = merged_df.set_index(['chain', 'resi', 'resn', 'resm'])
    df4_mask = merged_df.index.isin(df4.index)
    assert not merged_df.loc[df4_mask, 'feature4'].isnull().all()
    assert merged_df.loc[~df4_mask,  'feature4'].isnull().all()


#
# # Create dataframes with different indices to test merging
# df1 = pd.DataFrame({
#     'chain': ['A', 'A', 'B'],
#     'resi': [1, 2, 1],
#     'resn': ['ALA', 'GLY', 'ARG'],
#     'feature_residue_level': [0.1, 0.2, 0.3]
# })
#
# df1['resm'] = None  # Add resm column with None values for merging
#
# # df2 has multiple mutations per position
# df2 = pd.DataFrame({
#     'chain': ['A', 'A', 'A', 'A', 'B'],
#     'resi': [1, 1, 2, 2, 1],
#     'resn': ['ALA', 'ALA', 'GLY', 'GLY', 'ARG'],
#     'resm': ['GLU', 'ASN', 'ASP', 'ARG', 'THR',],
#     'feature_mutations': [0.4, 0.5, 0.6, 0.7, 0.8]
# })
#
# # df3 has residue level features at more indices
# df3 = pd.DataFrame({
#     'chain': ['A', 'A', 'B', 'B'],
#     'resi': [1, 2, 1, 2],
#     'resn': ['ALA', 'GLY', 'ARG', 'SER'],
#     'feature_residue_2': [0.8, 0.9, 1.0, 1.1]
# })
# df3['resm'] = None  # Add resm column with None values for merging
#
# # df4 has mutations in new residues in chain A
# df4 = pd.DataFrame({
#     'chain': ['A', 'A', 'A', 'A', 'B'],
#     'resi': [1, 2, 3, 3, 1],
#     'resn': ['ALA', 'GLY', 'TRP', 'TRP', 'ARG'],
#     'resm': ['GLU', 'ASP', 'HIS', 'LYS', 'GLN'],
#     'feature_mutations_2': [1.2, 1.3, 1.4, 1.5, 1.6]
# })
#
# # create df with only metadata columns for all four dfs
# baseline_df = pd.DataFrame({
#     'chain': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B'],
#     'resi': [1, 1, 2, 2, 3, 3, 1, 2],
#     'resn': ['ALA', 'ALA', 'GLY', 'GLY', 'TRP', 'TRP', 'ARG', 'SER'],
#     'resm': ['GLU', 'ASN', 'ASP', 'ARG', 'HIS', 'LYS', 'THR', None],
# })
#
# result_frames = [baseline_df]
# result_frames.append(df1)
# result_frames.append(df2)
# result_frames.append(df3)
# result_frames.append(df4)
#
# from functools import reduce
# def merge_dfs_on_chain_resi_resn(dfs):
#     return reduce(lambda left, right: pd.merge(left, right, on=['chain', 'resi', 'resn', 'resm'], how='outer'), dfs)
#
# merged_df = merge_dfs_on_chain_resi_resn(result_frames)