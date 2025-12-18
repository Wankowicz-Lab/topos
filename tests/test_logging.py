"""Tests for logging functionality across the biogenesis pipeline."""
import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from tests.test_utils import _make_config_file, _write_mmcif_file, _make_aaindex_data


def test_logging_appears_at_info_level(tmp_path, caplog):
    """Test that INFO logs appear when logging level is set to INFO."""
    # Configure logging to INFO level
    caplog.set_level(logging.INFO)
    
    # Import after setting logging level
    from src.pipeline import runner
    
    # Create a synthetic PDB file
    pdb_path = tmp_path / '8smv.cif'
    _write_mmcif_file(pdb_path, {'A': ['ALA', 'GLY', 'SER']}, '8smv')
    
    # Create AAIndex data
    aaindex_path = tmp_path / 'aaindex.csv'
    aaindex_data = _make_aaindex_data(['FASG890101', 'FASG890102'])
    aaindex_data.to_csv(aaindex_path, index=False)
    
    # Create kidera data
    kidera_path = tmp_path / 'kidera.csv'
    kidera_data = pd.DataFrame({
        'factor': [1, 2],
        'description': ['Factor 1', 'Factor 2'],
        **{aa: [1.0, 2.0] for aa in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 
                                      'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'ARG']}
    })
    kidera_data.to_csv(kidera_path, index=False)
    
    # Create a config file with local PDB path
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, pdb_id='8smv', name='test_protein', mutation_data_chain=None, 
                     aaindex_path=aaindex_path, mutation_data_path="")
    
    # Update config to include paths
    import tomli_w
    config_dict = {
        'pdb_id': '8smv',
        'name': 'test_protein',
        'pdb_path': str(pdb_path),
        'membrane_protein': False,
        'aaindex_path': str(aaindex_path),
        'kidera_path': str(kidera_path)
    }
    with config_path.open("wb") as f:
        tomli_w.dump(config_dict, f)
    
    # Create runner (should generate INFO logs)
    test_runner = runner.Runner(config_path=config_path)
    
    # Verify INFO logs were captured
    assert any('Initializing pipeline' in record.message for record in caplog.records)
    assert any('Loading configuration' in record.message for record in caplog.records)
    assert any('Configuration loaded successfully' in record.message for record in caplog.records)
    assert any('Using local PDB file' in record.message for record in caplog.records)
    assert any('Loading structure' in record.message for record in caplog.records)
    assert any('Structure loaded' in record.message for record in caplog.records)
    assert any('Creating context object' in record.message for record in caplog.records)
    assert any('Context object created successfully' in record.message for record in caplog.records)


def test_logging_not_appears_at_warning_level(tmp_path, caplog):
    """Test that INFO logs do NOT appear when logging level is set to WARNING."""
    # Configure logging to WARNING level (default)
    caplog.set_level(logging.WARNING)
    
    # Import after setting logging level
    from src.pipeline import runner
    
    # Create a synthetic PDB file
    pdb_path = tmp_path / '8smv.cif'
    _write_mmcif_file(pdb_path, {'A': ['ALA', 'GLY', 'SER']}, '8smv')
    
    # Create AAIndex data
    aaindex_path = tmp_path / 'aaindex.csv'
    aaindex_data = _make_aaindex_data(['FASG890101', 'FASG890102'])
    aaindex_data.to_csv(aaindex_path, index=False)
    
    # Create kidera data
    kidera_path = tmp_path / 'kidera.csv'
    kidera_data = pd.DataFrame({
        'factor': [1, 2],
        'description': ['Factor 1', 'Factor 2'],
        **{aa: [1.0, 2.0] for aa in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 
                                      'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'ARG']}
    })
    kidera_data.to_csv(kidera_path, index=False)
    
    # Create a config file with local PDB path
    config_path = tmp_path / 'config.toml'
    import tomli_w
    config_dict = {
        'pdb_id': '8smv',
        'name': 'test_protein',
        'pdb_path': str(pdb_path),
        'membrane_protein': False,
        'aaindex_path': str(aaindex_path),
        'kidera_path': str(kidera_path)
    }
    with config_path.open("wb") as f:
        tomli_w.dump(config_dict, f)
    
    # Create runner (should NOT generate visible INFO logs)
    test_runner = runner.Runner(config_path=config_path)
    
    # Verify INFO logs were NOT captured (only WARNING and above)
    info_logs = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_logs) == 0, "INFO logs should not appear at WARNING level"


def test_metric_calculation_logging(tmp_path, caplog):
    """Test that metric calculations produce appropriate INFO logs."""
    caplog.set_level(logging.INFO)
    
    from src.pipeline import runner
    
    # Create a synthetic PDB file
    pdb_path = tmp_path / '8smv.cif'
    _write_mmcif_file(pdb_path, {'A': ['ALA', 'GLY', 'SER']}, '8smv')
    
    # Create AAIndex data
    aaindex_path = tmp_path / 'aaindex.csv'
    aaindex_data = _make_aaindex_data(['FASG890101', 'FASG890102'])
    aaindex_data.to_csv(aaindex_path, index=False)
    
    # Create kidera data
    kidera_path = tmp_path / 'kidera.csv'
    kidera_data = pd.DataFrame({
        'factor': [1, 2],
        'description': ['Factor 1', 'Factor 2'],
        **{aa: [1.0, 2.0] for aa in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 
                                      'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'ARG']}
    })
    kidera_data.to_csv(kidera_path, index=False)
    
    # Create a config file with local PDB path
    config_path = tmp_path / 'config.toml'
    import tomli_w
    config_dict = {
        'pdb_id': '8smv',
        'name': 'test_protein',
        'pdb_path': str(pdb_path),
        'membrane_protein': False,
        'aaindex_path': str(aaindex_path),
        'kidera_path': str(kidera_path)
    }
    with config_path.open("wb") as f:
        tomli_w.dump(config_dict, f)
    
    # Create runner and run a simple metric
    test_runner = runner.Runner(config_path=config_path)
    test_runner.run(metrics=['sasa'])
    
    # Verify metric-specific logs
    assert any('Calculating metric: sasa' in record.message for record in caplog.records)
    assert any('Calculating SASA for' in record.message for record in caplog.records)
    assert any('SASA calculation completed' in record.message for record in caplog.records)
    assert any('Merging features' in record.message for record in caplog.records)
    assert any('Features merged successfully' in record.message for record in caplog.records)


def test_save_results_logging(tmp_path, caplog):
    """Test that saving results produces appropriate INFO logs."""
    caplog.set_level(logging.INFO)
    
    from src.pipeline import runner
    
    # Create a synthetic PDB file
    pdb_path = tmp_path / '8smv.cif'
    _write_mmcif_file(pdb_path, {'A': ['ALA', 'GLY', 'SER']}, '8smv')
    
    # Create AAIndex data
    aaindex_path = tmp_path / 'aaindex.csv'
    aaindex_data = _make_aaindex_data(['FASG890101', 'FASG890102'])
    aaindex_data.to_csv(aaindex_path, index=False)
    
    # Create kidera data
    kidera_path = tmp_path / 'kidera.csv'
    kidera_data = pd.DataFrame({
        'factor': [1, 2],
        'description': ['Factor 1', 'Factor 2'],
        **{aa: [1.0, 2.0] for aa in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 
                                      'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'ARG']}
    })
    kidera_data.to_csv(kidera_path, index=False)
    
    # Create a config file with local PDB path
    config_path = tmp_path / 'config.toml'
    import tomli_w
    config_dict = {
        'pdb_id': '8smv',
        'name': 'test_protein',
        'pdb_path': str(pdb_path),
        'membrane_protein': False,
        'aaindex_path': str(aaindex_path),
        'kidera_path': str(kidera_path)
    }
    with config_path.open("wb") as f:
        tomli_w.dump(config_dict, f)
    
    # Create runner, run metrics, and save
    test_runner = runner.Runner(config_path=config_path)
    test_runner.run(metrics=['sasa'])
    
    output_dir = tmp_path / 'output'
    test_runner.save_results(output_dir=output_dir)
    
    # Verify save-specific logs
    assert any('Features saved to:' in record.message for record in caplog.records)
    assert any('Metadata saved to:' in record.message for record in caplog.records)


def test_logger_naming_convention():
    """Test that loggers use the correct hierarchical naming convention."""
    from src.pipeline import runner
    from src.pipeline import batch_processing
    from src.structure import metrics
    from src.sequence import metrics as seq_metrics
    from src.structure import pdbtm
    from src.sequence import sequence_context
    from src.structure import structure_context
    
    # Verify logger names match module names
    assert runner.logger.name == 'src.pipeline.runner'
    assert batch_processing.logger.name == 'src.pipeline.batch_processing'
    assert metrics.logger.name == 'src.structure.metrics'
    assert seq_metrics.logger.name == 'src.sequence.metrics'
    assert pdbtm.logger.name == 'src.structure.pdbtm'
    assert sequence_context.logger.name == 'src.sequence.sequence_context'
    assert structure_context.logger.name == 'src.structure.structure_context'


def test_pipeline_runs_with_logging_enabled(tmp_path, caplog):
    """Test that the pipeline runs correctly with logging enabled."""
    caplog.set_level(logging.INFO)
    
    from src.pipeline import runner
    
    # Create a synthetic PDB file
    pdb_path = tmp_path / '8smv.cif'
    _write_mmcif_file(pdb_path, {'A': ['ALA', 'GLY', 'SER']}, '8smv')
    
    # Create AAIndex data
    aaindex_path = tmp_path / 'aaindex.csv'
    aaindex_data = _make_aaindex_data(['FASG890101', 'FASG890102'])
    aaindex_data.to_csv(aaindex_path, index=False)
    
    # Create kidera data
    kidera_path = tmp_path / 'kidera.csv'
    kidera_data = pd.DataFrame({
        'factor': [1, 2],
        'description': ['Factor 1', 'Factor 2'],
        **{aa: [1.0, 2.0] for aa in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 
                                      'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'ARG']}
    })
    kidera_data.to_csv(kidera_path, index=False)
    
    # Create a config file with local PDB path
    config_path = tmp_path / 'config.toml'
    import tomli_w
    config_dict = {
        'pdb_id': '8smv',
        'name': 'test_protein',
        'pdb_path': str(pdb_path),
        'membrane_protein': False,
        'aaindex_path': str(aaindex_path),
        'kidera_path': str(kidera_path)
    }
    with config_path.open("wb") as f:
        tomli_w.dump(config_dict, f)
    
    # Create runner and run
    test_runner = runner.Runner(config_path=config_path)
    test_runner.run(metrics=['sasa', 'kyte_doolittle'])
    
    # Verify pipeline completed successfully
    assert hasattr(test_runner, 'features')
    assert len(test_runner.features) > 0
    assert 'sasa' in test_runner.features.columns
    assert 'kyte_doolittle' in test_runner.features.columns
    
    # Verify logs were generated
    assert len(caplog.records) > 0
    assert any('Initializing pipeline' in record.message for record in caplog.records)

