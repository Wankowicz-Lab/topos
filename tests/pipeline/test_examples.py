from src.pipeline.runner import Runner
import pandas as pd
import pytest

def test_b2ar_example(tmp_path):
    b2ar_dir = 'examples/B2AR_DMS_example'
    runner = Runner(
        name='B2AR_test',
        config_path=f'{b2ar_dir}/B2AR_config.toml'
    )

    assert runner.context.extras['mutation_data'] is not None
    assert {'resi_struct', 'resi_mut', 'pdbtm_region'}.issubset(runner.context.residue_table.columns)

    runner.run()
    assert runner.features is not None
    assert {'packing_contact_density', 'sasa', 'blosum90'}.issubset(runner.features.columns)

    output_dir = tmp_path / "b2ar_test_output"
    runner.save_results(output_dir=output_dir)
    assert (output_dir / "4LDE_features.csv").exists()
    assert (output_dir / "4LDE_metadata.csv").exists()

    saved_features = pd.read_csv(output_dir / "4LDE_features.csv")
    assert 'packing_contact_density' in saved_features.columns
    assert 'sasa' in saved_features.columns
    assert 'blosum90' in saved_features.columns
    assert len(saved_features) > 2000

    saved_metadata = pd.read_csv(output_dir / "4LDE_metadata.csv")
    assert 'resi_struct' in saved_metadata.columns
    assert 'resi_mut' in saved_metadata.columns
    assert 'pdbtm_region' in saved_metadata.columns
    assert len(saved_metadata) > 200