import os
import pandas as pd
from topos.pipeline.runner import Runner


def create_chimerax_file(output_df: pd.DataFrame, output_dir: str, attribute_name: str,
                         descriptive_text: str = "") -> None:
    """
    Create a ChimeraX .defattr file for visualizing annotations.

    Parameters
    ----------
    output_df : pd.DataFrame
        DataFrame containing 'chain', 'resi', and attribute_name columns.
    output_dir : str
        Directory to save the .defattr file.
    attribute_name : str
        Name of the attribute to visualize.
    descriptive_text : str
        Descriptive text for the attribute file (default is empty).

    Returns
    -------
    None
    """

    # headers for .defattr file
    key_values = {
        "attribute": attribute_name,
        "match mode": "1-to-1",
        "recipient": "residues"
    }

    with open(os.path.join(output_dir, f"{attribute_name}.defattr"), "w") as f:
        # write descriptive text if provided
        if descriptive_text:
            f.write(f"# {descriptive_text}\n")

        # write headers
        for k, v in key_values.items():
            f.write(f"{k}: {v}\n")

        # write attributes
        for _, row in output_df.iterrows():
            f.write(f"\t/{row.chain}:{int(row.resi_struct)}\t{row[attribute_name]}\n")


runner = Runner(
    name='B2AR_test',
    pdb_id='4LDE'
)

runner.run()
features = runner.features

subset_feature = 'ss_domains'
subset_features = features[['chain', 'resi_struct', 'resn_struct', subset_feature]]
subset_features = subset_features.drop_duplicates(subset=['chain', 'resi_struct', 'resn_struct'])
subset_features = subset_features.loc[subset_features.resi_struct.notna(), :]

create_chimerax_file(output_df=subset_features,
                    output_dir='/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/biogenesis',
                    attribute_name=subset_feature,
                    descriptive_text=f'{subset_feature} for 4LDE')
