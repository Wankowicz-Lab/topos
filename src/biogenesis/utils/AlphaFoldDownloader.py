import requests
import os

class AlphaFoldDownloader:
    """
    Automatically downloads protein structures from the AlphaFold DB.
    Enables biogenesis to proceed even when local structure files are missing.
    """
    BASE_URL = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"

    @staticmethod
    def download(uniprot_id, output_dir="data/structures"):
        os.makedirs(output_dir, exist_ok=True)
        url = AlphaFoldDownloader.BASE_URL.format(uniprot_id)
        print(f"Fetching AlphaFold model for {uniprot_id}...")
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(output_dir, f"{uniprot_id}.pdb")
            with open(file_path, "wb") as f:
                f.write(response.content)
            return file_path
        return None
