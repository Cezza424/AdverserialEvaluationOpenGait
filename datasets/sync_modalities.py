import os
import pickle
from pathlib import Path

# Set the path to your combined directory
base_path = Path(r"C:\Users\25299867\Downloads\HealthGait\silhouette_Heatmap")

# Keep track of how many sequences were modified
modified_count = 0

print("Scanning for mismatched modalities...")

# Traverse the directory tree
for seq_dir in base_path.rglob('*'):
    if seq_dir.is_dir():
        sil_path = seq_dir / "1_sil.pkl"
        heat_path = seq_dir / "0_heatmap.pkl"

        # Only process folders that contain both files
        if sil_path.exists() and heat_path.exists():

            # Open and load both pickle files
            with open(sil_path, 'rb') as f:
                sil_data = pickle.load(f)
            with open(heat_path, 'rb') as f:
                heat_data = pickle.load(f)

            len_sil = len(sil_data)
            len_heat = len(heat_data)

            # If lengths do not match, synchronise them
            if len_sil != len_heat:
                min_len = min(len_sil, len_heat)

                # Slice arrays to the minimum length
                sil_data_synced = sil_data[:min_len]
                heat_data_synced = heat_data[:min_len]

                # Overwrite the original files with the synchronised arrays
                with open(sil_path, 'wb') as f:
                    pickle.dump(sil_data_synced, f)
                with open(heat_path, 'wb') as f:
                    pickle.dump(heat_data_synced, f)

                # Create a readable folder path for the print statement (e.g., PA001/FGS/WJ_1)
                sequence_name = "/".join(seq_dir.parts[-3:])
                print(
                    f"Synchronised {sequence_name}: Silhouette ({len_sil}) vs Heatmap ({len_heat}) -> Trimmed to {min_len}")
                modified_count += 1

print(f"\nProgramme complete. {modified_count} sequences were synchronised.")