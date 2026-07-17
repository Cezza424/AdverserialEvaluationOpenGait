import os
import json
import pickle
import numpy as np
from pathlib import Path


def convert_json_to_pkl(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Standard 17-keypoint COCO format expected by OpenGait
    expected_joints = [
        "nose", "l_eye", "r_eye", "l_ear", "r_ear",
        "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
        "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle"
    ]

    # Find all JSON files in the input directory
    json_files = list(input_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files to process.")

    for json_file in json_files:
        parts = json_file.parts

        # Ensure the path contains at least Subject/Sequence/File.json
        if len(parts) < 3:
            continue

        subject = parts[-3]
        sequence = parts[-2]

        # Clean up the filename to get the view (e.g., "WJ_1_AlphaPose.json" -> "WJ_1")
        view = parts[-1].replace("_AlphaPose.json", "").replace(".json", "")

        with open(json_file, 'r') as f:
            data = json.load(f)

        sequence_frames = []

        # Iterate through the list of frames
        for frame_data in data:
            joints_dict = frame_data.get("joints")

            # Skip frames where no person was detected (joints is null)
            if joints_dict is None:
                continue

            frame_joints = []
            for joint_name in expected_joints:
                if joint_name in joints_dict and joints_dict[joint_name] is not None:
                    x = joints_dict[joint_name]["x"]
                    y = joints_dict[joint_name]["y"]
                    score = 1.0  # OpenGait expects a confidence score; we default to 1.0
                else:
                    # Fallback for missing joints
                    x, y, score = 0.0, 0.0, 0.0

                frame_joints.append([x, y, score])

            sequence_frames.append(frame_joints)

        # OpenGait generally ignores sequences with fewer than 5 valid frames
        if len(sequence_frames) < 5:
            print(f"Skipping {subject}/{sequence}/{view} - only {len(sequence_frames)} valid frames.")
            continue

        # Convert to numpy array of shape (Frames, 17, 3)
        pose_array = np.array(sequence_frames, dtype=np.float32)

        # Create the OpenGait output directory structure: output_path/Subject/Sequence/
        save_dir = output_path / subject / sequence
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save as .pkl
        pkl_path = save_dir / f"{view}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(pose_array, f)

        print(f"Successfully saved {pose_array.shape} to {pkl_path}")


if __name__ == "__main__":
    # Define your paths here
    INPUT_DIRECTORY = r"C:\Users\25299867\Downloads\HealthGait\pose"
    OUTPUT_DIRECTORY = r"C:\Users\25299867\Downloads\HealthGait\posePKL"

    convert_json_to_pkl(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    print("Conversion complete.")