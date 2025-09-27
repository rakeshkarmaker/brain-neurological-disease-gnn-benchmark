# config/paths.py
import os

# Base paths
BASE_DIR = "F:/Programming codes/Machine Learning/Projects/brain-disease-segmentation-benchmark"
DATA_DIR = os.path.join(BASE_DIR, "data")
OASIS_DIR = os.path.join(DATA_DIR, "oasis1")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# OASIS-specific paths
OASIS_CSV_PATH = os.path.join(OASIS_DIR, "oasis_cross-sectional.csv")

# Create directories if they don't exist
for directory in [PROCESSED_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"Ensured directory exists: {directory}")

# Function to get subject paths
def get_subject_paths(subject_id):
    subject_dir = os.path.join(OASIS_DIR, subject_id)
    gray_matter_path = os.path.join(subject_dir, f"mwrc1{subject_id}_mpr_anon_fslswapdim_bet.nii.gz")
    white_matter_path = os.path.join(subject_dir, f"mwrc2{subject_id}_mpr_anon_fslswapdim_bet.nii.gz")
    return gray_matter_path, white_matter_path