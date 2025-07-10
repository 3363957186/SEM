#!/usr/bin/env bash
# #ADNI
# DATA_DIR="/SeaExp_1-ayan/MRI_PET/ADNI/processed/T1_nii"
# CSV_FILE="/SeaExp_1-ayan/MRI_PET/ADNI/processed/T1_nii/ADNI_T1.csv"
# #Output directory for FastSurfer
# OUTPUT_DIR="/SeaExp_1-ayan/MRI_PET/ADNI/processed/fastsurfer"

# A4
# DATA_DIR="/SeaExp_1-ayan/MRI_PET/A4/processed/T1_nii" # folder to data
# OUTPUT_DIR="/SeaExp_1-ayan/MRI_PET/A4/processed/fastsurfer"
# CSV_FILE="/SeaExp_1-ayan/MRI_PET/A4/processed/T1_nii/A4_T1.csv"

# AIBL
# DATA_DIR="/SeaExp_1-ayan/MRI_PET/AIBL/processed/T1_nii"
# OUTPUT_DIR="/SeaExp_1-ayan/MRI_PET/AIBL/processed/fastsurfer"
# CSV_FILE="/SeaExp_1-ayan/MRI_PET/AIBL/processed/T1_nii/AIBL_T1.csv"

# # OASIS3
# DATA_DIR="/SeaExp_1-ayan/MRI_PET/OASIS3/processed/T1_nii"
# OUTPUT_DIR="/SeaExp_1-ayan/MRI_PET/OASIS3/processed/fastsurfer"
# CSV_FILE="/SeaExp_1-ayan/MRI_PET/OASIS3/processed/T1_nii/OASIS3_T1.csv"

# # FHS
# DATA_DIR="/encryptedfs/varuna/data/processed/T1_nii"
# OUTPUT_DIR="/encryptedfs/varuna/data/processed/fastsurfer"
# CSV_FILE="/encryptedfs/varuna/data/processed/T1_nii/FHS_T1.csv"

# # HABS
# DATA_DIR="/SeaExp_1-ayan/MRI_PET/HABS/raw/T1/ADNI_MPRAGE"
# OUTPUT_DIR="/SeaExp_1-ayan/MRI_PET/HABS/processed/fastsurfer"
# CSV_FILE="/SeaExp_1-ayan/MRI_PET/HABS/raw/T1/ADNI_MPRAGE/HABS_T1_ADNIMPRAGE.csv"

# NACC
DATA_DIR="/SeaExp_1-ayan/MRI_PET/NACC/processed/T1_nii"
OUTPUT_DIR="/SeaExp_1-ayan/MRI_PET/NACC/processed/fastsurfer"
CSV_FILE="/SeaExp_1-ayan/MRI_PET/NACC/processed/T1_nii/NACC_T1.csv"

cat "$CSV_FILE" | while read t1_image; do
    subject_id=$(basename "${t1_image}" ".nii")
    # Run FastSurfer docker command
    docker run --gpus device=3 \
        -v "${DATA_DIR}":/data \
        -v "${OUTPUT_DIR}":/output \
        -v /SeaExp_1-ayan/license.txt:/fs_license \
        --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
        --fs_license /fs_license \
        --sd /output \
        --sid "${subject_id}" \
        --t1 /data/$(basename "${t1_image}") \
        --seg_only
done