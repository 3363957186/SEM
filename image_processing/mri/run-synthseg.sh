#!/usr/bin/env bash
# Assign options
# conda activate synthseg_38

while getopts ":i:o:" opt; do
  case $opt in
    i)
      inpath=$OPTARG  # CSV file with paths to images to process
      ;;
    o)
      outpath=$OPTARG # Output directory
      ;;
  esac
done

# ./run-synthseg.sh -i /projectnb/vkolagrp/varuna/A4/processed/FLAIR_nii/flair_paths.csv -o /projectnb/vkolagrp/varuna/synthseg/A4/FLAIR
# ./run-synthseg.sh -i /projectnb/vkolagrp/varuna/A4/processed/T2star/t2st_paths.csv -o /projectnb/vkolagrp/varuna/synthseg/A4/T2star
# ./run-synthseg.sh -i /projectnb/vkolagrp/varuna/ADNI/processed/FLAIR_nii/flair_paths.csv -o /projectnb/vkolagrp/varuna/synthseg/ADNI/FLAIR
# ./run-synthseg.sh -i /projectnb/vkolagrp/varuna/ADNI/processed/T2st_nii/t2st_paths.csv -o /projectnb/vkolagrp/varuna/synthseg/ADNI/T2star
# ./run-synthseg.sh -i /projectnb/vkolagrp/varuna/HABS/processed/FLAIR_nii/flair_paths.csv -o /projectnb/vkolagrp/varuna/synthseg/HABS/FLAIR

mkdir -p "$outpath"

export TF_FORCE_GPU_ALLOW_GROWTH=true

while IFS= read -r image_path; do
  # extract filename
  base_name=$(basename "$image_path")
  file_name="${base_name%.*}" # removes file extension

  # make output paths
  output_subj="$outpath/${file_name}"
  if [ -d "$output_subj" ]; then
    echo "Subject "$output_subj" already processed. Skipping"
    continue
  else
    mkdir -p "$output_subj"
    parc_file="$output_subj/${file_name}_parc.csv"
    qc_file="$output_subj/${file_name}_qc.csv"
    post_file="$output_subj/maps"
    resample_dir="$output_subj"
    # run SynthSeg if the folder doesn't exist
    python /projectnb/vkolagrp/varuna/mri_pet/SynthSeg/scripts/commands/SynthSeg_predict.py --i "$image_path" --o "$output_subj" --parc --vol "$parc_file" --qc "$qc_file" --resample "$resample_dir" 
  fi
done < "$inpath"

# for folder in "$PWD"/*/; do
#   # Check if the folder contains any file ending with 'synthseg.nii'
#   if ! find "$folder" -type f -name '*synthseg.nii' | grep -q .; then
#     # If no such file is found, delete the folder
#     echo "Deleting folder: $folder" >> not_processed.csv
#     rm -rf "$folder"
#   else
#     echo "Keeping folder: $folder"
#   fi
# done