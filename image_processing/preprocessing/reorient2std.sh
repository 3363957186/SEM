#!/bin/bash
# reorient 2 mni standard space

# Assign options
while getopts ":i:o:" opt; do
  case $opt in
    i)
      inpath=$OPTARG  # input directory # /SeaExp_1/MRI_PET/GAAIN/processed/T1_nii or /SeaExp_1/MRI_PET/GAAIN/processed/Amyloid_nii
      ;;
    o)
      outpath=$OPTARG # output folder # /SeaExp_1/MRI_PET/GAAIN/processed/reoriented/Amyloid or /SeaExp_1/MRI_PET/GAAIN/processed/reoriented/T1
      ;;
  esac
done

mkdir -p "$outpath"

for input in $inpath/*.nii; do
  echo $input
  fslreorient2std $input $outpath/$(basename "$input" .nii)_reoriented.nii
done

