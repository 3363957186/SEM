#!/usr/bin/env bash

## This script will copy all nifti files from the raw folder to the processed folder

# Assign options
while getopts ":d:m:t:" opt; do
  case $opt in
    d)
      main_dir=$OPTARG  # main GAAIN Tracer directory
      ;;
    m)
      modality=$OPTARG  modality Amyloid or T1
      ;;
    t)
      tracer=$OPTARG # tracer that we are calibrating (PiB, FLU, NAV or FBP)
      ;;
  esac
done

raw_dir="$main_dir"/raw

# Create output folder
outfolder="$main_dir"/processed/"$modality"_nii
if [[ -d "$outfolder" ]]; then
  echo "copying nifti files to "$outfolder""
else
  mkdir -p "$outfolder"
fi

cd "$raw_dir"
if [ "$modality" == "T1" ] && [ "$tracer" == "FLU" ]; then
  find $PWD -type f -name "*MR*nii" >> copy_path.csv
elif [ "$modality" == "Amyloid" ] && [ "$tracer" == "PiB" ]; then
  find $PWD -type f -name "*PiB*nii" >> copy_path.csv
elif [ "$modality" == "Amyloid" ] && [ "$tracer" == "FLU" ]; then
  find $PWD -type f -name "*C11*nii" >> copy_path.csv
  find $PWD -type f -name "*F18*nii" >> copy_path.csv
fi

cat copy_path.csv | while read line; do
  if [ "$modality" == "T1" ] && [ "$tracer" == "FLU" ]; then
    filename=$(basename "$line")
    tracer_part=`echo "$filename" | awk -F '_' '{print $4}'`
    group_part=`echo "$filename" | awk -F '_' '{print $2}'`
    if [ "$group_part" != "AD" ]; then
      group_part="YC"
    fi
    num_part=`echo "$filename" | awk -F '_' '{print $3}'`
    rid="$group_part""$num_part"
    new_name="$rid"_"$tracer_part".nii
    echo "$new_name"
    cp "$line" "$outfolder"/$"$new_name"
  elif [ "$modality" == "Amyloid" ] && [ "$tracer" == "FLU" ]; then
    filename=$(basename "$line")
    tracer_part=`echo "$filename" | awk -F '_' '{print $2}'`
    if [[ $tracer_part == *C11* ]]; then
      group_part=`echo "$filename" | awk -F '_' '{print $3}'`
      num_part=`echo "$filename" | awk -F '_' '{print $4}'`
    else
      group_part=`echo "$filename" | awk -F '_' '{print $4}'`
      num_part=`echo "$filename" | awk -F '_' '{print $5}'`
      tracer_part="F18FMT"
    fi
    if [ "$group_part" != "AD" ]; then
      group_part="YC"
    fi
    rid="$group_part""$num_part"
    new_name="$rid"_"$tracer_part".nii
    echo "$new_name"
    cp "$line" "$outfolder"/$"$new_name"
  else
    cp "$line" "$outfolder"
  fi
done
rm copy_path.csv