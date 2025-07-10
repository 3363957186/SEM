#!/usr/bin/env bash
# this copyfile script is different because A4 only provides us with niftis. Therefore the folder structure is different.
# Assign options
while getopts ":d:i:m:" opt; do
  case $opt in
    d)
      main_dir=$OPTARG  # main directory
      ;;
    i)
      sessions=$OPTARG # rids
      ;;
    m)
      modality=$OPTARG #modality
      ;;
  esac
done

raw_dir="$main_dir"/raw

if [ -n "$sessions" ]; then
  ids=("$sessions")  # process specified id
else
  ids=($(ls "$raw_dir"))  # process all ids in the directory
fi

# Create output folder
outfolder="$main_dir"/processed/"$modality"_nii
if [[ -d "$outfolder" ]]; then
  echo "copying nifti files to "$outfolder""
else
  mkdir -p "$outfolder"
fi

cd "$raw_dir"
for id in "${ids[@]}"; do 
  echo "Processing $id"
  if [ "$modality" == "T1" ]; then 
    # nii_path=`find "$id" -maxdepth 3 -name '*T1.nii'`
    find $raw_dir/$id -type f -name '*.nii' | grep "T1" >> copy_path.csv
  elif [ "$modality" == "FLAIR" ]; then
    find $raw_dir/$id -type f -name  '*FLAIR*.nii' | grep "FLAIR" >> copy_path.csv
#   elif [ "$modality" == "Amyloid" ]; then
#     find $raw_dir/$id -type f -name  '*Amyloid.nii' >> copy_path.csv
#   elif [ "$modality" == "Tau" ]; then
#     find $raw_dir/$id -type f -name '*Tau.nii' >> copy_path.csv
  fi
done
cat copy_path.csv | while IFS=, read -r line; do
  path="$line"
  a4_unique="${path#${raw_dir}}"
  id=`echo "$a4_unique" | awk -F '/' '{print $2}'`
  date=`echo "$a4_unique" | awk -F '/' '{print $4}'`
  image=`echo "$a4_unique" | awk -F '/' '{print $5}'`
  new_filename="${id}_${date}_${image}_${modality}.nii"
  echo "Copying Files "$new_filename""
  destination="$outfolder/$new_filename"
  cp "$path" "$destination"
done
rm copy_path.csv