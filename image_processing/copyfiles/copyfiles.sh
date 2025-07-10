#!/usr/bin/env bash

# Assign options
while getopts ":d:c:i:m:" opt; do
  case $opt in
    d)
      main_dir=$OPTARG  # main directory
      ;;
    c)
      cohort=$OPTARG  # cohort
      ;;
    i)
      sessions=$OPTARG # ids
      ;;
    m)
      modality=$OPTARG # modality
      ;;
  esac
done

raw_dir="$main_dir/raw"

if [ -n "$sessions" ]; then
  ids=("$sessions")  # process specified id
else
  ids=($(ls "$raw_dir"))  # process all ids in the directory
fi

# Create output folder
outfolder="$main_dir/processed/${modality}_nii"
if [[ -d "$outfolder" ]]; then
  echo "copying nifti files to $outfolder"
else
  mkdir -p "$outfolder"
fi

cd "$raw_dir" || exit

for id in "${ids[@]}"; do 
  # echo "Processing $id"
  if [ "$modality" == "T1" ]; then 
    find "$raw_dir/$id" -type f -name '*T1.nii*' >> copy_path.csv
  elif [ "$modality" == "FLAIR" ]; then
    find "$raw_dir/$id" -type f -name  '*FLAIR.nii*' >> copy_path.csv
  elif [ "$modality" == "T2star" ]; then
    find "$raw_dir/$id" -type f -name '*T2star.nii*' >> copy_path.csv
  elif [ "$modality" == "Amyloid" ]; then
    find "$raw_dir/$id" -type f -name  '*Amyloid.nii*' >> copy_path.csv
  elif [ "$modality" == "Tau" ]; then
    find "$raw_dir/$id" -type f -name '*Tau.nii*' >> copy_path.csv
  fi
done

if [ "$cohort" == 'OASIS' ]; then
  cat copy_path.csv | while read -r line; do
    id=$(echo "$line" | awk -F '/' '{print $7}')
    new_id=$(echo "$id" | sed -e 's/sub-//' -e 's/ses-//' -e 's/sess-//')
    new_name="${new_id}_${modality}.nii.gz"
    cp "$line" "$outfolder"/"$new_name"
  done
fi
rm copy_path.csv