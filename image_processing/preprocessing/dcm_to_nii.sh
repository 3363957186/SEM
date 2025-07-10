#!/bin/bash

# Assign options
while getopts ":d:i:m:c:" opt; do
  case $opt in
    d)
      inpath=$OPTARG  # Raw directory
      ;;
    i)
      sessions=$OPTARG # specific adni ids
      ;;
    m)
      modality=$OPTARG #modality
      ;;
    c)
      cohort=$OPTARG # cohort
      ;;
  esac
done

ml dcm2niix
if [ -n "$sessions" ]; then
  ids=("$sessions")  # process specified ID
else
  ids=($(ls "$inpath"))  # process all IDs in the directory
fi

cd "$inpath"
for id in "${ids[@]}"; do
  echo "Processing subject $id"
  if [ "$cohort" == "AIBL" ]; then
    rid="$id"
  else
    rid=`echo "$id" | awk -F '_' '{print $3}'`
  fi
  if [ "$modality" == "T1" ]; then 
    modality_folder=$(find "$id" -mindepth 1 -maxdepth 1 -type d \( -name '*Accelerated*' -o -name '*Sag*' -o -name '*MPRAGE*' -o -name '*SPGR*' \))
  elif [ "$modality" == "FLAIR" ]; then 
    modality_folder=`find "$id" -mindepth 1 -maxdepth 1 -type d -name '*FLAIR*'`
  elif [ "$modality" == "Amyloid" ]; then
    modality_folder=$(find "$id" -mindepth 1 -maxdepth 1 -type d \( -iname '*AV45*' -o -iname '*FBB*' -o -iname '*PIB*' -o -iname '*NAV*' -o -iname '*Brain*' -o -iname '*FLU*' -o -iname '*SUM*' -o -iname '*img*' -o -iname '*RSRCH*' \))
    # modality_folder=`find "$id" -mindepth 1 -maxdepth 1 -type d \( -name 'AV45*' -o -name 'FBB*' \)`
  elif [ "$modality" == "Tau" ]; then
    modality_folder=`find "$id" -mindepth 1 -maxdepth 1 -type d -name 'AV1451*'`
  fi
  #run dcm2niix
  for folder in $modality_folder; do
    dcm_files=$(find $folder -type f -name '*.dcm')
    paths=`echo "$dcm_files" | awk 'BEGIN{FS=OFS="/"}{NF--; print}' | sort | uniq`
    mkdir -p "$inpath"/"$folder"/nifti
    mkdir -p "$inpath"/"$folder"/dicom
    for path in $paths; do
      date_folder=`echo "$path" | awk -F '/' '{print $3}'`
      image_folder=`echo "$path" | awk -F '/' '{print $4}'`
      out_name="${rid}_${date_folder}_${image_folder}_${modality}"
      echo "$out_name"
      dcm2niix -f "$out_name" -o $inpath/$folder/nifti $inpath/$folder/$date_folder/$image_folder
      if [ $? -eq 0 ]; then
        echo "dcm2niix succeeded, moving directory."
        mv "$inpath"/"$folder"/"$date_folder" "$inpath"/"$folder"/dicom
      else
        echo "dcm2niix failed, not moving directory."
      fi
    done
  done
done
