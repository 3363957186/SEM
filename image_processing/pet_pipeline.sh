while getopts ":d:c:i:m:" opt; do
  case $opt in
    d)
      main_dir=$OPTARG  # main processed directory directory
      ;;
    c)
      cohort=$OPTARG  # cohort
      ;;
    i)
      id_hash=$OPTARG # id csv file
      ;;
    cp)
      copyfiles=$OPTARG # whether to run copyfiles
      ;;
    reo)
      reorient=$OPTARG # whether to run reorient
      ;;
    rea)
      realign=$OPTARG # whether to run realign
      ;;
    cor)
      coreg=$OPTARG
  esac
done

cwd = "$PWD"
if [[ $copyfiles == '1']]; then
    if [[ "$cohort" == "A4"]]; then
        ./copyfiles/a4_copyfiles.sh -d $main_dir -m "T1"
        ./copyfiles/a4_copyfiles.sh -d $main_dir -m "Amyloid"
    else
        ./copyfiles/copyfiles.sh -c $cohort -d $main_dir -m "T1"
        ./copyfiles/copyfiles.sh -c $cohort -d $main_dir -m "Amyloid"
fi

if [[ $reorient == '1']]; then
    # reorient2std amyloid
    t1_copy_folder="$main_dir/processed/T1_nii"
    amy_copy_folder="$main_dir/processed/Amyloid_nii"

    reoriented_t1_folder="$main_dir/processed/reoriented/T1_nii"
    reoriented_amy_folder="$main_dir/processed/reoriented/Amyloid_nii"

    ./reorient2std.sh -i $t1_copy_folder -o $reoriented_t1_folder
    ./reorient2std.sh -i $amy_copy_folder -o $reoriented_amy_folder

    gunzip $reoriented_t1_folder/*
    gunzip $reoriented_amy_folder/*
fi

if [[ $realign == '1']]; then
    cd preprocessing
    matlab -r "realign_all_niftis($cohort, $main_dir, 'Amyloid'); exit;"
    matlab -r "realign_all_niftis($cohort, $main_dir, 'T1'); exit;"
    cd ../
fi

if [[ $coreg == '1']]; then
    cd pet
    matlab -r "coreg_klunk($cohort, $main_dir, $id_hash); exit;" >> {$cohort}_log.txt
    cd ../
fi

# need to fix this
cd $main_dir/coreg_klunk/{$cohort}_T1_Amyloid
mkdir -p normalized_wskull/Amyloid
mkdir -p normalized_wskull/T1
mkdir other
mv wm* normalized_wskull/T1
mv w* normalized_wskull/Amyloid
mv * other
mv other/normalized_wskull ../
mv other/logs ../

cd $cwd

python analysis/extract_suvrs_centiloids.py --cohort $cohort --pet_path $main_dir/coreg_klunk/{$cohort}_T1_Amyloid/normalized_wskull/Amyloid' --gaain_ctx_path '/SeaExp_1/MRI_PET/GAAIN/Centiloid_Std_VOI/nifti/2mm/voi_ctx_2mm.nii' --gaain_cereb_path '/SeaExp_1/MRI_PET/GAAIN/Centiloid_Std_VOI/nifti/2mm/voi_WhlCbl_2mm.nii' --outdir '$main_dir'
