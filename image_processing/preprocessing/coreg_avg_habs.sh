#!/bin/bash

input_dir="/SeaExp_1/MRI_PET/HABS/processed/Amyloid_nii"
output_dir="/SeaExp_1/MRI_PET/HABS/processed/coreg_avg"

# input_dir="/SeaExp_1/MRI_PET/HABS/processed/Amyloid_nii/test"
# output_dir="/SeaExp_1/MRI_PET/HABS/processed/Amyloid_nii/test/output"

mkdir -p "$output_dir"

for nifti_file in "$input_dir"/*.nii; do
    base_name=$(basename "$nifti_file" .nii)
    echo "base_name:""$base_name"
    
    # split into frames
    fslsplit "$nifti_file" "${output_dir}/${base_name}_frame" -t
    
    # get first frame to which all other images will be coregistered
    base_frame="${output_dir}/${base_name}_frame0000.nii.gz"
    ls "$base_frame"

    # use flirt to coregister
    for frame in "${output_dir}/${base_name}_frame"*.nii.gz; do
        coreg_frame="${output_dir}/coreg_$(basename "$frame")"
        if [ "$frame" != "$base_frame" ]; then
            flirt -in "$frame" -ref "$base_frame" -out "$coreg_frame" -omat "${coreg_frame%.nii.gz}.mat"
        else
            cp "$frame" "$coreg_frame"
        fi
    done
    
    # combine coregistered frames into one 4D volume
    fslmerge -t "${output_dir}/coreg_${base_name}_dynamic_image.nii.gz" "${output_dir}/coreg_${base_name}_frame"*.nii.gz

    # average into a single 3D image
    fslmaths "${output_dir}/coreg_${base_name}_dynamic_image.nii.gz" -Tmean "${output_dir}/${base_name}_coregavg.nii.gz"
    
    # remove intermediate files
    rm "${output_dir}/${base_name}_frame"*.nii.gz
    rm "${output_dir}/coreg_${base_name}_frame"*.nii.gz
    rm "${output_dir}/coreg_${base_name}_frame"*.mat

    # check final image
    fslinfo "${output_dir}/${base_name}_coregavg.nii.gz"
done