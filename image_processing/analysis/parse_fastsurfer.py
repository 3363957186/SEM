
import pandas as pd
import os
import argparse

# to process a single stats file
def process_stats_file(file_path, cohort):
    file_parts = file_path.split('/')
    filename = file_parts[6]
    parts = filename.split('_')
    subject_id = parts[0]
    if cohort == 'ADNI' or cohort == 'A4' or cohort == 'AIBL':
        image_id = parts[5]
    elif cohort == 'OASIS3':
        image_id = parts[2]
    elif cohort == 'FHS' or cohort == 'HABS' or cohort == 'NACC':
        image_id = parts[1]
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_processing = False
        for line in lines:
            if line.strip().startswith('# ColHeaders'):
                start_processing = True
                continue
            if start_processing and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                data.append({
                'SubjectID': subject_id,
                'ImageID': image_id,
                'Index': parts[0],
                'SegId': parts[1],
                'NVoxels': parts[2],
                'Volume_mm3': parts[3],
                'StructName': parts[4],
                'normMean': parts[5],
                'normStdDev': parts[6],
                'normMin': parts[7],
                'normMax': parts[8],
                'normRange': parts[9]
            })
                
    return pd.DataFrame(data)


def main(cohort, input_csv, output_csv):
    # read the CSV containing file paths
    filepaths_df = pd.read_csv(input_csv, header=None, names=['FilePath'])
    all_data = []

    for file_path in filepaths_df['FilePath']:
        # process each stats file
        df = process_stats_file(file_path, cohort)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    # pivot the combined df to get one row per subject
    pivot_df = combined_df.pivot_table(index=['SubjectID', 'ImageID'], 
                            columns='StructName', 
                            values='Volume_mm3', 
                            aggfunc='first').reset_index()

    pivot_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    """
    example bash comand: find $PWD -type f -name "aseg+DKT.stats" >> oasis3_stats_filepaths.csv # find stats files paths using bash in the processing directory
    example script command: python parse_fastsurfer.py "NACC" "/SeaExp_1-ayan/MRI_PET/NACC/processed/fastsurfer/nacc_stats_filepaths.csv" "/home/varunaja/mri_pet/data/NACC/nacc_fastsurfer_vols.csv"
    """
    parser = argparse.ArgumentParser(description="Process stats files and generate a pivoted volume CSV.")
    parser.add_argument("cohort", help="Cohort name")
    parser.add_argument("input_csv", help="Path to the input CSV file containing stats file paths.")
    parser.add_argument("output_csv", help="Path to the output CSV file for the pivoted volume data.")
    
    args = parser.parse_args()
    
    main(args.cohort, args.input_csv, args.output_csv)
