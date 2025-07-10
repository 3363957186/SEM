import os
import nibabel as nib
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime


def get_cmd_line():
    parser = ArgumentParser()
    parser.add_argument("--cohort", type=str, help='Cohort processing')
    parser.add_argument("--pet_path", type=str, help='Path to preprocessed amyloid PET')
    parser.add_argument("--gaain_ctx_path", type=str, help='Path to GAAIN cortical mask')
    parser.add_argument("--gaain_cereb_path", type=str, help='Path to GAAIN cerebellar grey mask')
    parser.add_argument("--outdir", type=str, help='Path to output directory')

    args = parser.parse_args()
    option_dir = {'cohort': args.cohort, 'pet_path': args.pet_path, 'gaain_ctx_path': args.gaain_ctx_path,
                  'gaain_cereb_path': args.gaain_cereb_path, 'outdir': args.outdir}
    return option_dir


def concat_suvrs(cohort, results_dict, pet_file, ctx_suv, cereb_suv, suvrs):
    
    file_name = os.path.basename(pet_file)
    if cohort == 'GAAIN':
        rid = file_name.split('_', 1)[0]
        rid = rid[1:] if rid.startswith('w') else rid
        print("Saving SUV and SUVRs for subject {}".format(rid))
        results_dict[rid] = {
            'Cortical SUV': ctx_suv,
            'Cerebellar SUV': cereb_suv,
            'SUVr': suvrs}
    elif cohort == 'FHS':
        rid = file_name.split('_', 1)[0]
        rid = rid[1:] if rid.startswith('w') else rid
        print("Saving SUV and SUVRs for subject {}".format(rid))
        results_dict[rid] = {
            'Cortical DVR': ctx_suv,
            'Cerebellar DVR': cereb_suv,
            'DVR Normalized': suvrs}
    elif cohort == 'HABS':
        parts = file_name.split('_')
        rid = file_name.split('_', 1)[0]
        rid = rid[1:] if rid.startswith('w') else rid
        date_part = '_'.join(parts[1:2])
        image_id = f'{rid}_{date_part}'
        print("Saving SUV and SUVRs for subject {}".format(rid))
        results_dict[rid] = {
            'PIB_SessionDate': date_part,
            'Cortical SUV': ctx_suv,
            'Cerebellar SUV': cereb_suv,
            'SUVr Normalized': suvrs}
    else:
        parts = file_name.split('_')
        if len(parts) >= 6:
            rid_part = '_'.join(parts[:1])
            loniuid_part = '_'.join(parts[5:6])
            rid_part = rid_part[1:] if rid_part.startswith('w') else rid_part
            image_id = f'{rid_part}_{loniuid_part}'
            print("Saving SUV and SUVRs for subject {}".format(image_id))
            results_dict[image_id] = {
                'RID': rid_part,
                'LONIUID.AMY': loniuid_part,
                'Cortical SUV': ctx_suv,
                'Cerebellar SUV': cereb_suv,
                'SUVr': suvrs}
    return results_dict


def calculate_suvrs(cohort, amyloid_pet_path, cortical_mask_path, cerebellar_mask_path, out_folder):
    cortical_mask_img = nib.load(cortical_mask_path)
    cerebellar_mask_img = nib.load(cerebellar_mask_path)
    cortical_mask_data = cortical_mask_img.get_fdata()
    cerebellar_mask_data = cerebellar_mask_img.get_fdata()

    results_dict = {}

    pet_files = [os.path.join(amyloid_pet_path, file) for file in os.listdir(amyloid_pet_path)]
    for pet_file in pet_files:
        amyloid_img = nib.load(pet_file)
        amyloid_data = amyloid_img.get_fdata()

        # extract SUV values using gaain masks
        cortical_suv = np.nanmean(amyloid_data[cortical_mask_data > 0])
        cerebellar_suv = np.nanmean(amyloid_data[cerebellar_mask_data > 0])

        # calculate SUVrs
        suvr = cortical_suv / cerebellar_suv
        results_dict = concat_suvrs(cohort, results_dict, pet_file, cortical_suv, cerebellar_suv, suvr)

    df = pd.DataFrame.from_dict(results_dict, orient='index')

    df.reset_index(inplace=True)
    if cohort == 'GAAIN' or cohort == 'FHS' or cohort == 'HABS':
        df.rename(columns={'index': 'RID'}, inplace=True)
    else:
        df.rename(columns={'index': 'RID_LONIUID'}, inplace=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suvr_csv_file = os.path.join(out_folder, f'suvr_results_{timestamp}.csv')
    df.to_csv(suvr_csv_file, index=False)


if __name__ == '__main__':
    options = get_cmd_line()
    calculate_suvrs(cohort=options['cohort'], 
                    amyloid_pet_path=options['pet_path'],
                    cortical_mask_path=options['gaain_ctx_path'],
                    cerebellar_mask_path=options['gaain_cereb_path'],
                    out_folder=options['outdir'])
