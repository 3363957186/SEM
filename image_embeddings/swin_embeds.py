"""
This script extracts embeddings from volumetric MRIs using a pretrained SwinUNETR model.

Input:
- T1 images: intensity-normalized FastSurfer processed images (orig_nu.mgz)
- Non-T1 images: resampled SynthSeg processed images (*_resampled.nii)

Data organization:
- T1: {base_path}/{cohort}/processed/fastsurfer/{subject}/
- Non-T1: {base_path}/{cohort}/processed/synthseg/{subject}/

Output: whole-brain embeddings saved as .npy files
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
import os
import nibabel as nib
import logging
import argparse


logger = logging.getLogger(__name__)
log_dir = '/projectnb/vkolagrp/varuna/embed/logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename='logs/whole_embed_1.log', encoding='utf-8', level=logging.DEBUG)


def get_rid_folder(cohort: str, modality: str = None, base_path: str = "/projectnb/vkolagrp/varuna/test_embeds"):
    """
    Fetches a list of RIDs based on the directories present in the base path for a given cohort.

    Args:
        cohort (str): The name of the cohort.
        modality (str, optional): The MRI modality (e.g., FLAIR, T1, T2star etc.).
        base_path (str): The base directory where each cohort's data is stored.

    Returns:
        list: List of RIDs based on directory names, or an empty list if the directory does not exist or other errors occur.
    """
    if modality == "T1":
        directory_path = os.path.join(base_path, cohort, modality, "processed", "fastsurfer")
    else:
        directory_path = os.path.join(base_path, cohort, modality, "processed", "synthseg")
        
    try:
        rids = [rid for rid in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, rid))]
        print(rids)
        return rids
    except FileNotFoundError:
        print(f"Directory not found: {directory_path}. Please check the path.")
        return []
    except Exception as e:
        print(f"An error occurred: {str(e)}.")
        return []


def pad_image_to_256(img):
    target_size = (256, 256, 256)
    padding = [0, 0] 
    for i, size in enumerate(img.shape[2:], start=2): 
        diff = target_size[i-2] - size
        pad_one_side = diff // 2
        pad_other_side = diff - pad_one_side
        padding = [pad_one_side, pad_other_side] + padding
    padded_img = F.pad(img, padding, "constant", 0) 
    return padded_img


class WholeBrainDataset(Dataset):
    def __init__(self, rid: str, cohort: str, modality: str, base_path: str = "/projectnb/vkolagrp/varuna/test_embeds", transforms=None):
        self.rid = rid
        self.cohort = cohort
        self.modality = modality
        self.base_path = base_path
        
        if modality == "T1":
            # for T1, use fastsurfer processed images
            self.base_folder = os.path.join(base_path, cohort, modality, "processed", "fastsurfer")
            self.brain = nib.load(os.path.join(self.base_folder, rid, "mri", "orig_nu.mgz")).get_fdata()
        else:
            # for non-T1, use synthseg resampled images
            self.base_folder = os.path.join(base_path, cohort, modality, "processed", "synthseg")
            image_path = os.path.join(self.base_folder, rid, f"{rid}_resampled.nii")
            self.brain = nib.load(image_path).get_fdata()
        
    def __len__(self):
        return 1 
        
    def __getitem__(self, idx):
        img = torch.tensor(self.brain).unsqueeze(0).float()
        if self.modality != "T1":
            img = pad_image_to_256(img)
        return img


class Swin(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        image_size = 256
        self.swinunetr = SwinUNETR(
            in_channels=1,
            out_channels=1,
            img_size=(image_size)*3,
            feature_size=48,
            use_checkpoint=True,
        )
        pretrained_pth = "model_swinvit.pt"
        model_dict = torch.load(pretrained_pth, map_location="cpu")
        model_dict["state_dict"] = {k.replace("swinViT.", "module."): v for k, v in model_dict["state_dict"].items()}
        self.swinunetr.load_from(model_dict)
        self.swinunetr.train(False)
        
    def forward(self, x_in):
        hidden_states_out = self.swinunetr.swinViT(x_in, self.swinunetr.normalize)
        dec4 = self.swinunetr.encoder10(hidden_states_out[4])
        return dec4


if __name__ == "__main__":
    """
    example script command: python swin_embeds.py "ADNI" "T1"
    """
    parser = argparse.ArgumentParser(description="Process whole brain embeddings")
    parser.add_argument("cohort", help="string for name of cohort to process")
    parser.add_argument("modality", help="string indicating MRI modality")
    parser.add_argument("--base_path", default="/projectnb/vkolagrp/varuna/test_embeds", help="Base path for data")
    
    args = parser.parse_args()
    cohort = args.cohort
    modality = args.modality
    base_path = args.base_path
    
    model = Swin()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    logging.debug(device)
    model.to(device)
    os.makedirs(f"/projectnb/vkolagrp/varuna/embeddings/test/{cohort}/{modality}", exist_ok=True)
    rids = get_rid_folder(cohort, modality, base_path)
    model.eval()
    
    with torch.no_grad():
        for rid in rids:
            print("Starting RID", rid)
            dir_path = f"/projectnb/vkolagrp/varuna/embeddings/test/{cohort}/{modality}/{rid}/"
            if os.path.exists(dir_path):
                logging.info(f"Directory {dir_path} already exists. Skipping {rid}.")
                continue
                
            logging.info(f"Processing {rid}")
            os.makedirs(dir_path, exist_ok=True)
            
            try:
                dataset = WholeBrainDataset(rid, cohort, modality, base_path)
                dl = DataLoader(dataset, batch_size=1)
                
                for img in dl:
                    logging.info(f"Processing whole brain for RID {rid}")
                    img = img.to(device)
                    
                    try:
                        dat = model(img)
                        dat = dat.to("cpu").numpy()
                        print("Embedding shape:", dat.shape)
                        np.save(f"/projectnb/vkolagrp/varuna/embeddings/test/{cohort}/{modality}/{rid}/{modality}_embedding_{rid}.npy", dat)
                        logging.info(f"Successfully saved embedding for RID {rid}")
                    except Exception as e:
                        logging.error(f"Error processing embedding for RID {rid}: {e}")
                        print(e)
                    break
                    
            except Exception as e:
                logging.error(f"Error loading data for RID {rid}: {e}")
                print(e)
