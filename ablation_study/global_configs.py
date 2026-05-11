import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

TEXT_DIM = 0
ACOUSTIC_DIM = 0
VISUAL_DIM = 0
# HCF_DIM is 0 for regression datasets (MOSI/MOSEI/SIMSv2) and 4 for the
# HKT-aligned MHD/MSD tasks (UR-FUNNY, MUStARD). Keep it exposed at module
# level so downstream code (`data_humor.build_humor_loaders`, ALBERT wrapper,
# etc.) can read a single source of truth via `global_configs.HCF_DIM`.
HCF_DIM = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_dataset_config(dataset_name):
    global TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM, HCF_DIM

    dataset_configs = {
        "mosi":  {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 47,  "TEXT_DIM": 768, "HCF_DIM": 0},
        "mosei": {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 35,  "TEXT_DIM": 768, "HCF_DIM": 0},
        "simsv2": {"ACOUSTIC_DIM": 25, "VISUAL_DIM": 177, "TEXT_DIM": 768, "HCF_DIM": 0},
        # MHD (UR-FUNNY) and MSD (MUStARD): HKT feature slicing
        # https://github.com/matalvepu/HKT/blob/main/global_config.py
        # acoustic_features_list = range(0, 60)    -> ACOUSTIC_DIM = 60
        # visual_features_list   = range(55, 91)   -> VISUAL_DIM   = 36
        # HCF = 4
        "ur_funny": {"ACOUSTIC_DIM": 60, "VISUAL_DIM": 36, "TEXT_DIM": 768, "HCF_DIM": 4},
        "mustard":  {"ACOUSTIC_DIM": 60, "VISUAL_DIM": 36, "TEXT_DIM": 768, "HCF_DIM": 4},
    }

    config = dataset_configs.get(dataset_name)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ACOUSTIC_DIM = config["ACOUSTIC_DIM"]
    VISUAL_DIM = config["VISUAL_DIM"]
    TEXT_DIM = config["TEXT_DIM"]
    HCF_DIM = config["HCF_DIM"]
