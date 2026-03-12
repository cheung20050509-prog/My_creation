import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

TEXT_DIM = 0
ACOUSTIC_DIM = 0
VISUAL_DIM = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_dataset_config(dataset_name):
    global TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM

    dataset_configs = {
        "mosi":  {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 47, "TEXT_DIM": 768},
        "mosei": {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 35, "TEXT_DIM": 768},
    }

    config = dataset_configs.get(dataset_name)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ACOUSTIC_DIM = config["ACOUSTIC_DIM"]
    VISUAL_DIM = config["VISUAL_DIM"]
    TEXT_DIM = config["TEXT_DIM"]
