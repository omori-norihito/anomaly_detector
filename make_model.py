#!/usr/bin/env python3
import os
import sys
import tarfile
from pathlib import Path

sys.path.append('./ind_knn_ad/indad')
import wget

import torch
import random
import numpy as np

from ind_knn_ad.indad.models import PatchCore
from ind_knn_ad.indad.data import MVTecDataset

method = "patchcore"
cls = "metal_nut"
DATASETS_PATH = Path("./ind_knn_ad/datasets")

# seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def download():
    if not (DATASETS_PATH / cls).exists():
        print(f"   Could not find '{cls}' in '{DATASETS_PATH}/'. Downloading ... ")
        url = f"https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/{cls}.tar.xz"
        wget.download(url)
        with tarfile.open(f"{cls}.tar.xz") as tar:
            tar.extractall(DATASETS_PATH)
        os.remove(f"{cls}.tar.xz")
        print("")  # force newline


model = PatchCore(
                  f_coreset=.10,
                  backbone_name="wide_resnet50_2",
                  )

results = {}
print(f"\n█│ Running {method} on {cls} dataset.")
print(f" ╰{'─'*(len(method)+len(cls)+23)}\n")
download()
os.chdir("./ind_knn_ad")
train_ds, test_ds = MVTecDataset(cls).get_dataloaders()

print("   Training ...")
model.fit(train_ds)
print("   Testing ...")
image_rocauc, pixel_rocauc = model.evaluate(test_ds)

print(f"\n   ╭{'─'*(len(cls)+15)}┬{'─'*20}┬{'─'*20}╮")
print(f"   │ Test results {cls} │ image_rocauc: {image_rocauc:.2f} │ pixel_rocauc: {pixel_rocauc:.2f} │")
print(f"   ╰{'─'*(len(cls)+15)}┴{'─'*20}┴{'─'*20}╯")
results[cls] = [float(image_rocauc), float(pixel_rocauc)]

image_results = [v[0] for _, v in results.items()]
average_image_roc_auc = sum(image_results)/len(image_results)
image_results = [v[1] for _, v in results.items()]
average_pixel_roc_auc = sum(image_results)/len(image_results)

total_results = {
    "per_class_results": results,
    "average image rocauc": average_image_roc_auc,
    "average pixel rocauc": average_pixel_roc_auc,
    "model parameters": model.get_parameters(),
}

os.chdir("..")
model_file = f'{method}_{cls}.pth'
torch.save(model, model_file)
print("Model file creation is complete.")
