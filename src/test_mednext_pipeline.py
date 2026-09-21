import torch

from data.kits23_dataset import KiTS23Dataset
from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import (
    create_mednext_v1,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_ROOT = (
    r"D:\(Path)\Research"
    r"\2-step-segmentation-kidney-tumor-and-cyst"
    r"\data\KiTS23"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=(32, 128, 128),
)

sample = dataset[0]

image = sample["image"].unsqueeze(0).to(DEVICE)
label = sample["label"].unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------
# MedNeXt-S
# ---------------------------------------------------------

model = create_mednext_v1(
    num_input_channels=1,
    num_classes=4,
    model_id="S",
    kernel_size=3,
    deep_supervision=False,
)

model = model.to(DEVICE)


# ---------------------------------------------------------
# Forward pass
# ---------------------------------------------------------

print("Device:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0))

print("Input:", image.shape)
print("Target:", label.shape)

with torch.no_grad():
    output = model(image)

print("Output:", output.shape)
print("Output device:", output.device)

print("Target labels:", label.unique().tolist())
print("Output channels:", output.shape[1])