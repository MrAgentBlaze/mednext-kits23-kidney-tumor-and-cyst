import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.kits23_dataset import KiTS23Dataset
from load_split import load_case_list
from losses import DiceCELoss

from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import (
    create_mednext_v1,
)


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = (
    r"D:\Brian Lala\Research"
    r"\2-step-segmentation-kidney-tumor-and-cyst"
    r"\data\KiTS23"
)

PATCH_SIZE = (32, 128, 128)
BATCH_SIZE = 2
NUM_BATCHES = 5
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Load train split
# ============================================================

train_cases = load_case_list(
    "configs/train_cases.txt"
)

print(f"Training cases: {len(train_cases)}")


# ============================================================
# Dataset
# ============================================================

train_dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=PATCH_SIZE,
    cases=train_cases,
    sampling_mode="mixed",
)


# ============================================================
# DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


# ============================================================
# MedNeXt-S
# ============================================================

model = create_mednext_v1(
    num_input_channels=1,
    num_classes=4,
    model_id="S",
    kernel_size=3,
    deep_supervision=False,
)

model = model.to(DEVICE)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = DiceCELoss(
    dice_weight=0.5,
    ce_weight=0.5,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Training smoke test
# ============================================================

model.train()

print()
print("Starting 5-batch training smoke test...")
print()

initial_loss = None

for batch_idx, batch in enumerate(train_loader):

    if batch_idx >= NUM_BATCHES:
        break

    images = batch["image"].to(
        DEVICE,
        non_blocking=True,
    )

    labels = batch["label"].to(
        DEVICE,
        non_blocking=True,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    optimizer.zero_grad(set_to_none=True)

    outputs = model(images)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = criterion(
        outputs,
        labels,
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Optimizer update
    # --------------------------------------------------------

    optimizer.step()

    if initial_loss is None:
        initial_loss = loss.item()

    print(
        f"Batch {batch_idx + 1}/{NUM_BATCHES} | "
        f"Loss: {loss.item():.6f} | "
        f"Input: {tuple(images.shape)} | "
        f"Output: {tuple(outputs.shape)}"
    )


# ============================================================
# Final verification
# ============================================================

print()
print("Smoke test completed successfully.")
print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss:   {loss.item():.6f}")

if not torch.isfinite(loss):
    raise RuntimeError(
        "Loss became NaN or infinite."
    )

print("Loss is finite.")
print("End-to-end training pipeline PASSED.")