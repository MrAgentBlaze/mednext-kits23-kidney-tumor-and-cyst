import torch
from torch.utils.data import DataLoader

from data.kits23_dataset import KiTS23Dataset
from load_split import load_case_list


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


# ============================================================
# Load case splits
# ============================================================

train_cases = load_case_list(
    "configs/train_cases.txt"
)

val_cases = load_case_list(
    "configs/val_cases.txt"
)


# ============================================================
# Create datasets
# ============================================================

train_dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=PATCH_SIZE,
    cases=train_cases,
    sampling_mode="tumor",
)

val_dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=PATCH_SIZE,
    cases=val_cases,
    sampling_mode="tumor",
)


# ============================================================
# Create DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# Test training DataLoader
# ============================================================

print("Testing training DataLoader...")

train_batch = next(iter(train_loader))

print(
    f"Image shape: {tuple(train_batch['image'].shape)}"
)

print(
    f"Label shape: {tuple(train_batch['label'].shape)}"
)

print(
    f"Cases: {train_batch['case']}"
)

print(
    f"Image dtype: {train_batch['image'].dtype}"
)

print(
    f"Label dtype: {train_batch['label'].dtype}"
)

print(
    f"Image range: "
    f"{train_batch['image'].min().item():.4f} - "
    f"{train_batch['image'].max().item():.4f}"
)

print(
    f"Labels present: "
    f"{torch.unique(train_batch['label']).tolist()}"
)


# ============================================================
# Test validation DataLoader
# ============================================================

print("\nTesting validation DataLoader...")

val_batch = next(iter(val_loader))

print(
    f"Image shape: {tuple(val_batch['image'].shape)}"
)

print(
    f"Label shape: {tuple(val_batch['label'].shape)}"
)

print(
    f"Cases: {val_batch['case']}"
)

print(
    f"Image dtype: {val_batch['image'].dtype}"
)

print(
    f"Label dtype: {val_batch['label'].dtype}"
)

print(
    f"Labels present: "
    f"{torch.unique(val_batch['label']).tolist()}"
)


# ============================================================
# Shape assertions
# ============================================================

assert train_batch["image"].shape == (
    BATCH_SIZE,
    1,
    32,
    128,
    128,
)

assert train_batch["label"].shape == (
    BATCH_SIZE,
    32,
    128,
    128,
)

assert val_batch["image"].shape == (
    BATCH_SIZE,
    1,
    32,
    128,
    128,
)

assert val_batch["label"].shape == (
    BATCH_SIZE,
    32,
    128,
    128,
)


# ============================================================
# Final result
# ============================================================

print("\nDataLoader verification PASSED.")