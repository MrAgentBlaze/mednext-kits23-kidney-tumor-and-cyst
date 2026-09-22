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
    patch_size=(32, 128, 128),
    cases=train_cases,
    sampling_mode="tumor",
)

val_dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=(32, 128, 128),
    cases=val_cases,
    sampling_mode="tumor",
)


# ============================================================
# Basic verification
# ============================================================

print(f"Training dataset size:   {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


# ============================================================
# Verify training samples
# ============================================================

print("\nTesting training dataset...")

for i in range(10):

    sample = train_dataset[i]

    case = sample["case"]

    if case not in train_cases:
        raise RuntimeError(
            f"Training dataset returned invalid case: {case}"
        )

    labels = sample["label"].unique().tolist()

    print(
        f"{case}: "
        f"labels={labels}, "
        f"shape={tuple(sample['image'].shape)}"
    )


# ============================================================
# Verify validation samples
# ============================================================

print("\nTesting validation dataset...")

for i in range(10):

    sample = val_dataset[i]

    case = sample["case"]

    if case not in val_cases:
        raise RuntimeError(
            f"Validation dataset returned invalid case: {case}"
        )

    labels = sample["label"].unique().tolist()

    print(
        f"{case}: "
        f"labels={labels}, "
        f"shape={tuple(sample['image'].shape)}"
    )


# ============================================================
# Final verification
# ============================================================

if set(train_cases) & set(val_cases):

    raise RuntimeError(
        "Train/validation overlap detected!"
    )


print("\nDataset split verification PASSED.")