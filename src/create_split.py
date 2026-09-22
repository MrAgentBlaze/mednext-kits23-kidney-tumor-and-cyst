from pathlib import Path
import random

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path(
    r"D:\Brian Lala\Research"
    r"\2-step-segmentation-kidney-tumor-and-cyst"
    r"\data\KiTS23"
)

SEED = 42
TRAIN_RATIO = 0.80


# ============================================================
# Find cases
# ============================================================

cases = sorted(
    [
        p.name
        for p in DATA_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("case_")
    ]
)

print(f"Total cases found: {len(cases)}")


# ============================================================
# Reproducible shuffle
# ============================================================

random.seed(SEED)

random.shuffle(cases)


# ============================================================
# Train / validation split
# ============================================================

train_count = int(len(cases) * TRAIN_RATIO)

train_cases = sorted(cases[:train_count])
val_cases = sorted(cases[train_count:])


# ============================================================
# Verify
# ============================================================

print(f"Training cases:   {len(train_cases)}")
print(f"Validation cases: {len(val_cases)}")

overlap = set(train_cases) & set(val_cases)

print(f"Case overlap:     {len(overlap)}")

if overlap:
    raise RuntimeError(
        f"Train/validation overlap detected: {overlap}"
    )


# ============================================================
# Save split
# ============================================================

SPLIT_DIR = Path("configs")
SPLIT_DIR.mkdir(exist_ok=True)

train_file = SPLIT_DIR / "train_cases.txt"
val_file = SPLIT_DIR / "val_cases.txt"

train_file.write_text(
    "\n".join(train_cases) + "\n",
    encoding="utf-8",
)

val_file.write_text(
    "\n".join(val_cases) + "\n",
    encoding="utf-8",
)


# ============================================================
# Final output
# ============================================================

print()
print("Split created successfully.")
print(f"Train file: {train_file}")
print(f"Val file:   {val_file}")
print()
print("First 10 training cases:")
for case in train_cases[:10]:
    print(f"  {case}")

print()
print("First 10 validation cases:")
for case in val_cases[:10]:
    print(f"  {case}")