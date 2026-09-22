from data.kits23_dataset import KiTS23Dataset

import numpy as np
import nibabel as nib
from pathlib import Path


DATA_ROOT = (
    r"D:\Brian Lala\Research"
    r"\2-step-segmentation-kidney-tumor-and-cyst"
    r"\data\KiTS23"
)


# ============================================================
# Check KiTS23 volume dimensions
# ============================================================

data_root = Path(DATA_ROOT)

depths = []

for case_dir in sorted(data_root.glob("case_*")):
    image_path = case_dir / "imaging.nii.gz"
    image = nib.load(image_path)

    depth, height, width = image.shape
    depths.append((case_dir.name, depth, height, width))


print(f"Total cases: {len(depths)}")

min_case = min(depths, key=lambda x: x[1])
max_case = max(depths, key=lambda x: x[1])

print(
    f"Minimum depth: {min_case[1]} "
    f"({min_case[0]}, shape={min_case[1:]})"
)

print(
    f"Maximum depth: {max_case[1]} "
    f"({max_case[0]}, shape={max_case[1:]})"
)

shallow_cases = [x for x in depths if x[1] < 32]

print(f"Cases with depth < 32: {len(shallow_cases)}")

for case in shallow_cases:
    print(case)


# ============================================================
# Test target sampling on first 20 cases
# ============================================================

sample_targets = ["tumor", "cyst"]


def sampling_test(sample_target):

    dataset = KiTS23Dataset(
        data_root=DATA_ROOT,
        patch_size=(32, 128, 128),
        sampling_mode=sample_target,
    )

    for i in range(20):
        sample = dataset[i]
        labels = sample["label"].unique().tolist()
        print(
            f"{sample['case']}: labels = {labels}"
        )


for target in sample_targets:
    print(f"\nSampling target: {target}")
    sampling_test(target)

# ============================================================
# Test 1,000 random target patches
# ============================================================

for target in ["tumor", "cyst"]:

    print(f"\nSampling target: {target}")

    dataset = KiTS23Dataset(
        data_root=DATA_ROOT,
        patch_size=(32, 128, 128),
        sampling_mode=target,
    )

    tumor_count = 0
    cyst_count = 0

    for i in range(1000):

        sample = dataset[
            np.random.randint(len(dataset))
        ]

        labels = sample["label"].unique().tolist()

        if 2 in labels:
            tumor_count += 1

        if 3 in labels:
            cyst_count += 1

    print(
        f"Patches containing tumor: "
        f"{tumor_count}/1000"
    )

    print(
        f"Patches containing cyst:  "
        f"{cyst_count}/1000"
    )