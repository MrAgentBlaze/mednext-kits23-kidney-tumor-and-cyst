from pathlib import Path

import nibabel as nib
import numpy as np


DATA_ROOT = Path(
    r"D:\Brian Lala\Research"
    r"\2-step-segmentation-kidney-tumor-and-cyst"
    r"\data\KiTS23"
)


class_case_counts = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
}


for case_dir in sorted(DATA_ROOT.glob("case_*")):

    seg_path = case_dir / "segmentation.nii.gz"

    segmentation = np.asarray(
        nib.load(seg_path).dataobj
    )

    labels = np.unique(segmentation)

    for label in labels:
        label = int(label)

        if label in class_case_counts:
            class_case_counts[label] += 1


print("Cases containing each label:")
print("Background:", class_case_counts[0])
print("Kidney:", class_case_counts[1])
print("Tumor:", class_case_counts[2])
print("Cyst:", class_case_counts[3])