from data.kits23_dataset import KiTS23Dataset

import numpy as np


DATA_ROOT = (
    r"D:\Brian Lala\Research"
    r"\2-step-segmentation-kidney-tumor-and-cyst"
    r"\data\KiTS23"
)


dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=(32, 128, 128),
    sampling_mode="mixed",
)


tumor_count = 0
cyst_count = 0
kidney_count = 0
background_only_count = 0


NUM_SAMPLES = 200


for i in range(NUM_SAMPLES):

    sample = dataset[
        np.random.randint(len(dataset))
    ]

    labels = sample["label"].unique().tolist()

    if 1 in labels:
        kidney_count += 1

    if 2 in labels:
        tumor_count += 1

    if 3 in labels:
        cyst_count += 1

    if labels == [0]:
        background_only_count += 1


print()
print("Mixed sampling test")
print("-------------------")

print(
    f"Samples containing kidney: "
    f"{kidney_count}/{NUM_SAMPLES}"
)

print(
    f"Samples containing tumor:  "
    f"{tumor_count}/{NUM_SAMPLES}"
)

print(
    f"Samples containing cyst:   "
    f"{cyst_count}/{NUM_SAMPLES}"
)

print(
    f"Background-only samples:   "
    f"{background_only_count}/{NUM_SAMPLES}"
)