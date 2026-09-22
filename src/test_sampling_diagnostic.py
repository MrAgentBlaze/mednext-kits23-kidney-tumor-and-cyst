import numpy as np

from data.kits23_dataset import KiTS23Dataset


DATA_ROOT = (
    r"D:\Brian Lala\Research"
    r"\2-step-segmentation-kidney-tumor-and-cyst"
    r"\data\KiTS23"
)


dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=(32, 128, 128),
)


print("Testing 100 samples...\n")

class_counts = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
}

samples_with_tumor = 0
samples_with_cyst = 0
samples_with_foreground = 0


for i in range(100):

    # Pick a random case
    case_index = np.random.randint(len(dataset))

    sample = dataset[case_index]

    labels = sample["label"].numpy()

    unique_labels = np.unique(labels)

    for label in unique_labels:
        class_counts[int(label)] += 1

    if 1 in unique_labels or 2 in unique_labels or 3 in unique_labels:
        samples_with_foreground += 1

    if 2 in unique_labels:
        samples_with_tumor += 1

    if 3 in unique_labels:
        samples_with_cyst += 1


print("Class occurrence across 100 patches:")
print("Background:", class_counts[0])
print("Kidney:", class_counts[1])
print("Tumor:", class_counts[2])
print("Cyst:", class_counts[3])

print()
print("Patches containing foreground:", samples_with_foreground)
print("Patches containing tumor:", samples_with_tumor)
print("Patches containing cyst:", samples_with_cyst)