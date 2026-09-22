from collections import Counter

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


for i in range(10):

    sample = dataset[i]

    labels = sample["label"].unique().tolist()

    print(
        f"{sample['case']}: labels = {labels}"
    )