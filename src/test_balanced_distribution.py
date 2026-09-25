from collections import Counter

from data.kits23_dataset import KiTS23Dataset


DATA_ROOT = r"D:\Brian Lala\Research\2-step-segmentation-kidney-tumor-and-cyst\data\KiTS23"


dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=(32, 128, 128),
    sampling_mode="balanced",
)


NUM_SAMPLES = 1000

choices = [
    dataset._choose_sampling_mode()
    for _ in range(NUM_SAMPLES)
]

counts = Counter(choices)

print("Balanced sampling distribution")
print("------------------------------")

for mode in ["random", "foreground", "tumor", "cyst"]:

    count = counts[mode]
    percentage = count / NUM_SAMPLES * 100

    print(
        f"{mode:12s}: "
        f"{count:4d}/{NUM_SAMPLES} "
        f"({percentage:5.1f}%)"
    )