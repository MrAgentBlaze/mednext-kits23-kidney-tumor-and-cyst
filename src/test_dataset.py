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

print("Number of cases:", len(dataset))

sample = dataset[0]

print("Case:", sample["case"])
print("Image shape:", sample["image"].shape)
print("Image dtype:", sample["image"].dtype)
print("Image min:", sample["image"].min().item())
print("Image max:", sample["image"].max().item())

print("Label shape:", sample["label"].shape)
print("Label dtype:", sample["label"].dtype)
print("Labels:", sample["label"].unique().tolist())