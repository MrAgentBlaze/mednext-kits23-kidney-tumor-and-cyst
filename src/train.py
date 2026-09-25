import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.kits23_dataset import KiTS23Dataset
from load_split import load_case_list
from losses import DiceCELoss
from metrics import calculate_dice_scores

from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import (
    create_mednext_v1,
)


# ==================================================
# Configuration
# ==================================================

DATA_ROOT = (
    r"D:\Brian Lala\Research"
    r"\2-step-segmentation-kidney-tumor-and-cyst"
    r"\data\KiTS23"
)

TRAIN_SPLIT = r"configs\train_cases.txt"
VAL_SPLIT = r"configs\val_cases.txt"

PATCH_SIZE = (32, 128, 128)

BATCH_SIZE = 2

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0

OUTPUT_DIR = Path("outputs") / "mednext_s"

BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth"
LAST_MODEL_PATH = OUTPUT_DIR / "last_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if device.type == "cuda":
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ==================================================
# Output directory
# ==================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# Load splits
# ==================================================

train_cases = load_case_list(
    TRAIN_SPLIT
)

val_cases = load_case_list(
    VAL_SPLIT
)

print(
    f"Training cases: {len(train_cases)}"
)

print(
    f"Validation cases: {len(val_cases)}"
)


# ==================================================
# Datasets
# ==================================================

train_dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=PATCH_SIZE,
    cases=train_cases,
    sampling_mode="balanced",
)

val_dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=PATCH_SIZE,
    cases=val_cases,
    sampling_mode="foreground",
)


# ==================================================
# DataLoaders
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ==================================================
# Model
# ==================================================

model = create_mednext_v1(
    num_input_channels=1,
    num_classes=4,
    model_id="S",
    kernel_size=3,
    deep_supervision=False,
).to(device)


# ==================================================
# Loss
# ==================================================

criterion = DiceCELoss(
    dice_weight=0.5,
    ce_weight=0.5,
).to(device)


# ==================================================
# Optimizer
# ==================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ==================================================
# AMP
# ==================================================

scaler = torch.amp.GradScaler("cuda")


# ==================================================
# Training history
# ==================================================

with open(
    HISTORY_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "epoch",
            "train_loss",
            "val_loss",
            "kidney_dice",
            "tumor_dice",
            "cyst_dice",
            "mean_foreground_dice",
        ]
    )


# ==================================================
# Best score
# ==================================================

best_mean_dice = -1.0


# ==================================================
# Training
# ==================================================

print()
print("=" * 60)
print(
    f"Starting MedNeXt-S training "
    f"for {NUM_EPOCHS} epochs"
)
print("=" * 60)


for epoch in range(1, NUM_EPOCHS + 1):

    # ------------------------------------------------
    # Training phase
    # ------------------------------------------------

    model.train()

    train_loss_total = 0.0

    for batch in train_loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        train_loss_total += loss.item()

    train_loss = (
        train_loss_total
        / len(train_loader)
    )


    # ------------------------------------------------
    # Validation phase
    # ------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    kidney_scores = []
    tumor_scores = []
    cyst_scores = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                outputs = model(images)

                val_loss = criterion(
                    outputs,
                    labels,
                )

            val_loss_total += (
                val_loss.item()
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            scores = calculate_dice_scores(
                predictions,
                labels,
            )

            kidney_scores.append(
                scores["kidney"]
            )

            tumor_scores.append(
                scores["tumor"]
            )

            cyst_scores.append(
                scores["cyst"]
            )


    val_loss = (
        val_loss_total
        / len(val_loader)
    )

    kidney_dice = (
        sum(kidney_scores)
        / len(kidney_scores)
    )

    tumor_dice = (
        sum(tumor_scores)
        / len(tumor_scores)
    )

    cyst_dice = (
        sum(cyst_scores)
        / len(cyst_scores)
    )

    mean_foreground_dice = (
        kidney_dice
        + tumor_dice
        + cyst_dice
    ) / 3.0


    # ------------------------------------------------
    # Save history
    # ------------------------------------------------

    with open(
        HISTORY_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                epoch,
                train_loss,
                val_loss,
                kidney_dice,
                tumor_dice,
                cyst_dice,
                mean_foreground_dice,
            ]
        )


    # ------------------------------------------------
    # Checkpoints
    # ------------------------------------------------

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "kidney_dice": kidney_dice,
        "tumor_dice": tumor_dice,
        "cyst_dice": cyst_dice,
        "mean_foreground_dice": mean_foreground_dice,
    }

    torch.save(
        checkpoint,
        LAST_MODEL_PATH,
    )

    if mean_foreground_dice > best_mean_dice:

        best_mean_dice = (
            mean_foreground_dice
        )

        torch.save(
            checkpoint,
            BEST_MODEL_PATH,
        )

        best_marker = " <-- BEST"

    else:

        best_marker = ""


    # ------------------------------------------------
    # Epoch results
    # ------------------------------------------------

    print()
    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    print(
        f"  Train loss: "
        f"{train_loss:.6f}"
    )

    print(
        f"  Val loss:   "
        f"{val_loss:.6f}"
    )

    print(
        f"  Kidney Dice: "
        f"{kidney_dice:.4f}"
    )

    print(
        f"  Tumor Dice:  "
        f"{tumor_dice:.4f}"
    )

    print(
        f"  Cyst Dice:   "
        f"{cyst_dice:.4f}"
    )

    print(
        f"  Mean Dice:   "
        f"{mean_foreground_dice:.4f}"
        f"{best_marker}"
    )


# ==================================================
# Finished
# ==================================================

print()
print("=" * 60)
print("Training completed.")
print("=" * 60)

print(
    f"Best mean foreground Dice: "
    f"{best_mean_dice:.4f}"
)

print(
    f"Best model: "
    f"{BEST_MODEL_PATH}"
)

print(
    f"Last model: "
    f"{LAST_MODEL_PATH}"
)

print(
    f"History: "
    f"{HISTORY_PATH}"
)