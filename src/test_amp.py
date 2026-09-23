import torch
from torch.utils.data import DataLoader

from data.kits23_dataset import KiTS23Dataset
from load_split import load_case_list
from losses import DiceCELoss

from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import (
    create_mednext_v1,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_ROOT = (
    r"D:\Brian Lala\Research"
    r"\2-step-segmentation-kidney-tumor-and-cyst"
    r"\data\KiTS23"
)

PATCH_SIZE = (32, 128, 128)
BATCH_SIZE = 2
NUM_BATCHES = 5
LEARNING_RATE = 1e-4

TRAIN_SPLIT = r"configs\train_cases.txt"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# Load training cases
# --------------------------------------------------

train_cases = load_case_list(TRAIN_SPLIT)

print(f"Training cases: {len(train_cases)}")


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=PATCH_SIZE,
    cases=train_cases,
    sampling_mode="mixed",
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = create_mednext_v1(
    num_input_channels=1,
    num_classes=4,
    model_id="S",
    kernel_size=3,
    deep_supervision=False,
).to(device)


# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = DiceCELoss(
    dice_weight=0.5,
    ce_weight=0.5,
).to(device)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5,
)


# --------------------------------------------------
# Automatic Mixed Precision
# --------------------------------------------------

scaler = torch.amp.GradScaler("cuda")


# --------------------------------------------------
# Training smoke test
# --------------------------------------------------

model.train()

losses = []

print()
print("Starting 5-batch AMP training smoke test...")
print()


for batch_index, batch in enumerate(train_loader):

    if batch_index >= NUM_BATCHES:
        break

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

    # ----------------------------------------------
    # Forward pass + loss under AMP
    # ----------------------------------------------

    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

    # ----------------------------------------------
    # AMP backward pass
    # ----------------------------------------------

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()

    # ----------------------------------------------
    # Record result
    # ----------------------------------------------

    loss_value = loss.item()

    losses.append(loss_value)

    print(
        f"Batch {batch_index + 1}/{NUM_BATCHES} | "
        f"Loss: {loss_value:.6f} | "
        f"Input: {tuple(images.shape)} | "
        f"Output: {tuple(outputs.shape)}"
    )


# --------------------------------------------------
# Verification
# --------------------------------------------------

initial_loss = losses[0]
final_loss = losses[-1]

print()
print("AMP smoke test completed successfully.")
print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss:   {final_loss:.6f}")

assert all(
    torch.isfinite(
        torch.tensor(loss_value)
    )
    for loss_value in losses
)

print("All losses are finite.")
print("Automatic Mixed Precision pipeline PASSED.")