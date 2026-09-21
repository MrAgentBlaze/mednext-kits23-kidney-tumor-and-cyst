import torch
import torch.nn as nn
import torch.optim as optim

from data.kits23_dataset import KiTS23Dataset
from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import (
    create_mednext_v1,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_ROOT = (
    r"D:\Brian Lala\Research"
    r"\2-step-segmentation-kidney-tumor-and-cyst"
    r"\data\KiTS23"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

dataset = KiTS23Dataset(
    data_root=DATA_ROOT,
    patch_size=(32, 128, 128),
)

sample = dataset[0]

image = sample["image"].unsqueeze(0).to(DEVICE)
target = sample["label"].unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = create_mednext_v1(
    num_input_channels=1,
    num_classes=4,
    model_id="S",
    kernel_size=3,
    deep_supervision=False,
)

model = model.to(DEVICE)


# ---------------------------------------------------------
# Loss + optimizer
# ---------------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
)


# ---------------------------------------------------------
# Initial parameter value
# ---------------------------------------------------------

first_parameter_before = next(
    model.parameters()
).detach().clone()


# ---------------------------------------------------------
# Training step
# ---------------------------------------------------------

model.train()

optimizer.zero_grad()

output = model(image)

loss = criterion(output, target)

print("Loss before backward:", loss.item())

loss.backward()


# ---------------------------------------------------------
# Inspect gradients
# ---------------------------------------------------------

parameters_with_grad = 0
parameters_without_grad = 0
total_gradient = 0.0

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        if parameter.grad is None:
            parameters_without_grad += 1
        else:
            parameters_with_grad += 1
            total_gradient += parameter.grad.abs().sum().item()

print("Parameters with gradients:", parameters_with_grad)
print("Parameters without gradients:", parameters_without_grad)
print("Total absolute gradient:", total_gradient)


# ---------------------------------------------------------
# Inspect first trainable parameter
# ---------------------------------------------------------

for name, parameter in model.named_parameters():

    if parameter.requires_grad and parameter.grad is not None:

        print("First parameter with gradient:", name)
        print("Requires grad:", parameter.requires_grad)

        print(
            "Gradient absolute sum:",
            parameter.grad.abs().sum().item()
        )

        first_parameter_before = parameter.detach().clone()

        optimizer.step()

        first_parameter_after = parameter.detach().clone()

        parameter_change = (
            first_parameter_after - first_parameter_before
        ).abs().sum().item()

        print("Parameter change:", parameter_change)

        break


# ---------------------------------------------------------
# GPU
# ---------------------------------------------------------

print("GPU:", torch.cuda.get_device_name(0))