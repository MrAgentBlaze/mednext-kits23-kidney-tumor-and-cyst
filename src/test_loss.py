import torch

from losses import DiceCELoss


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


loss_fn = DiceCELoss().to(device)


# Simulated MedNeXt output
logits = torch.randn(
    2,
    4,
    32,
    128,
    128,
    device=device,
)


# Simulated KiTS23 labels
targets = torch.randint(
    0,
    4,
    (
        2,
        32,
        128,
        128,
    ),
    device=device,
)


loss = loss_fn(
    logits,
    targets,
)


print(f"Device: {device}")
print(f"Logits shape:  {tuple(logits.shape)}")
print(f"Targets shape: {tuple(targets.shape)}")
print(f"Loss: {loss.item():.6f}")
print(f"Loss finite: {torch.isfinite(loss).item()}")


assert loss.ndim == 0
assert torch.isfinite(loss)
assert loss.item() > 0


print()
print("Dice + CE loss test PASSED.")