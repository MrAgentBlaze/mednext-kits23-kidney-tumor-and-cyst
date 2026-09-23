import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """
    Multiclass Dice + Cross-Entropy loss.

    Dice is calculated for foreground classes only:
        1 = kidney
        2 = tumor
        3 = cyst

    Background (class 0) is excluded from the Dice calculation.
    """

    def __init__(
        self,
        dice_weight=0.5,
        ce_weight=0.5,
        smooth=1e-5,
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits:
            [B, C, D, H, W]

        targets:
            [B, D, H, W]
        """

        # ---------------------------------
        # Cross Entropy
        # ---------------------------------

        ce_loss = F.cross_entropy(
            logits,
            targets,
        )

        # ---------------------------------
        # Dice
        # ---------------------------------

        num_classes = logits.shape[1]

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        targets_one_hot = F.one_hot(
            targets,
            num_classes=num_classes,
        )

        # [B,D,H,W,C] -> [B,C,D,H,W]
        targets_one_hot = targets_one_hot.permute(
            0, 4, 1, 2, 3
        ).float()

        dice_losses = []

        # Ignore background class 0
        for class_index in range(1, num_classes):

            prediction = probabilities[:, class_index]
            target = targets_one_hot[:, class_index]

            prediction = prediction.reshape(
                prediction.shape[0],
                -1,
            )

            target = target.reshape(
                target.shape[0],
                -1,
            )

            intersection = (
                prediction * target
            ).sum(dim=1)

            denominator = (
                prediction.sum(dim=1)
                + target.sum(dim=1)
            )

            dice = (
                (2.0 * intersection + self.smooth)
                / (denominator + self.smooth)
            )

            dice_losses.append(
                1.0 - dice.mean()
            )

        dice_loss = torch.stack(
            dice_losses
        ).mean()

        # ---------------------------------
        # Combined loss
        # ---------------------------------

        total_loss = (
            self.dice_weight * dice_loss
            + self.ce_weight * ce_loss
        )

        return total_loss