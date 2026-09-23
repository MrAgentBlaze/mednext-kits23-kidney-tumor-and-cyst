import torch


def dice_score(prediction, target, class_index, smooth=1e-5):
    """
    Calculate Dice score for one class.

    prediction:
        [B, D, H, W] integer class predictions

    target:
        [B, D, H, W] integer ground-truth labels
    """

    pred_class = prediction == class_index
    target_class = target == class_index

    intersection = (
        pred_class & target_class
    ).sum().float()

    pred_sum = pred_class.sum().float()
    target_sum = target_class.sum().float()

    dice = (
        2.0 * intersection + smooth
    ) / (
        pred_sum + target_sum + smooth
    )

    return dice.item()


def calculate_dice_scores(prediction, target):
    """
    Calculate Dice for kidney, tumor and cyst.

    Classes:
        1 = kidney
        2 = tumor
        3 = cyst
    """

    return {
        "kidney": dice_score(
            prediction,
            target,
            class_index=1,
        ),
        "tumor": dice_score(
            prediction,
            target,
            class_index=2,
        ),
        "cyst": dice_score(
            prediction,
            target,
            class_index=3,
        ),
    }