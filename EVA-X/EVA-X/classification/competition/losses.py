import torch
import torch.nn.functional as F
from torch import nn

from .labels import (
    NUM_COMPETITION_CLASSES,
    NUM_MODEL_OUTPUTS,
    UNCERTAINTY_POLICIES,
)


def decode_competition_logits(logits):
    """Convert mixed binary/three-class outputs to ten positive probabilities."""
    if logits.ndim != 2 or logits.shape[1] != NUM_MODEL_OUTPUTS:
        raise ValueError(f"logits must have shape [batch, {NUM_MODEL_OUTPUTS}]")
    probabilities = logits.new_empty((logits.shape[0], NUM_COMPETITION_CLASSES))
    output_index = 0
    for label_index, policy in enumerate(UNCERTAINTY_POLICIES):
        if policy == "U-MultiClass":
            group = logits[:, output_index:output_index + 3].softmax(dim=1)
            probabilities[:, label_index] = group[:, 1] / (group[:, 0] + group[:, 1]).clamp_min(1e-12)
            output_index += 3
        else:
            probabilities[:, label_index] = logits[:, output_index].sigmoid()
            output_index += 1
    return probabilities


class CompetitionLoss(nn.Module):
    def __init__(self, ones_label_smoothing=0.0):
        super().__init__()
        if not 0 <= ones_label_smoothing < 1:
            raise ValueError("ones_label_smoothing must be in [0, 1)")
        self.ones_label_smoothing = ones_label_smoothing

    def forward(self, logits, targets):
        if logits.ndim != 2 or logits.shape[1] != NUM_MODEL_OUTPUTS:
            raise ValueError(f"logits must have shape [batch, {NUM_MODEL_OUTPUTS}]")
        if targets.shape != (logits.shape[0], NUM_COMPETITION_CLASSES):
            raise ValueError(f"targets must have shape [batch, {NUM_COMPETITION_CLASSES}]")

        losses = []
        output_index = 0
        for label_index, policy in enumerate(UNCERTAINTY_POLICIES):
            target = targets[:, label_index]
            if policy == "U-MultiClass":
                classes = torch.where(target == -1, 2, target).long()
                losses.append(F.cross_entropy(
                    logits[:, output_index:output_index + 3], classes, reduction="none"
                ).mean())
                output_index += 3
                continue

            prediction = logits[:, output_index]
            output_index += 1
            if policy == "U-Ones":
                target = torch.where(target == -1, 1.0, target)
                if self.ones_label_smoothing:
                    amount = self.ones_label_smoothing
                    target = target * (1 - amount) + 0.5 * amount
                losses.append(F.binary_cross_entropy_with_logits(prediction, target))
            elif policy == "Binary":
                if torch.any(target == -1):
                    raise ValueError("No Finding uses binary labels and cannot contain -1")
                losses.append(F.binary_cross_entropy_with_logits(prediction, target))
            else:
                valid = target >= 0
                if valid.any():
                    losses.append(F.binary_cross_entropy_with_logits(prediction[valid], target[valid]))
        return torch.stack(losses).mean()
