import numpy as np
from sklearn.metrics import roc_auc_score

from .labels import COMPETITION_LABELS


def compute_competition_metrics(targets, probabilities, threshold=0.5):
    """Compute strict 10-class Macro-AUC and Macro-F1 for study-level outputs."""
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if targets.ndim != 2 or targets.shape[1] != len(COMPETITION_LABELS):
        raise ValueError(f"targets must have shape [studies, {len(COMPETITION_LABELS)}]")
    if targets.shape[0] == 0:
        raise ValueError("targets must contain at least one study")
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have the same shape")
    if not np.isfinite(targets).all() or not np.isin(targets, (-1, 0, 1)).all():
        raise ValueError("competition targets must contain only -1/0/1 values")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    auc_per_class = []
    missing_auc_classes = []
    valid = targets != -1
    for index, label in enumerate(COMPETITION_LABELS):
        known_targets = targets[valid[:, index], index]
        known_probabilities = probabilities[valid[:, index], index]
        if np.unique(known_targets).size < 2:
            auc_per_class.append(None)
            missing_auc_classes.append(label)
        else:
            auc_per_class.append(float(roc_auc_score(known_targets, known_probabilities)))

    predictions = probabilities >= threshold
    true_positives = np.logical_and(np.logical_and(predictions, targets == 1), valid).sum(axis=0)
    false_positives = np.logical_and(np.logical_and(predictions, targets == 0), valid).sum(axis=0)
    false_negatives = np.logical_and(np.logical_and(~predictions, targets == 1), valid).sum(axis=0)
    denominator = 2 * true_positives + false_positives + false_negatives
    f1_per_class = np.divide(
        2 * true_positives,
        denominator,
        out=np.zeros(len(COMPETITION_LABELS), dtype=np.float64),
        where=denominator != 0,
    )

    return {
        "auc_per_class": auc_per_class,
        "macro_auc": None if missing_auc_classes else float(np.mean(auc_per_class)),
        "missing_auc_classes": missing_auc_classes,
        "f1_per_class": f1_per_class.tolist(),
        "macro_f1": float(np.mean(f1_per_class)),
        "threshold": threshold,
    }
