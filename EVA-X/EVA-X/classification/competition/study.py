import torch


def aggregate_study_logits(study_ids, image_logits, study_order=None):
    """Average image logits per Study_id, preserving requested study order."""
    if image_logits.ndim != 2:
        raise ValueError(f"image_logits must have shape [images, classes], got {tuple(image_logits.shape)}")
    if len(study_ids) != image_logits.shape[0]:
        raise ValueError("study_ids and image_logits must contain the same number of images")
    if not study_ids:
        raise ValueError("cannot aggregate an empty image batch")

    sums = {}
    counts = {}
    observed_order = []
    for study_id, logits in zip(study_ids, image_logits):
        try:
            exists = study_id in sums
        except TypeError as error:
            raise ValueError(f"Study_id must be hashable: {study_id!r}") from error
        if exists:
            sums[study_id] = sums[study_id] + logits
            counts[study_id] += 1
        else:
            sums[study_id] = logits
            counts[study_id] = 1
            observed_order.append(study_id)

    order = observed_order if study_order is None else list(study_order)
    if len(order) != len(set(order)):
        raise ValueError("study_order must not contain duplicate Study_id values")
    if set(order) != set(sums):
        missing = set(order) - set(sums)
        unexpected = set(sums) - set(order)
        raise ValueError(f"Study_id mismatch; missing predictions: {missing}, unexpected predictions: {unexpected}")

    return order, torch.stack([sums[study_id] / counts[study_id] for study_id in order])
