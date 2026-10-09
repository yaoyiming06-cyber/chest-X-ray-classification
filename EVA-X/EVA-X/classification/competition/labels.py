import math


COMPETITION_LABELS = (
    "Enlarged Cardiomediastinum",
    "Pneumothorax",
    "Consolidation",
    "Pneumonia",
    "Edema",
    "Cardiomegaly",
    "Atelectasis",
    "Lung Opacity",
    "Pleural Effusion",
    "No Finding",
)

NUM_COMPETITION_CLASSES = len(COMPETITION_LABELS)
COMPETITION_CLASS_IDS = tuple(range(1, NUM_COMPETITION_CLASSES + 1))
UNCERTAINTY_POLICIES = (
    "U-MultiClass",
    "U-SelfTrained",
    "U-SelfTrained",
    "U-SelfTrained",
    "U-Ones",
    "U-MultiClass",
    "U-Ones",
    "U-Ignore",
    "U-MultiClass",
    "Binary",
)
MULTICLASS_LABEL_INDICES = tuple(
    index for index, policy in enumerate(UNCERTAINTY_POLICIES) if policy == "U-MultiClass"
)
NUM_MODEL_OUTPUTS = NUM_COMPETITION_CLASSES + 2 * len(MULTICLASS_LABEL_INDICES)


def class_id_to_index(class_id):
    try:
        numeric_id = int(class_id)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid competition class ID: {class_id!r}") from error
    if str(class_id).strip() != str(numeric_id) or numeric_id not in COMPETITION_CLASS_IDS:
        raise ValueError(f"Competition class ID must be in 1..{NUM_COMPETITION_CLASSES}: {class_id!r}")
    return numeric_id - 1


def normalize_competition_label(value, label_index):
    """Keep uncertain labels intact so the training policy can handle them."""
    if not 0 <= label_index < NUM_COMPETITION_CLASSES:
        raise ValueError(f"label_index must be in 0..{NUM_COMPETITION_CLASSES - 1}: {label_index}")
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "nan", "none", "null")):
        return 0.0
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid competition label value: {value!r}") from error
    if math.isnan(numeric_value):
        return 0.0
    if numeric_value == -1:
        return -1.0
    if numeric_value in (0, 1):
        return numeric_value
    raise ValueError(f"Competition labels must be 0, 1, -1, or missing: {value!r}")
