import json
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(r"D:\project\dataset")
TRAIN_SOURCE = DATA_ROOT / "chexpert_train_split.csv"
VAL_SOURCE = DATA_ROOT / "chexpert_val_1100.csv"
TEST_SOURCE = DATA_ROOT / "chexpert_test_1200.csv"
TRAIN_OUTPUT = DATA_ROOT / "chexpert_train_expandedval.csv"
VAL_OUTPUT = DATA_ROOT / "chexpert_val_expanded_consolidation.csv"
MANIFEST_OUTPUT = DATA_ROOT / "chexpert_expandedval_manifest.json"
HOLDOUT_POSITIVE_PATIENTS = 50
SEED = 20260927
LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def load_frontal(path):
    frame = pd.read_csv(path)
    frame = frame.loc[frame["Frontal/Lateral"].eq("Frontal")].copy()
    frame["_patient_id"] = frame["Path"].str.extract(r"(patient\d+)")[0]
    if frame["_patient_id"].isna().any():
        raise ValueError(f"Could not parse patient ID from {path}")
    return frame


def patient_ids(frame):
    return set(frame["_patient_id"])


def summary(frame):
    return {
        "images": int(len(frame)),
        "patients": int(frame["_patient_id"].nunique()),
        "positive_images": {
            label: int(frame[label].eq(1).sum()) for label in LABELS
        },
        "positive_patients": {
            label: int(frame.loc[frame[label].eq(1), "_patient_id"].nunique())
            for label in LABELS
        },
    }


def main():
    for path in (TRAIN_SOURCE, VAL_SOURCE, TEST_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (TRAIN_OUTPUT, VAL_OUTPUT, MANIFEST_OUTPUT):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    train = load_frontal(TRAIN_SOURCE)
    val = load_frontal(VAL_SOURCE)
    test = load_frontal(TEST_SOURCE)

    if train["Path"].duplicated().any() or val["Path"].duplicated().any():
        raise ValueError("Duplicate image paths found in train or validation")
    if patient_ids(train) & patient_ids(val):
        raise ValueError("Source train and validation patients overlap")
    if patient_ids(train) & patient_ids(test):
        raise ValueError("Source train and test patients overlap")
    if patient_ids(val) & patient_ids(test):
        raise ValueError("Source validation and test patients overlap")

    positive_patients = train.loc[train["Consolidation"].eq(1), "_patient_id"].drop_duplicates()
    if len(positive_patients) < HOLDOUT_POSITIVE_PATIENTS:
        raise ValueError("Not enough Consolidation-positive patients to hold out")
    selected = positive_patients.sample(
        n=HOLDOUT_POSITIVE_PATIENTS, random_state=SEED
    )
    selected_ids = set(selected)

    heldout = train.loc[train["_patient_id"].isin(selected_ids)].copy()
    reduced_train = train.loc[~train["_patient_id"].isin(selected_ids)].copy()
    expanded_val = pd.concat([val, heldout], ignore_index=True)

    if patient_ids(reduced_train) & patient_ids(expanded_val):
        raise ValueError("Patient leakage remains after split")
    if patient_ids(test) & patient_ids(expanded_val):
        raise ValueError("Expanded validation overlaps test patients")

    reduced_train.drop(columns="_patient_id").to_csv(TRAIN_OUTPUT, index=False)
    expanded_val.drop(columns="_patient_id").to_csv(VAL_OUTPUT, index=False)
    manifest = {
        "seed": SEED,
        "holdout_positive_patients_requested": HOLDOUT_POSITIVE_PATIENTS,
        "heldout_patient_ids": sorted(selected_ids),
        "sources": {
            "train": str(TRAIN_SOURCE),
            "validation": str(VAL_SOURCE),
            "test_check_only": str(TEST_SOURCE),
        },
        "outputs": {"train": str(TRAIN_OUTPUT), "validation": str(VAL_OUTPUT)},
        "train": summary(reduced_train),
        "validation": summary(expanded_val),
        "test_patient_overlap": 0,
        "train_validation_patient_overlap": 0,
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
