import csv
import json
import random
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path(r"D:\project\dataset")
SOURCE = DATA_DIR / "chexpert_train_5.csv"
LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
TARGETS = {"val": 1100, "test": 1200}


def effective_label(value, label):
    if value in (None, ""):
        return 0
    value = float(value)
    if value == -1:
        return int(label in {"Atelectasis", "Edema"})
    return int(value > 0)


def subset_with_exact_size(groups, target, rng):
    groups = list(groups)
    rng.shuffle(groups)
    previous = [-2] * (target + 1)
    previous[0] = -1

    for index, (patient, rows) in enumerate(groups):
        size = len(rows)
        if size > target:
            continue
        for total in range(target, size - 1, -1):
            if previous[total] == -2 and previous[total - size] != -2:
                previous[total] = index

    if previous[target] == -2:
        return None

    selected = set()
    total = target
    while total:
        index = previous[total]
        patient, rows = groups[index]
        selected.add(patient)
        total -= len(rows)
    return selected


def has_both_classes(rows):
    for label in LABELS:
        values = {effective_label(row.get(label), label) for row in rows}
        if values != {0, 1}:
            return False
    return True


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        rows = [row for row in reader if row["Frontal/Lateral"] == "Frontal"]
        fields = reader.fieldnames

    groups = defaultdict(list)
    for row in rows:
        patient = row["Path"].split("/")[1]
        groups[patient].append(row)
    group_items = list(groups.items())

    chosen = None
    for seed in range(2026, 12026):
        rng = random.Random(seed)
        val_patients = subset_with_exact_size(group_items, TARGETS["val"], rng)
        if val_patients is None:
            continue
        remaining = [(patient, patient_rows) for patient, patient_rows in group_items if patient not in val_patients]
        test_patients = subset_with_exact_size(remaining, TARGETS["test"], rng)
        if test_patients is None:
            continue

        val_rows = [row for patient in val_patients for row in groups[patient]]
        test_rows = [row for patient in test_patients for row in groups[patient]]
        if has_both_classes(val_rows) and has_both_classes(test_rows):
            chosen = seed, val_patients, test_patients, val_rows, test_rows
            break

    if chosen is None:
        raise RuntimeError("Could not find patient-level splits with all five labels represented as 0 and 1")

    seed, val_patients, test_patients, val_rows, test_rows = chosen
    train_patients = set(groups) - val_patients - test_patients
    train_rows = [row for patient in train_patients for row in groups[patient]]

    outputs = {
        "chexpert_train_split.csv": train_rows,
        "chexpert_val_1100.csv": val_rows,
        "chexpert_test_1200.csv": test_rows,
    }
    for name, output_rows in outputs.items():
        output_rows.sort(key=lambda row: row["Path"])
        with (DATA_DIR / name).open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)

    manifest = {
        "source": str(SOURCE),
        "seed": seed,
        "split_unit": "patient",
        "frontal_only": True,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "labels": LABELS,
    }
    (DATA_DIR / "chexpert_local_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    for name, output_rows in outputs.items():
        print(name, {label: sum(effective_label(row.get(label), label) for row in output_rows) for label in LABELS})


if __name__ == "__main__":
    main()
