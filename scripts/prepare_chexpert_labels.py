import csv
from pathlib import Path


DATA_ROOT = Path(r"D:\project\dataset\PNG")
SOURCE_CSV = Path(r"D:\project\findings_fixed.csv")
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
FIELDS = ["Path", "Frontal/Lateral", *LABELS]


def main():
    rows = []
    source_rows = 0
    missing_rows = 0

    with SOURCE_CSV.open(encoding="utf-8-sig", newline="") as source:
        for record in csv.DictReader(source):
            source_rows += 1
            relative_path = Path(record["path_to_image"].replace("\\", "/")).with_suffix(".png")
            if not (DATA_ROOT / relative_path).is_file():
                missing_rows += 1
                continue

            view = "Frontal" if relative_path.stem.lower().endswith("_frontal") else "Lateral"
            rows.append({
                "Path": relative_path.as_posix(),
                "Frontal/Lateral": view,
                **{label: record.get(label, "") for label in LABELS},
            })

    rows.sort(key=lambda row: row["Path"])
    outputs = {
        "chexpert_matched_5.csv": rows,
        "chexpert_train_5.csv": [row for row in rows if row["Path"].startswith("train/")],
        "chexpert_valid_5.csv": [row for row in rows if row["Path"].startswith("valid/")],
    }

    for name, data in outputs.items():
        target = DATA_ROOT.parent / name
        with target.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(data)
        print(f"{target}: {len(data)} rows")

    print(f"source_rows={source_rows} matched={len(rows)} missing={missing_rows}")


if __name__ == "__main__":
    main()
