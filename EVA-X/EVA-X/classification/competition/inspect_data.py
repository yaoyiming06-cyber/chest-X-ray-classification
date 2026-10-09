import argparse
import collections
import csv
import json
from pathlib import Path

if __package__:
    from .labels import COMPETITION_LABELS
else:
    from labels import COMPETITION_LABELS


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TABLE_SUFFIXES = {".csv", ".tsv"}


def normalize_column(name):
    return "".join(char.lower() for char in name if char.isalnum())


def cell_value(row, column):
    return (row.get(column) or "").strip()


def summarize_table(path):
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel

        reader = csv.DictReader(handle, dialect=dialect)
        columns = reader.fieldnames or []
        normalized = {normalize_column(column): column for column in columns}
        subject_column = normalized.get("subjectid")
        study_column = normalized.get("studyid")
        label_column = normalized.get("predictclass")
        subjects = set()
        study_counts = collections.Counter()
        label_counts = collections.Counter()
        samples = []
        row_count = 0

        for row in reader:
            row_count += 1
            if len(samples) < 5:
                samples.append(row)
            subject_id = cell_value(row, subject_column) if subject_column else ""
            study_id = cell_value(row, study_column) if study_column else ""
            label = cell_value(row, label_column) if label_column else ""
            if subject_id:
                subjects.add(subject_id)
            if study_id:
                study_counts[study_id] += 1
            if label:
                label_counts[label] += 1

    return {
        "path": path,
        "columns": columns,
        "row_count": row_count,
        "samples": samples,
        "subjects": subjects,
        "study_counts": study_counts,
        "label_counts": label_counts,
    }


def print_table_report(summary, root):
    path = summary["path"]
    try:
        display_path = path.relative_to(root)
    except ValueError:
        display_path = path
    print(f"\nTable: {display_path}")
    print(f"  Rows: {summary['row_count']}")
    print(f"  Columns: {summary['columns']}")
    print(f"  Samples: {json.dumps(summary['samples'], ensure_ascii=False)}")
    if summary["subjects"]:
        print(f"  Unique Subject_id: {len(summary['subjects'])}")
    if summary["study_counts"]:
        counts = summary["study_counts"].values()
        multiple = sum(count > 1 for count in counts)
        print(
            f"  Unique Study_id: {len(summary['study_counts'])}; "
            f"studies with multiple rows: {multiple}; "
            f"maximum rows per study: {max(counts)}"
        )
    if summary["label_counts"]:
        print(f"  Predict_class counts: {dict(sorted(summary['label_counts'].items()))}")


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def main():
    parser = argparse.ArgumentParser(description="Inspect competition data files and metadata.")
    parser.add_argument("data_root", type=Path, help="Directory containing downloaded competition files")
    parser.add_argument("--train-csv", help="Training metadata path, relative to data_root or absolute")
    parser.add_argument("--valid-csv", help="Validation metadata path, relative to data_root or absolute")
    args = parser.parse_args()

    root = args.data_root.resolve()
    if not root.is_dir():
        parser.error(f"data_root is not a directory: {root}")
    if bool(args.train_csv) != bool(args.valid_csv):
        parser.error("--train-csv and --valid-csv must be supplied together")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    extension_counts = collections.Counter(path.suffix.lower() or "[no extension]" for path in files)
    image_count = sum(count for suffix, count in extension_counts.items() if suffix in IMAGE_SUFFIXES)
    tables = [path for path in files if path.suffix.lower() in TABLE_SUFFIXES]

    print(f"Data root: {root}")
    print(f"Files: {len(files)}; image files (JPG/PNG): {image_count}")
    print(f"Extensions: {dict(sorted(extension_counts.items()))}")
    print(f"Competition classes in model order ({len(COMPETITION_LABELS)}):")
    for index, label in enumerate(COMPETITION_LABELS):
        print(f"  {index}: {label}")
    print("First files:")
    for path in files[:30]:
        print(f"  {path.relative_to(root)} ({path.stat().st_size} bytes)")
    if len(files) > 30:
        print(f"  ... {len(files) - 30} more")

    if not tables:
        print("No CSV/TSV metadata files found.")
    for path in tables:
        print_table_report(summarize_table(path), root)

    if args.train_csv:
        train_path = resolve_path(root, args.train_csv)
        valid_path = resolve_path(root, args.valid_csv)
        train = summarize_table(train_path)
        valid = summarize_table(valid_path)
        if not train["subjects"] or not valid["subjects"]:
            parser.error("Both split files must contain a non-empty Subject_id column")
        overlap = train["subjects"] & valid["subjects"]
        print(f"\nSubject_id overlap between train and validation: {len(overlap)}")
        if overlap:
            print(f"  Example overlapping IDs: {sorted(overlap)[:10]}")


if __name__ == "__main__":
    main()
