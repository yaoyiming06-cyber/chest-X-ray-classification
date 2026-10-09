import csv
import math
import random
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .labels import (
    COMPETITION_LABELS,
    NUM_COMPETITION_CLASSES,
    UNCERTAINTY_POLICIES,
    normalize_competition_label,
)


LABEL_COLUMNS = tuple(f"label_{index}" for index in range(NUM_COMPETITION_CLASSES))


class CompetitionDataset(Dataset):
    """Read the competition's normalized, one-row-per-image CSV manifest."""

    def __init__(self, csv_path, image_root, transform=None, training=False, labeled=True, pseudo_label_csv=None):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.training = training
        self.labeled = labeled
        self.studies = OrderedDict()
        pseudo_labels = self._read_pseudo_labels(pseudo_label_csv) if training and pseudo_label_csv else {}
        self_trained_indices = tuple(
            index for index, policy in enumerate(UNCERTAINTY_POLICIES)
            if policy == "U-SelfTrained"
        )

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "Study_id"}
            if training:
                required.add("Subject_id")
            available_labels = set(reader.fieldnames or ()) & set(LABEL_COLUMNS)
            if available_labels and available_labels != set(LABEL_COLUMNS):
                raise ValueError(f"{self.csv_path} must contain all 10 label columns or none")
            self.labeled = self.labeled and available_labels == set(LABEL_COLUMNS)
            if training and not self.labeled:
                raise ValueError("training manifests must contain all 10 label columns")
            if self.labeled:
                required.update(LABEL_COLUMNS)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{self.csv_path} is missing manifest columns: {sorted(missing)}")

            for row_number, row in enumerate(reader, start=2):
                image_path = (row.get("image_path") or "").strip()
                subject_id = (row.get("Subject_id") or "").strip()
                study_id = (row.get("Study_id") or "").strip()
                if not image_path or not study_id or (training and not subject_id):
                    raise ValueError(f"{self.csv_path}:{row_number} has an empty path or required ID")
                if self.labeled:
                    try:
                        labels = tuple(
                            normalize_competition_label(row[column], index)
                            for index, column in enumerate(LABEL_COLUMNS)
                        )
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"{self.csv_path}:{row_number} has invalid labels") from error
                else:
                    labels = None

                if labels is not None and study_id in pseudo_labels:
                    labels = tuple(
                        pseudo_labels[study_id][index]
                        if label == -1 and index in self_trained_indices else label
                        for index, label in enumerate(labels)
                    )

                study = self.studies.setdefault(
                    study_id,
                    {"subject_id": subject_id, "labels": labels, "images": []},
                )
                if study["subject_id"] != subject_id:
                    raise ValueError(f"Study_id {study_id!r} maps to more than one Subject_id")
                if study["labels"] != labels:
                    raise ValueError(f"Images in Study_id {study_id!r} have inconsistent labels")
                study["images"].append(image_path)

        if not self.studies:
            raise ValueError(f"No image rows found in {self.csv_path}")
        if pseudo_label_csv:
            for study_id, study in self.studies.items():
                if any(study["labels"][index] == -1 for index in self_trained_indices) and study_id not in pseudo_labels:
                    raise ValueError(f"Missing self-training pseudo labels for Study_id {study_id!r}")
            unexpected = set(pseudo_labels) - set(self.studies)
            if unexpected:
                raise ValueError(f"Pseudo labels contain unknown Study_id values: {sorted(unexpected)[:5]}")
        self.image_rows = [
            (study_id, image_path, study["labels"])
            for study_id, study in self.studies.items()
            for image_path in study["images"]
        ]
        self.study_ids = list(self.studies)
        print(
            f"CompetitionDataset: {self.csv_path} "
            f"({len(self.studies)} studies, {len(self.image_rows)} images, "
            f"{len(COMPETITION_LABELS)} classes, labeled={self.labeled}, training={self.training})"
        )

    @staticmethod
    def _read_pseudo_labels(csv_path):
        values = {}
        columns = tuple(f"probability_{index}" for index in range(1, NUM_COMPETITION_CLASSES + 1))
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Study_id", *columns}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{csv_path} is missing pseudo-label columns: {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                study_id = (row.get("Study_id") or "").strip()
                if not study_id or study_id in values:
                    raise ValueError(f"{csv_path}:{row_number} has an empty or duplicate Study_id")
                probabilities = tuple(float(row[column]) for column in columns)
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities):
                    raise ValueError(f"{csv_path}:{row_number} has probabilities outside [0, 1]")
                values[study_id] = probabilities
        return values

    def __len__(self):
        return len(self.studies) if self.training else len(self.image_rows)

    def __getitem__(self, index):
        if self.training:
            study_id = self.study_ids[index]
            study = self.studies[study_id]
            image_path = random.choice(study["images"])
            labels = study["labels"]
        else:
            study_id, image_path, labels = self.image_rows[index]

        image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if not self.labeled:
            return image, study_id
        target = torch.tensor(labels, dtype=torch.float32)
        if self.training:
            return image, target
        return image, target, study_id
