import unittest
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from competition.dataset import CompetitionDataset, LABEL_COLUMNS
from competition.labels import class_id_to_index, normalize_competition_label, NUM_MODEL_OUTPUTS
from competition.losses import CompetitionLoss, decode_competition_logits
from competition.metrics import compute_competition_metrics
from competition.predict import predict_studies
from competition.study import aggregate_study_logits
from engines.engine_finetune import evaluate_chestxray


class CompetitionHelpersTest(unittest.TestCase):
    def test_class_ids_are_one_based(self):
        self.assertEqual(class_id_to_index("1"), 0)
        self.assertEqual(class_id_to_index(10), 9)
        with self.assertRaises(ValueError):
            class_id_to_index(0)

    def test_uncertain_labels_are_preserved_for_strategy_specific_loss(self):
        for index in range(10):
            self.assertEqual(normalize_competition_label("", index), 0.0)
            self.assertEqual(normalize_competition_label("NaN", index), 0.0)
            self.assertEqual(normalize_competition_label("null", index), 0.0)
        for index in range(10):
            self.assertEqual(normalize_competition_label(-1, index), -1.0)
            self.assertEqual(normalize_competition_label("-1", index), -1.0)
        self.assertEqual(normalize_competition_label("1", 0), 1.0)
        self.assertEqual(normalize_competition_label("0", 4), 0.0)
        with self.assertRaises(ValueError):
            normalize_competition_label(2, 0)

    def test_strategy_loss_masks_ignored_labels_and_trains_ones(self):
        logits = torch.zeros((1, NUM_MODEL_OUTPUTS), requires_grad=True)
        targets = torch.zeros((1, 10))
        targets[0, 0] = -1
        targets[0, 1] = -1
        targets[0, 4] = -1
        targets[0, 7] = -1
        loss = CompetitionLoss()(logits, targets)
        loss.backward()
        self.assertEqual(logits.grad[0, 11].item(), 0.0)  # U-Ignore
        self.assertEqual(logits.grad[0, 3].item(), 0.0)   # U-SelfTrained first stage
        self.assertLess(logits.grad[0, 6].item(), 0.0)    # U-Ones maps -1 to positive

    def test_multiclass_prediction_renormalizes_positive_and_negative(self):
        logits = torch.zeros((1, NUM_MODEL_OUTPUTS))
        logits[0, 0:3] = torch.tensor([0.0, 1.0, 9.0])
        probabilities = decode_competition_logits(logits)
        self.assertAlmostEqual(probabilities[0, 0].item(), torch.sigmoid(torch.tensor(1.0)).item())

    def test_study_logits_average_and_follow_requested_order(self):
        order, logits = aggregate_study_logits(
            ["s2", "s1", "s2"],
            torch.tensor([[1.0, 3.0], [5.0, 7.0], [3.0, 5.0]]),
            study_order=["s1", "s2"],
        )
        self.assertEqual(order, ["s1", "s2"])
        torch.testing.assert_close(logits, torch.tensor([[5.0, 7.0], [2.0, 4.0]]))

    def test_prediction_returns_study_probabilities(self):
        class Echo(torch.nn.Module):
            def forward(self, images):
                return images

        first = torch.zeros((2, NUM_MODEL_OUTPUTS))
        first[:, 0:3] = torch.tensor([0.0, 1.0, 9.0])
        first[:, 3] = torch.tensor([1.0, 3.0])
        second = torch.zeros((1, NUM_MODEL_OUTPUTS))
        second[:, 0:3] = torch.tensor([0.0, 1.0, 9.0])
        second[:, 3] = 5.0
        loader = [
            (first, ["s2", "s2"]),
            (second, ["s1"]),
        ]
        study_ids, probabilities = predict_studies(
            Echo(), loader, ["s1", "s2"], torch.device("cpu")
        )
        self.assertEqual(study_ids, ["s1", "s2"])
        np.testing.assert_allclose(
            probabilities,
            decode_competition_logits(torch.cat((second, torch.stack(((first[0] + first[1]) / 2,))))).numpy(),
        )

    def test_dataset_uses_soft_pseudo_labels_only_for_self_trained_unknowns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new("RGB", (8, 8), color="white").save(root / "a.png")
            row = {"image_path": "a.png", "Subject_id": "p1", "Study_id": "s1"}
            row.update({column: "0" for column in LABEL_COLUMNS})
            row["label_1"] = "-1"
            row["label_4"] = "-1"
            manifest = root / "train.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            pseudo = root / "pseudo.csv"
            pseudo.write_text(
                "Study_id," + ",".join(f"probability_{i}" for i in range(1, 11)) + "\n"
                "s1," + ",".join(["0.5", "0.37"] + ["0.5"] * 8) + "\n",
                encoding="utf-8",
            )
            dataset = CompetitionDataset(manifest, root, training=True, pseudo_label_csv=pseudo)
            _, target = dataset[0]
            self.assertAlmostEqual(target[1].item(), 0.37)
            self.assertEqual(target[4].item(), -1.0)

    def test_auc_is_undefined_when_a_class_has_one_target_value(self):
        targets = np.zeros((4, 10), dtype=np.int64)
        targets[:, 0] = [0, 0, 1, 1]
        probabilities = np.tile(np.linspace(0.1, 0.9, 4)[:, None], (1, 10))
        result = compute_competition_metrics(targets, probabilities)
        self.assertIsNone(result["macro_auc"])
        self.assertEqual(len(result["missing_auc_classes"]), 9)
        self.assertAlmostEqual(result["auc_per_class"][0], 1.0)

    def test_metrics_exclude_uncertain_validation_labels(self):
        targets = np.zeros((4, 10), dtype=np.int64)
        targets[:, 0] = [0, 1, -1, -1]
        probabilities = np.full((4, 10), 0.5)
        probabilities[:, 0] = [0.1, 0.9, 0.99, 0.01]
        result = compute_competition_metrics(targets, probabilities)
        self.assertEqual(result["auc_per_class"][0], 1.0)
        self.assertEqual(result["f1_per_class"][0], 1.0)

    def test_dataset_samples_one_image_per_study_for_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for image_name, study_id in (("a.png", "s1"), ("b.png", "s1"), ("c.png", "s2")):
                Image.new("RGB", (8, 8), color="white").save(root / image_name)
                row = {"image_path": image_name, "Subject_id": "p1", "Study_id": study_id}
                row.update({column: "0" for column in LABEL_COLUMNS})
                rows.append(row)
            manifest = root / "train.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            dataset = CompetitionDataset(manifest, root, training=True)
            self.assertEqual(len(dataset), 2)
            image, labels = dataset[0]
            self.assertEqual(tuple(image.size), (8, 8))
            self.assertEqual(tuple(labels.shape), (10,))

            validation = CompetitionDataset(manifest, root, training=False)
            self.assertEqual(len(validation), 3)
            self.assertEqual(validation[1][2], "s1")

    def test_unlabeled_test_manifest_does_not_require_subject_or_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new("RGB", (8, 8), color="white").save(root / "test.png")
            manifest = root / "test.csv"
            manifest.write_text("image_path,Study_id\ntest.png,s-test\n", encoding="utf-8")
            dataset = CompetitionDataset(manifest, root, labeled=False)
            self.assertEqual(len(dataset), 1)
            image, study_id = dataset[0]
            self.assertEqual(tuple(image.size), (8, 8))
            self.assertEqual(study_id, "s-test")

    def test_validation_engine_reports_study_level_metrics(self):
        class Echo(torch.nn.Module):
            def forward(self, images):
                return images

        rows = []
        for study_index in range(20):
            target = torch.tensor(
                [(study_index + class_index) % 2 for class_index in range(10)],
                dtype=torch.float32,
            )
            logits = torch.zeros(NUM_MODEL_OUTPUTS)
            logits[0:3] = torch.tensor([-2.0, 2.0, 0.0]) if target[0] else torch.tensor([2.0, -2.0, 0.0])
            logits[3] = target[1] * 4 - 2
            logits[4] = target[2] * 4 - 2
            logits[5] = target[3] * 4 - 2
            logits[6] = target[4] * 4 - 2
            logits[7:10] = torch.tensor([-2.0, 2.0, 0.0]) if target[5] else torch.tensor([2.0, -2.0, 0.0])
            logits[10] = target[6] * 4 - 2
            logits[11] = target[7] * 4 - 2
            logits[12:15] = torch.tensor([-2.0, 2.0, 0.0]) if target[8] else torch.tensor([2.0, -2.0, 0.0])
            logits[15] = target[9] * 4 - 2
            rows.append((logits, target, f"s{study_index}"))
        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_chestxray(
                DataLoader(rows, batch_size=5),
                Echo(),
                torch.device("cpu"),
                SimpleNamespace(dataset="competition", nb_classes=10, log_dir=temp_dir),
            )
        self.assertAlmostEqual(result["auc_avg"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)
        self.assertEqual(len(result["study_ids"]), 20)


if __name__ == "__main__":
    unittest.main()
