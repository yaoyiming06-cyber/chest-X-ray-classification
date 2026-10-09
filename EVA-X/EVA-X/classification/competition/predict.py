import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch
from timm.models import create_model
from torch.utils.data import DataLoader

import models.models_eva
from competition.dataset import CompetitionDataset
from competition.labels import NUM_MODEL_OUTPUTS, NUM_COMPETITION_CLASSES
from competition.losses import decode_competition_logits
from competition.study import aggregate_study_logits
from utils.datasets import build_transform


def config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def load_finetuned_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint or "args" not in checkpoint:
        raise ValueError("checkpoint must contain the EVA-X 'model' state and saved 'args'")
    config = checkpoint["args"]
    model_name = config_value(config, "model")
    input_size = config_value(config, "input_size", 224)
    if not model_name:
        raise ValueError("checkpoint args do not contain the model name")
    model = create_model(
        model_name,
        pretrained=False,
        img_size=input_size,
        num_classes=NUM_MODEL_OUTPUTS,
        drop_rate=config_value(config, "vit_dropout_rate", 0.0),
        drop_path_rate=config_value(config, "drop_path", 0.1),
        attn_drop_rate=config_value(config, "attn_drop_rate", 0.0),
        drop_block_rate=None,
        use_mean_pooling=config_value(config, "use_mean_pooling", False),
        use_checkpoint=config_value(config, "use_checkpoint", False),
        stop_grad_conv1=config_value(config, "stop_grad_conv1", False),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, config


@torch.inference_mode()
def predict_studies(model, loader, study_order, device):
    image_logits = []
    image_study_ids = []
    for images, study_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            image_logits.append(model(images).float().cpu())
        image_study_ids.extend(study_ids)

    logits = torch.cat(image_logits, dim=0)
    study_ids, study_logits = aggregate_study_logits(image_study_ids, logits, study_order)
    return study_ids, decode_competition_logits(study_logits).numpy()


def main():
    parser = argparse.ArgumentParser(description="Run EVA-X competition predictions at Study level.")
    parser.add_argument("--manifest", required=True, type=Path, help="Unlabeled normalized one-row-per-image manifest")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Fine-tuned EVA-X checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Internal Study-level probability CSV")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, saved_args = load_finetuned_model(args.checkpoint, device)
    transform_args = SimpleNamespace(
        model=config_value(saved_args, "model"),
        dataset="competition",
        input_size=config_value(saved_args, "input_size", 224),
        build_timm_transform=False,
    )
    dataset = CompetitionDataset(
        args.manifest,
        args.image_root,
        transform=build_transform(False, transform_args),
        labeled=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    study_ids, probabilities = predict_studies(model, loader, dataset.study_ids, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    probability_columns = [f"probability_{index}" for index in range(1, NUM_COMPETITION_CLASSES + 1)]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Study_id", *probability_columns])
        for study_id, row in zip(study_ids, probabilities):
            writer.writerow([study_id, *[f"{value:.8f}" for value in row]])
    print(f"Wrote {len(study_ids)} Study predictions to {args.output}")


if __name__ == "__main__":
    main()
