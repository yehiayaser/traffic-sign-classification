import argparse
import csv
import json
import random
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.datasets import GTSRB

CLASS_IDS = list(range(43))
IMAGE_SIZE = 96
SEED = 42


class TrainingSigns(Dataset):
    """Read the official training CSVs and retain physical-sign track IDs."""

    def __init__(self, data_dir, transform):
        self.transform = transform
        GTSRB(data_dir, split="train", download=True)
        image_root = data_dir / "gtsrb" / "GTSRB" / "Training"
        self.samples = []
        self.labels = []
        self.groups = []
        for label in CLASS_IDS:
            class_dir = image_root / f"{label:05d}"
            with (class_dir / f"GT-{label:05d}.csv").open(newline="") as annotations:
                for row in csv.DictReader(annotations, delimiter=";"):
                    filename = row["Filename"]
                    match = re.fullmatch(r"(\d{5})_\d{5}\.ppm", filename)
                    if match is None or int(row["ClassId"]) != label:
                        raise ValueError(f"Unexpected GTSRB annotation: {row}")
                    self.samples.append(class_dir / filename)
                    self.labels.append(label)
                    self.groups.append(f"{label}:{match[1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        with Image.open(self.samples[index]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]


def build_model(name, pretrained=True):
    if name == "cnn":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, len(CLASS_IDS)),
        )
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_IDS))
    return model


def scores(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, labels=CLASS_IDS, average="macro", zero_division=0),
    }


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    labels, predictions = [], []
    for images, targets in loader:
        predictions.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
        labels.extend(targets.tolist())
    return labels, predictions


def fit(name, train_set, val_loader, test_loader, args, device):
    torch.manual_seed(SEED)
    model = build_model(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if name == "cnn" else 1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    checkpoint = args.output / f"{name}.pt"
    history = []
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        model.train()
        loss_total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(labels)
        validation = scores(*predict(model, val_loader, device))
        entry = {
            "epoch": epoch,
            "train_loss": loss_total / len(train_set),
            **validation,
            "train_and_validation_seconds": round(perf_counter() - start, 2),
        }
        history.append(entry)
        print(name, entry, flush=True)
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    truth, predictions = predict(model, test_loader, device)
    matrix = confusion_matrix(truth, predictions, labels=CLASS_IDS)
    with (args.output / f"{name}_confusion.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["true/predicted", *CLASS_IDS])
        writer.writerows([label, *row] for label, row in zip(CLASS_IDS, matrix.tolist()))

    errors = sorted(
        [(int(matrix[true, predicted]), true, predicted)
         for true in CLASS_IDS for predicted in CLASS_IDS
         if true != predicted and matrix[true, predicted] > 0],
        reverse=True,
    )[:5]
    return {
        "history": history,
        "best_validation_macro_f1": best_f1,
        "test": scores(truth, predictions),
        "top_confusions": [
            {"true_id": true, "predicted_id": predicted, "count": count}
            for count, true, predicted in errors
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = TrainingSigns(args.data, transform)
    test = GTSRB(args.data, split="test", transform=transform, download=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_ids, val_ids = next(splitter.split(train.samples, train.labels, train.groups))
    train_groups = {train.groups[i] for i in train_ids}
    val_groups = {train.groups[i] for i in val_ids}
    if train_groups & val_groups:
        raise RuntimeError("Physical sign groups overlap between train and validation.")
    train_set = Subset(train, train_ids.tolist())
    val_loader = DataLoader(Subset(train, val_ids.tolist()), batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)

    report = {
        "seed": SEED,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "versions": {"torch": str(torch.__version__)},
        "image_size": IMAGE_SIZE,
        "batch_size": args.batch_size,
        "split": {"train": len(train_set), "validation": len(val_ids), "test": len(test)},
        "models": {},
    }
    for name in ("cnn", "resnet18"):
        report["models"][name] = fit(name, train_set, val_loader, test_loader, args, device)
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
