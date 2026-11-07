#!/usr/bin/env python3
"""Fine-tune the Inception-v3 backbone on CUB-200 and write its features (latent.npy)."""
import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ccw_net.run import pick_device

LABEL_FILES = ("concept.npy", "class.npy")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class CUBConceptDataset(Dataset):
    """CUB-200 images with their concept and class labels."""

    def __init__(self, image_paths, concepts, classes, transform):
        self.image_paths = image_paths
        self.concepts = concepts
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.image_paths[idx]).convert("RGB"))
        return img, self.concepts[idx], self.classes[idx]


class JointCBM(nn.Module):
    """Joint CBM of Koh et al. (2020): Inception-v3 features, 112 concept logits, a linear species head."""

    def __init__(self, weights_path, n_concepts=112, n_classes=200):
        super().__init__()
        self.backbone = models.inception_v3(weights=None, aux_logits=True)
        self.backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=False)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.backbone.AuxLogits.fc = nn.Linear(768, n_concepts)
        self.concept_heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(n_concepts)])
        self.class_head = nn.Linear(n_concepts, n_classes)

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, models.InceptionOutputs):
            features = features.logits
        concept_logits = torch.cat([h(features) for h in self.concept_heads], dim=1)
        return self.class_head(torch.sigmoid(concept_logits)), concept_logits, features


def read_image_paths(cub_dir):
    """Image paths in CUB-200 image-id order."""
    with open(cub_dir / "images.txt") as f:
        return [str(cub_dir / "images" / line.strip().split(maxsplit=1)[1]) for line in f]


def validation_pass(model, loader, device):
    """The recipe's validation forward pass (epoch 1 and every 5th); it keeps the released features' random stream."""
    model.eval()
    with torch.no_grad():
        for imgs, _, _ in loader:
            model(imgs.to(device))


def fine_tune(model, train_loader, val_loader, args, device):
    """SGD on the normalised joint loss of Koh et al."""
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0004, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=0.1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, concepts_b, classes_b in train_loader:
            imgs, concepts_b, classes_b = imgs.to(device), concepts_b.to(device), classes_b.to(device)
            optimizer.zero_grad()
            class_logits, concept_logits, _ = model(imgs)
            n_attr = concept_logits.shape[1]
            concept_loss = F.binary_cross_entropy_with_logits(
                concept_logits, concepts_b, reduction="sum") / (len(imgs) * n_attr)
            raw_loss = F.cross_entropy(class_logits, classes_b) + args.attr_loss_weight * n_attr * concept_loss
            (raw_loss / (1.0 + args.attr_loss_weight * n_attr)).backward()
            optimizer.step()
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            validation_pass(model, val_loader, device)
        print(f"epoch {epoch}/{args.epochs}", flush=True)


def extract_features(model, loader, device):
    """Backbone features of every image in `loader`, in loader order."""
    model.eval()
    features = []
    with torch.no_grad():
        for imgs, _, _ in loader:
            features.append(model(imgs.to(device))[2].cpu().numpy())
    return np.concatenate(features, axis=0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cub-dir", type=Path, required=True,
                        help="CUB_200_2011 directory that contains images.txt and images/")
    parser.add_argument("--weights", type=Path, required=True,
                        help="torchvision Inception-v3 ImageNet state dict (inception_v3_google-0cc3c7bd.pth)")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("CCW_DATA_ROOT", REPO_ROOT / "data")))
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="concept.npy and class.npy from preprocess_cub.py (default: <data-root>/cub200_finetuned)")
    parser.add_argument("--output", type=Path, default=None,
                        help="dataset dir to write (default: <data-root>/cub200_finetuned)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--attr-loss-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, cuda or mps; default picks the best available")
    parser.add_argument("--force", action="store_true", help="refit when the output already has latent.npy")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    warnings.filterwarnings("ignore")
    labels_dir = args.labels_dir or args.data_root / "cub200_finetuned"
    output = args.output or args.data_root / "cub200_finetuned"
    if (output / "latent.npy").exists() and not args.force:
        print(f"{output / 'latent.npy'} exists, nothing to do; pass --force to refit", flush=True)
        return output

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    output.mkdir(parents=True, exist_ok=True)

    image_paths = read_image_paths(args.cub_dir)
    concepts = np.load(labels_dir / "concept.npy")
    classes = np.load(labels_dir / "class.npy")
    train_idx, val_idx = train_test_split(np.arange(len(classes)), test_size=0.2, stratify=classes,
                                          random_state=args.seed)

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def loader(idx, transform, shuffle):
        dataset = CUBConceptDataset(
            [image_paths[i] for i in idx],
            torch.tensor(concepts[idx], dtype=torch.float32),
            torch.tensor(classes[idx], dtype=torch.long),
            transform)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    model = JointCBM(args.weights).to(device)
    fine_tune(model, loader(train_idx, train_transform, True), loader(val_idx, val_transform, False), args, device)
    np.save(output / "latent.npy", extract_features(model, loader(np.arange(len(image_paths)), val_transform, False),
                                                    device))
    if output.resolve() != labels_dir.resolve():
        for name in LABEL_FILES:
            shutil.copyfile(labels_dir / name, output / name)
    print(f"wrote {output / 'latent.npy'}", flush=True)
    return output


if __name__ == "__main__":
    main()
