#!/usr/bin/env python3
"""Class-level CUB-200-2011 attribute labels (concept.npy) and species labels (class.npy)."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

KOH_112_INDICES = [
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
    51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101,
    106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153,
    157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194, 196, 198,
    202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240,
    242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293,
    294, 298, 299, 304, 305, 308, 309, 310, 311
]
NUM_ATTRIBUTES = 312


def load_from_raw_cub(cub_dir, output_dir):
    """Write concept.npy [N, 112] and class.npy [N] in image-id order."""
    classes = {}
    with open(cub_dir / "image_class_labels.txt") as f:
        for line in f:
            img_id, class_id = line.strip().split()
            classes[int(img_id)] = int(class_id) - 1

    attrs = np.zeros((len(classes), NUM_ATTRIBUTES), dtype=np.int64)
    with open(cub_dir / "attributes" / "image_attribute_labels.txt") as f:
        for line in f:
            parts = line.strip().split()
            attrs[int(parts[0]) - 1, int(parts[1]) - 1] = int(parts[2])

    classes_np = np.array([classes[i + 1] for i in range(len(classes))], dtype=np.int64)
    class_to_images = defaultdict(list)
    for img_id, cls in classes.items():
        class_to_images[cls].append(img_id - 1)
    class_attrs = {cls: (attrs[idxs].mean(axis=0) > 0.5).astype(int) for cls, idxs in class_to_images.items()}

    koh_indices = [i - 1 for i in KOH_112_INDICES]
    concept_np = np.zeros((len(classes), 112), dtype=np.int64)
    for img_id in range(len(classes)):
        concept_np[img_id] = class_attrs[classes_np[img_id]][koh_indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "concept.npy", concept_np)
    np.save(output_dir / "class.npy", classes_np)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cub-dir", type=Path, required=True, help="Path to the raw CUB_200_2011 directory")
    parser.add_argument("--output", type=Path, default=Path("data/cub200_finetuned"), help="Output directory")
    args = parser.parse_args()
    load_from_raw_cub(args.cub_dir, args.output)
    print(f"wrote concept.npy and class.npy to {args.output}", flush=True)


if __name__ == "__main__":
    main()
