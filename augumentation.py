"""
Augment the YOLO training set in datax/ in place.

For every image in datax/images/train (with a matching label file in
datax/labels/train), generate N augmented copies and save them alongside
the originals (images/train, labels/train), each with a unique suffix.

Validation data (images/val, labels/val) is left untouched so no augmented
tile derived from a val drawing can leak into train.
"""

import os
import cv2
import albumentations as A

# ---- config ----------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datax")
IMAGES_DIR = os.path.join(DATA_DIR, "images", "train")
LABELS_DIR = os.path.join(DATA_DIR, "labels", "train")
AUGS_PER_IMAGE = 3          # how many augmented copies to create per source image
IMG_EXTS = (".jpg", ".jpeg", ".png")

transform = A.Compose(
    [
        A.RandomRotate90(p=1.0),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.2,
        ),
        A.GaussNoise(p=0.1),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.2,
        clip=True,
    ),
)


def read_yolo_labels(label_path):
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, x, y, w, h = line.split()
            class_labels.append(int(float(cls)))
            bboxes.append([float(x), float(y), float(w), float(h)])
    return bboxes, class_labels


def write_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, "w") as f:
        for cls, (x, y, w, h) in zip(class_labels, bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    image_files = sorted(
        f
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(IMG_EXTS) and "_aug" not in f
    )
    print(f"Found {len(image_files)} source images in {IMAGES_DIR}")

    created = 0
    for image_name in image_files:
        stem, ext = os.path.splitext(image_name)
        image_path = os.path.join(IMAGES_DIR, image_name)
        label_path = os.path.join(LABELS_DIR, stem + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            print(f"  skip (unreadable): {image_name}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = read_yolo_labels(label_path)

        for i in range(AUGS_PER_IMAGE):
            augmented = transform(
                image=image, bboxes=bboxes, class_labels=class_labels
            )
            aug_image = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["class_labels"]

            out_name = f"{stem}_aug{i+1}{ext}"
            out_image_path = os.path.join(IMAGES_DIR, out_name)
            out_label_path = os.path.join(LABELS_DIR, f"{stem}_aug{i+1}.txt")

            cv2.imwrite(out_image_path, aug_image)
            write_yolo_labels(out_label_path, aug_bboxes, aug_labels)
            created += 1

        print(f"  augmented: {image_name}")

    print(f"Done. Created {created} augmented image/label pairs in {IMAGES_DIR}")


if __name__ == "__main__":
    main()
