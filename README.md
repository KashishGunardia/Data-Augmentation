# YOLO Training Data Augmentation

This script augments the YOLO training dataset located in the `datax/` directory.

For every source image in:

```text
datax/images/train/
```

that has a matching YOLO label file in:

```text
datax/labels/train/
```

the script creates **3 augmented copies by default** and saves them alongside the original training images and labels.

Validation data is **never modified**.

---

## Directory Structure

The expected dataset structure is:

```text
project/
│
├── augment.py
│
└── datax/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   │
    │   └── val/
    │       └── ...
    │
    └── labels/
        ├── train/
        │   ├── image1.txt
        │   ├── image2.txt
        │   └── ...
        │
        └── val/
            └── ...
```

After running the script:

```text
datax/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image1_aug1.jpg
│   │   ├── image1_aug2.jpg
│   │   ├── image1_aug3.jpg
│   │   ├── image2.jpg
│   │   ├── image2_aug1.jpg
│   │   └── ...
│   │
│   └── val/
│       └── ...
│
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image1_aug1.txt
    │   ├── image1_aug2.txt
    │   ├── image1_aug3.txt
    │   └── ...
    │
    └── val/
        └── ...
```

---

## What the Script Does

The script:

1. Finds all `.jpg`, `.jpeg`, and `.png` images inside `datax/images/train/`.
2. Ignores images that already contain `_aug` in their filename.
3. Reads the corresponding YOLO label file.
4. Applies random image transformations.
5. Automatically transforms the bounding boxes along with the image.
6. Creates 3 augmented copies for each source image.
7. Saves each augmented image in the training image directory.
8. Saves the corresponding transformed YOLO labels.
9. Leaves the validation dataset completely untouched.

---

## Augmentations Used

The script uses the **Albumentations** library.

### 1. Random 90° Rotation

```python
A.RandomRotate90(p=1.0)
```

Every augmented image receives a random 90-degree rotation.

Possible rotations include:

* 0°
* 90°
* 180°
* 270°

The YOLO bounding boxes are rotated together with the image.

---

### 2. Random Affine Transformation

```python
A.Affine(
    scale=(0.9, 1.1),
    translate_percent=(-0.05, 0.05),
    p=0.3
)
```

With a probability of 30%, the image receives:

* Scaling between 90% and 110%
* Horizontal translation
* Vertical translation

The maximum translation is approximately 5% of the image dimensions.

---

### 3. Random Brightness and Contrast

```python
A.RandomBrightnessContrast(
    brightness_limit=0.1,
    contrast_limit=0.1,
    p=0.2
)
```

With a probability of 20%, the image brightness and contrast are slightly changed.

This helps the model handle different lighting conditions.

---

### 4. Gaussian Noise

```python
A.GaussNoise(p=0.1)
```

With a probability of 10%, random Gaussian noise is added to the image.

This can help the model become more robust to noisy images.

---

## YOLO Bounding Box Handling

The script expects labels in standard YOLO format:

```text
class_id center_x center_y width height
```

For example:

```text
0 0.512500 0.430000 0.250000 0.300000
```

The bounding boxes are passed to Albumentations using:

```python
format="yolo"
```

This is important because geometric transformations such as rotation, scaling, and translation must also update the bounding boxes.

The script uses:

```python
min_visibility=0.2
```

This means a bounding box must remain at least partially visible after augmentation. Boxes that become too heavily cropped can be removed.

It also uses:

```python
clip=True
```

so bounding boxes are clipped to remain within the image boundaries.

---

## Configuration

The main configuration is at the top of the script:

```python
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "datax"
)

AUGS_PER_IMAGE = 3
```

### Number of Augmentations

Change:

```python
AUGS_PER_IMAGE = 3
```

to control how many augmented copies are generated per source image.

For example:

```python
AUGS_PER_IMAGE = 5
```

creates:

```text
image1_aug1.jpg
image1_aug2.jpg
image1_aug3.jpg
image1_aug4.jpg
image1_aug5.jpg
```

---

## Installation

Install the required Python packages:

```powershell
pip install opencv-python albumentations
```

If you are using a virtual environment, activate it first.

---

## Running the Script

Place the script next to the `datax` folder:

```text
project/
├── augment.py
└── datax/
```

Then run:

```powershell
python augment.py
```

The script will print information such as:

```text
Found 100 source images in .../datax/images/train
  augmented: image1.jpg
  augmented: image2.jpg
  augmented: image3.jpg
  ...
Done. Created 300 augmented image/label pairs in .../datax/images/train
```

---

## Naming Convention

Original:

```text
door_001.jpg
door_001.txt
```

Augmented copies:

```text
door_001_aug1.jpg
door_001_aug1.txt

door_001_aug2.jpg
door_001_aug2.txt

door_001_aug3.jpg
door_001_aug3.txt
```

The image and label always use the same base filename so YOLO can match them correctly.

---

## Important: Validation Data

The script intentionally operates **only on the training dataset**:

```text
datax/images/train/
datax/labels/train/
```

It does **not** modify:

```text
datax/images/val/
datax/labels/val/
```

This is important for preventing **data leakage**.

If an original drawing/tile belongs to validation, an augmented version of that same drawing should not be added to training. Otherwise, the model could effectively see very similar data during training and validation, producing misleading validation results.

Therefore:

> **Augmentation is applied only to training images.**

---

## Re-running the Script

The script ignores files containing `_aug`:

```python
and "_aug" not in f
```

Therefore, augmented images are not used as new source images.

For example, if the directory contains:

```text
image1.jpg
image1_aug1.jpg
image1_aug2.jpg
image1_aug3.jpg
```

only:

```text
image1.jpg
```

is treated as the original source.

This prevents exponential augmentation such as:

```text
image1
  ↓
image1_aug1
  ↓
image1_aug1_aug1
  ↓
...
```

However, **re-running the script will overwrite the same `_aug1`, `_aug2`, etc. files with new random augmentations** rather than creating additional numbered copies.

---

## Expected Dataset Growth

If there are:

```text
100 original training images
```

and:

```python
AUGS_PER_IMAGE = 3
```

the script creates:

```text
100 original images
+
300 augmented images
=
400 training images
```

The number of label files increases in the same way.

### Formula

```text
Augmented images = Original training images × AUGS_PER_IMAGE
```

```text
Total training images =
Original training images × (1 + AUGS_PER_IMAGE)
```

For example:

| Original Images | Augments/Image | New Augmented Images | Total Images |
| --------------: | -------------: | -------------------: | -----------: |
|             100 |              3 |                  300 |          400 |
|             500 |              3 |                1,500 |        2,000 |
|           1,000 |              3 |                3,000 |        4,000 |

---

## Why Augment the Dataset?

Data augmentation increases the variety of training examples without requiring new manually labelled images.

For object detection, it can help the
