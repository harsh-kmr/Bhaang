# Bhaang

A Python library for medical imaging research. Provides dataset loaders, model management utilities, training helpers, and evaluation metrics designed for PyTorch-based workflows.

This library accompanies the MICCAI submission by Harsh Kumar (harshkumar1@iisc.ac.in), IISc Bangalore.

---

## Installation

```bash
git clone https://github.com/<your-repo>/bhaang.git
cd bhaang
pip install -e .
```

**Dependencies** are installed automatically: `numpy`, `torch`, `torchvision`, `pillow`, `transformers`, `tqdm`, `tabulate`, `medmnist`, `albumentations`, `pycocotools`.

---

## ⚠️ Hardcoded Paths — Action Required

Several dataset loaders contain **absolute paths from the development machine** that will not work on your system. You must update them before use:

| File | Line | Hardcoded Path | What to change it to |
|------|------|----------------|----------------------|
| [bhaang/dataset/medmnist/read_data.py](bhaang/dataset/medmnist/read_data.py#L31) | 31 | `/home/harsh/Lab_Work/dataset/medmnist/` | Your local directory to download/store MedMNIST `.npz` files |
| [bhaang/dataset/LC25000/read_data.py](bhaang/dataset/LC25000/read_data.py#L8) | 8 | `/mnt/65af7ccf-.../LC25000/lung_colon_image_set` | Root of your LC25000 dataset (must contain `train_labels.csv`, `val_labels.csv`, `test_labels.csv`) |
| [bhaang/dataset/HAM10000/read_data.py](bhaang/dataset/HAM10000/read_data.py#L9) | 9 | `/mnt/65af7ccf-.../dataverse_files` | Root of your HAM10000 dataset (must contain `HAM10000_metadata_with_images_*.csv`) |
| [bhaang/dataset/brain_tumour_segentation/read_data.py](bhaang/dataset/brain_tumour_segentation/read_data.py#L17) | 17 | `/home/harsh/Lab_Work/Bhaang/bhaang/dataset/brain_tumour_segentation/` | Path to your brain tumour segmentation dataset directory (must contain `train/`, `test/`, `valid/` subdirectories with COCO `.json` annotations) |

> **Note:** The MedMNIST loader can auto-download data if `download=True` (default), but it still writes to the hardcoded root path. Update line 31 before using it.

---

## Modules

### 1. Datasets (`bhaang.dataset`)

All dataset modules expose a `pytorch_dataset` class compatible with `torch.utils.data.DataLoader`.

#### MedMNIST

Wraps the [MedMNIST](https://medmnist.com/) benchmark. Supports all MedMNIST datasets at sizes 28, 64, 128, and 224.

```python
from bhaang.dataset.medmnist.read_data import pytorch_dataset

# Download and load OrganAMNIST at 64x64
dataset = pytorch_dataset(
    split='train',          # 'train', 'val', or 'test'
    data_flag='organamnist',
    download=True,
    as_rgb=False,           # True to force 3-channel output
    size=64,
    transform=None
)

img, label = dataset[0]    # img: torch.Tensor, label: torch.Tensor (scalar)
print(len(dataset))
print(dataset)             # prints full dataset info
```

**Remember to update `self.root` on line 31** to a writable directory on your machine.

#### LC25000 — Lung and Colon Cancer Histopathology

5-class classification: `colon_n`, `colon_aca`, `lung_n`, `lung_aca`, `lung_scc`.

```python
from bhaang.dataset.LC25000.read_data import pytorch_dataset

dataset = pytorch_dataset(
    split='train',              # 'train', 'val', or 'test'
    image_size=(224, 224),
    as_rgb=True,
    transform=None
)

img, label = dataset[0]        # label is an integer 0–4
```

Expects `train_labels.csv`, `val_labels.csv`, `test_labels.csv` at `data_path` (line 8), each with columns `abs_filepath` and `label`.

#### HAM10000 — Skin Lesion Classification

7-class dermoscopy dataset.

```python
from bhaang.dataset.HAM10000.read_data import pytorch_dataset

dataset = pytorch_dataset(
    split='train',
    image_size=(128, 128),
    as_rgb=True,
    transform=None
)

img, label = dataset[0]
```

Expects `HAM10000_metadata_with_images_train.csv`, `..._val.csv`, `..._test.csv` at `data_dir` (line 9), each with columns `image_path` and `dx`.

#### MATEK19 — Blood Cell Morphology

15-class blood cell classification dataset.

```python
from bhaang.dataset.MATEK19.read_data import pytorch_dataset

dataset = pytorch_dataset(split='train')
img, label = dataset[0]
```

#### Brain Tumour Segmentation

COCO-format binary segmentation dataset (640×640 images).

```python
from bhaang.dataset.brain_tumour_segentation.read_data import pytorch_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    ToTensorV2()
])

dataset = pytorch_dataset(split='train', transform=transform)
img, mask = dataset[0]     # img: (C, H, W) float, mask: (H, W) float
```

Expects `train/`, `test/`, `valid/` subdirectories under `data_path` (line 17), each containing images and a `_annotations.coco.json` file.

---

### 2. Model Management (`bhaang.Medical_imaging.model`)

```python
from bhaang.Medical_imaging.model import Model_master

# Load from Torch Hub
mm = Model_master(model_provider='torch')
model = mm.get_model('pytorch/vision', 'resnet18', weights='DEFAULT')

# Load from Hugging Face Hub
mm = Model_master(model_provider='huggingface')
model = mm.get_model(None, 'google/vit-base-patch16-224')

# Load a locally saved model (torch.save'd full model)
mm = Model_master(model_provider='local')
model = mm.get_model('/path/to/model.pt')

# Inspect the model
mm.get_accurate_param_count()

import torch
sample = torch.randn(1, 3, 224, 224)
mm.display_model_layers(sample_input=sample)

# Replace the classification head for transfer learning
mm.add_classification_head(num_classes=5)

# Save
mm.save_model('/path/to/output_model.pt')
```

---

### 3. Metrics (`bhaang.Medical_imaging.metrics`)

```python
from bhaang.Medical_imaging.metrics import get_metrics_classification, get_metrics_regression

# Classification (mode: 'binary', 'micro', 'macro', 'samples', 'weighted')
metrics = get_metrics_classification(y_true, y_pred, mode='macro', device='cpu')
# Returns: {'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'}

# Regression
metrics = get_metrics_regression(y_true, y_pred, device='cpu')
# Returns: {'mae', 'mse', 'rmse', 'r2'}
```

Accepts both `torch.Tensor` and `np.ndarray` inputs.

---

### 4. Early Stopping (`bhaang.engine.earlystopping`)

```python
from bhaang.engine.earlystopping import EarlyStopping

early_stopping = EarlyStopping(
    mode='min',             # 'min' for loss, 'max' for accuracy
    patience=10,
    verbose=True,
    delta=0.0005,
    path='checkpoints/best_model.pt'
)

for epoch in range(max_epochs):
    # ... training loop ...
    val_loss = evaluate(model, val_loader)

    early_stopping(val_loss, model, optimizer=optimizer, epoch=epoch)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# Restore best model
checkpoint = EarlyStopping.load_checkpoint(
    path='checkpoints/best_model.pt',
    model=model,
    optimizer=optimizer
)
print(f"Best epoch: {checkpoint['epoch']}, Best score: {checkpoint['best_score']}")
```

---

### 5. Logging (`bhaang.Medical_imaging.logger`)

```python
from bhaang.Medical_imaging.logger import CSVLogger, TextLogger

csv_log = CSVLogger(log_dir='logs/', filename='metrics')
csv_log.log({'epoch': 1, 'loss': 0.42, 'accuracy': 0.87})

text_log = TextLogger(log_dir='logs/', filename='training')
text_log.log("Epoch 1 complete.")
```

Log files are auto-named with a date suffix to avoid overwrites.

---

## Project Structure

```
bhaang/
├── dataset/
│   ├── medmnist/           # MedMNIST wrapper
│   ├── LC25000/            # Lung & colon cancer histopathology
│   ├── HAM10000/           # Skin lesion classification
│   ├── MATEK19/            # Blood cell morphology
│   └── brain_tumour_segentation/   # COCO-format segmentation
├── Medical_imaging/
│   ├── model.py            # Model_master: load, inspect, modify, save
│   ├── metrics.py          # Classification and regression metrics
│   └── logger.py           # CSV and text loggers
└── engine/
    └── earlystopping.py    # EarlyStopping with checkpoint save/load
```


