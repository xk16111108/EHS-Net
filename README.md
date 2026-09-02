# EHS-Net

Official PyTorch model implementation of EHS-Net for medical image segmentation.

This repository contains the model architecture and checkpoint-loading utilities only. Training scripts, datasets, preprocessing pipelines, and evaluation code are not included.

## Repository structure

```text
EHS-Net-model/
`-- model/
    |-- __init__.py
    |-- architecture.py
    `-- ehsnet.py
```

## Requirements

- Python 3.9 or later
- PyTorch
- NumPy
- einops
- timm
- mamba-ssm or a compatible `selective_scan` implementation
- NATTEN

Install PyTorch, `mamba-ssm`, and NATTEN using versions compatible with your CUDA environment. The remaining Python dependencies can be installed with:

```bash
pip install numpy einops timm
```

## Model initialization

```python
import torch

from model import EHSNet

model = EHSNet(input_channels=3, num_classes=1)
model.eval()

x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    mask_logits, edge_logits = model(x)

print(mask_logits.shape)
print(edge_logits.shape)
```

For single-channel images, the default three-channel model automatically repeats the input channel:

```python
x = torch.randn(1, 1, 256, 256)
mask_logits, edge_logits = model(x)
```

## Loading checkpoints

Load weights into an initialized model:

```python
from model import EHSNet, load_checkpoint

model = EHSNet(input_channels=3, num_classes=1)
load_checkpoint(model, "path/to/checkpoint.pth", map_location="cpu")
```

Alternatively, create the model directly from a checkpoint. The number of output classes is inferred from the final prediction layer:

```python
from model import build_model_from_checkpoint

model = build_model_from_checkpoint(
    "path/to/checkpoint.pth",
    map_location="cpu",
    input_channels=3,
)
```

The loader supports raw state dictionaries and checkpoints stored under `model`, `state_dict`, `model_state_dict`, `net`, or `network`. It also removes common `module.` and `model.` prefixes and maps legacy `vmunet.*` parameter names to the current `network.*` structure.

## Output

`EHSNet.forward` returns two tensors:

1. `mask_logits`: segmentation logits resized to the input spatial resolution.
2. `edge_logits`: auxiliary boundary logits resized to the input spatial resolution.

Apply an activation appropriate for the task during inference, such as `torch.sigmoid` for binary segmentation or `torch.softmax` for multiclass segmentation.

## Notes

- Input tensors must use the `NCHW` layout.
- Checkpoint loading is strict by default.
- The repository intentionally contains model code only, apart from this README.
