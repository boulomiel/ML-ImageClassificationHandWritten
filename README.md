# MNIST MLP (PyTorch)

A minimal, modular PyTorch project that trains a Multi‑Layer Perceptron (MLP) to classify handwritten digits from the MNIST dataset.  
It cleanly separates the **model**, **data loading**, **training**, and **validation** logic.

## Project structure

```
.
├── main.py              # Entry point: wires everything together, runs training/validation, plots curves, makes a sample prediction
├── mlp.py               # 3-layer MLP: 784 → 512 → 512 → 10 (ReLU)
├── dataset_loader.py    # MNIST Dataset + DataLoaders (train / validation)
├── trainer.py           # One training epoch over the training DataLoader
├── validator.py         # One evaluation pass over the validation DataLoader
└── predictable.py       # Abstract base class (interface) for components that produce (loss, accuracy)
```

## Requirements

- Python 3.10+ (uses modern type hints like `tuple[float, float]`)
- PyTorch ≥ 2.0
- TorchVision
- Matplotlib
- (Optional) CUDA for GPU acceleration

Quick install (CPU build shown; replace with CUDA build if you have a compatible GPU):

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision matplotlib
```

> **Note on `functorch`:** `validator.py` uses a `Tensor` type hint from `functorch.dim`. In PyTorch 2.x, functorch is merged into PyTorch; if you run into an import error, either install `functorch` or replace that type hint with `from torch import Tensor`.

## Data

By default, `dataset_loader.py` looks for MNIST under:

```
resource/lib/publicdata/data
```

TorchVision can download MNIST automatically if configured to do so.  
If your environment doesn’t already contain the dataset, you can either:

- Point `MNIST_ROOT` in `dataset_loader.py` to a directory you control, and (if needed) enable downloading in the MNIST dataset constructor, **or**
- Create the above directory and let TorchVision store the data there.

Typical transforms include `transforms.ToTensor()` and normalization; adjust them in `dataset_loader.py` if you want to experiment.

## How it works

- **Model** — `MLP` in `mlp.py` is a simple feed‑forward network:
  - Input: 28×28 grayscale image (flattened to 784)
  - Hidden: 512 → 512 with ReLU activations
  - Output: 10 logits (one per MNIST class)

- **Training** — `Trainer` in `trainer.py` iterates once over the training loader, computing:
  - Forward pass → loss (`nn.CrossEntropyLoss`)
  - Accuracy via `argmax` on the logits
  - Backward pass + step with the provided optimizer

  It returns a pair `(epoch_loss, epoch_accuracy)` normalized by the number of batches/samples.

- **Validation** — `Validator` in `validator.py` evaluates the model with `torch.no_grad()`, computes loss/accuracy, and returns `(val_loss, val_accuracy)`.

- **Entry point** — `main.py`:
  - Selects `cuda` if available
  - Instantiates `MLP`, `DatasetLoader`, `Trainer`, `Validator`
  - Trains for a configurable number of epochs (default: 20)
  - Plots training/validation curves with Matplotlib
  - Shows a sample image and prints the predicted class with its probability

## Running

From the project root:

```bash
python main.py
```

You should see epoch‑by‑epoch loss/accuracy in the console, then a Matplotlib figure with curves and a grayscale example digit. The script also prints a line like:

```
Predicted class 7 with probability 0.98
```

## Customization

- **Batch size** — controlled in `dataset_loader.py` (default: 32).
- **Epochs** — change `num_epochs` in `main.py`.
- **Optimizer / LR** — create your optimizer in `main.py`, e.g.:
  ```python
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
  ```
  You can switch to `Adam` easily:
  ```python
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  ```
- **Transforms** — modify in `dataset_loader.py`.
- **Model** — tweak hidden sizes or add dropout/batch norm in `mlp.py`.

## Tips & Troubleshooting

- **Shape mismatches**: Ensure images are flattened to 784 before passing to the MLP. In `main.py` there’s an example of reshaping before a single-image prediction.
- **Dataset not found**: Update `MNIST_ROOT` or enable downloading in the TorchVision MNIST dataset call.
- **`functorch` import**: If you hit `ModuleNotFoundError: functorch`, see the note above in *Requirements*.
- **Determinism**: `torch.manual_seed(0)` is set in `dataset_loader.py` to make runs more reproducible.

## License

Specify your project’s license here (e.g., MIT). MNIST is provided by Yann LeCun and colleagues.

## Acknowledgements

- PyTorch
- TorchVision
- MNIST dataset by Yann LeCun et al.
