# Image restoration with a residual ResNet34 U-Net

Restore RGB photographs affected by synthetic film damage: dirt, dots,
scratches, smut, spots, sprinkles, stain, long hair, and short hair.

**The final model is trained and available locally.** On 1,654 held-out synthetic
test pairs it achieved **38.5015 dB PSNR and 0.985564 SSIM**, compared with
25.0674 dB / 0.908485 for the damaged inputs. Training completed 120 total epochs;
validation selected epoch 119. These scores describe this synthetic urban-image
benchmark; real photograph quality requires separate assessment.

This repository contains the active source code, configurations, launchers, and tests.
Datasets, checkpoints, generated outputs, and archived experiments are kept
locally and are excluded from Git.

The [complete technical report](PROJECT_REPORT.md) and final papers are included:

- [IEEE PDF](reports%20and%20papers/final/film_damage_restoration_ieee_final.pdf) and [Word document](reports%20and%20papers/final/film_damage_restoration_ieee_final.docx)
- [Springer PDF](reports%20and%20papers/final/film_damage_restoration_springer_final.pdf) and [Word document](reports%20and%20papers/final/film_damage_restoration_springer_final.docx)

## Restore an image

Drag an image onto [restore_image.bat](restore_image.bat) on Windows, or run:

```powershell
python restore_image.py "D:\Photos\damaged.jpg"
```

The default model is `checkpoints/final_restoration.pt`. Output is saved beside
the source as `damaged_restored.png`; repeated runs use numbered filenames.
Running without an image path opens a text prompt. An explicit output must be a
new PNG filename:

```powershell
python restore_image.py "D:\Photos\damaged.jpg" --output "D:\Photos\restored.png"
python restore_image.py "D:\Photos\damaged.jpg" --device cpu
```

Inference works offline with local weights, uses CUDA when available, preserves
the oriented image dimensions, and processes large images in overlapping tiles.
Grayscale inputs produce grayscale output. Transparency is composited on white.

## Create synthetic damage

Drag a photo onto [damage_image.bat](damage_image.bat) for an interactive menu,
or specify the choices directly:

```powershell
python damage_image.py "D:\Photos\photo.jpg" --types scratches hair --severity medium --seed 42 --output-dir "D:\Photos\damaged"
```

Types: `dirt`, `dots`, `scratches`, `smut`, `spots`, `sprinkles`, `stain`, `hair`,
`hair-short`. Severities: `light`, `medium`, `heavy`. Training used light and
medium damage. Each generation saves a damaged PNG, mask PNG, and JSON settings.
On disk, **black means damage and white means clean**. The default output folder
is `damaged_photos/`. Existing output files receive numbered names.

## Environment

The recorded training environment used Python 3.11, PyTorch 2.5.1+cu121,
torchvision 0.20.1+cu121, and an NVIDIA RTX 4050 Laptop GPU. Package versions are
recorded in [requirements-v2.txt](requirements-v2.txt). The existing machine
already has the environment. To reproduce it in a compatible environment:

```powershell
python -m pip install -r requirements-v2.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

The checkpoint, datasets, run artifacts, and damage assets are local files;
many are excluded from Git. A source checkout alone may not contain them.

## Current project map

| Location | Purpose |
|---|---|
| `restore_image.py`, `restore_image.bat` | Restore user images |
| `damage_image.py`, `damage_image.bat` | Generate synthetic damage |
| `checkpoints/final_restoration.pt` | Final inference export |
| `configs/restoration_final.yaml` | Final training configuration |
| `configs/restoration_v2.yaml` | Parent training and preparation configuration |
| `src/dataset/` | Current damage adapter and lossless pair preparation |
| `src/model/` | Architecture, losses, training, evaluation, inference |
| `data/clean_splits/` | Source photographs split before corruption |
| `damage dataset/` | PNG artifact assets used by generation |
| `restoration_dataset_v2/` | Current aligned clean/damaged/mask pairs |
| `runs/restoration_v2/` | Parent checkpoint and 35-epoch training history |
| `runs/restoration_final/` | Final 85-epoch continuation, checkpoints, status |
| `results/final_model/` | Final test metrics and 20 comparison images |
| `results/project_report/` | Report statistics and charts in PNG/PDF/SVG |
| `tests/` | Existing regression checks |

## Evaluation and training reference

The completed evaluation is stored locally at `results/final_model/metrics.json`.
To run another evaluation and save a separate result:

```powershell
python src/model/evaluate.py --dataset_dir restoration_dataset_v2 --checkpoint checkpoints/final_restoration.pt --output_json results/final_model/recomputed_metrics.json
```

The final run is complete. Its status is recorded in
`runs/restoration_final/status.json`.
For an interrupted copy of this run, the resume command is:

```powershell
python src/model/train_final.py --config configs/restoration_final.yaml --resume runs/restoration_final/last.pt
```

Use a separate configuration and unused run directory for a new experiment.

Prepare data and start parent training with:

```powershell
python src/dataset/prepare_v2.py --config configs/restoration_v2.yaml
python src/model/train_v2.py --config configs/restoration_v2.yaml
```

Provide the source photographs under `data/clean_splits/` and damage assets under
`damage dataset/` before preparation. Training and inference require the local
files named in their configurations; they are not downloaded with this repository.

Run the regression checks with:

```powershell
python -m unittest discover -s tests
```
