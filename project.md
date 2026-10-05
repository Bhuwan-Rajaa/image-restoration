I am building an IMAGE RESTORATION project using two existing GitHub repositories:

DAMAGE GENERATION:
https://github.com/DoraStern/FilmDamageSimulator

RESTORATION:
https://github.com/daniela997/DustScratchRemoval

DATASET:
Intel Image Classification dataset

IMPORTANT:
This is an IMAGE RESTORATION project only.

There is NO image colorization component.

Do not convert grayscale to color.
Do not predict color channels.
Do not use a GAN for colorization.
Do not add any colorization architecture.

The task is:

CLEAN IMAGE
→ SYNTHETIC FILM DAMAGE
→ DAMAGED IMAGE + DAMAGE MASK
→ RESTORATION MODEL
→ RESTORED IMAGE
→ COMPARE RESTORED IMAGE WITH ORIGINAL CLEAN IMAGE

Do NOT implement the entire system in one shot.

Work in clearly separated phases.

At the end of every phase:

1. Run the relevant tests.
2. Generate sample outputs where applicable.
3. Report what was implemented.
4. Report what worked.
5. Report what failed.
6. Report any assumptions made.
7. Report the exact next phase.

Do not proceed to a large-scale generation/training run until the current phase has been validated.

---

# 1. PROJECT OBJECTIVE

Build a complete synthetic-data image restoration pipeline using the Intel Image Classification dataset.

The Intel dataset contains several scene classes.

For THIS project, use ONLY:

- buildings
- street

Ignore all other classes.

The classes are used to provide visually different natural/urban scenes for restoration.

The classes are NOT the restoration targets.

The actual restoration target is:

damaged image → original clean image

The class labels should only be retained for controlled analysis of restoration performance.

---

# 2. REPOSITORY ROLES

The two GitHub repositories have DIFFERENT responsibilities.

## Repository A — DAMAGE GENERATION

https://github.com/DoraStern/FilmDamageSimulator

Use this repository ONLY for:

- synthetic film artifact generation
- damage mask generation
- applying synthetic damage to clean images
- artifact assets/statistical generation logic

Do not use its restoration training implementation.

Do not use its restoration model.

Do not train the restoration model from this repository.

The FilmDamageSimulator repository provides scripts for generating synthetic damage masks and applying them to target images, including its `damage_generator` and `synthetic` components. Inspect the actual implementation rather than assuming its CLI supports every individual artifact type.

---

## Repository B — RESTORATION

https://github.com/daniela997/DustScratchRemoval

Use this repository ONLY as the basis for:

- image restoration architecture
- U-Net architecture
- transfer-learning approach
- loss functions
- restoration training/inference logic

Do not use this repository to generate the synthetic dataset.

Inspect its README, notebooks, source code, model definitions, loss functions, preprocessing, training code and inference code before adapting anything.

The repository describes a U-Net-based restoration approach with transfer learning and several VGG16-based perceptual-loss configurations, including an SSIM-based variant. Reuse/adapt those components where appropriate rather than inventing an unrelated restoration architecture.

---

# 3. VERY IMPORTANT: NO COLORIZATION

This project is NOT colorization.

The input is already an RGB image.

The output must also be an RGB image.

Therefore:

Input:
RGB damaged image

Output:
RGB restored image

Target:
RGB clean image

Do NOT:

- convert RGB → LAB for color prediction
- predict `a,b` channels
- use grayscale as the main input
- use a colorization U-Net
- use a GAN for colorization
- add colorization losses
- add colorization metrics

The only objective is removal/reconstruction of synthetic film damage.

---

# 4. REQUIRED DAMAGE TYPES

Use ONLY these seven synthetic damage/artifact types:

1. dirt
2. dots
3. scratches
4. smut
5. spot / spots
6. sprinkle / sprinkles
7. stain

Do NOT intentionally use:

- hair
- long hair
- short hair
- lint
- unrelated film artifacts
- other artifact categories

Do not assume that the FilmDamageSimulator's existing high-level damage categories correspond one-to-one with these names.

Inspect the actual `synthetic/` assets and generator implementation.

Determine exactly how:

- dirt
- dots
- scratches
- smut
- spots
- sprinkles
- stain

are represented.

Create an adapter around the existing simulator so that our project can explicitly select these artifact sources.

Do not create fake random OpenCV circles/lines to substitute for repository artifacts unless the repository genuinely lacks the requested artifact.

---

# 5. INTEL DATASET

Use the Intel Image Classification dataset as the source of clean images.

Only:

buildings
street

are allowed.

Example source:

INTEL_ROOT/
    train/
        buildings/
        street/
    test/
        buildings/
        street/

The actual local structure may differ.

The implementation must detect/configure the dataset root instead of hardcoding a machine-specific path.

Ignore all other Intel classes.

---

# 6. TRAIN / VALIDATION / TEST SPLIT

Prevent data leakage.

The split MUST be performed on the ORIGINAL CLEAN images before any damage variants are created.

Preferred split:

70% training
15% validation
15% testing

Use a configurable random seed.

Keep class distribution approximately balanced/stratified where possible.

CRITICAL:

If one original clean image generates:

image_001_v1
image_001_v2
image_001_v3

ALL of those variants must remain in the same split as the original image.

Never place different variants of the same original image into different splits.

This is essential to prevent leakage.

---

# 7. CLEAN IMAGE PREPROCESSING

Create a configurable preprocessing pipeline.

Initial target:

256 × 256 RGB

but make this configurable.

Possible preprocessing:

- resize
- center crop / random crop where appropriate
- RGB conversion
- tensor conversion
- normalization consistent with the selected restoration model

Do not introduce unnecessary transformations.

Document every preprocessing operation.

---

# 8. DATA AUGMENTATION

Apply augmentation only to training data.

Possible augmentations:

- horizontal flip
- small rotation
- resize/crop
- small brightness variation
- small contrast variation
- small saturation variation

Do not apply random training augmentation to validation/test images.

IMPORTANT:

The augmentation must happen consistently for the clean/damaged pair.

The damage mask must undergo spatial transformations consistently with its corresponding image.

Do not independently transform the damaged image and clean target in a way that breaks pixel correspondence.

---

# 9. DAMAGE GENERATION PIPELINE

For each clean image:

1. Load the original clean RGB image.
2. Resize/preprocess it.
3. Select one or more of the seven allowed artifact types.
4. Generate the artifact/damage mask using FilmDamageSimulator.
5. Apply the generated damage to the clean image using FilmDamageSimulator.
6. Save:
   - clean image
   - damaged image
   - damage mask
   - metadata

The clean image MUST remain untouched.

Never overwrite the source Intel image.

---

# 10. DAMAGE VARIANTS

Support multiple damaged variants per clean image.

Example:

clean:
buildings_000001.jpg

variants:

buildings_000001_v01
buildings_000001_v02
buildings_000001_v03

Each variant should use a different random seed/damage configuration.

Support:

- single damage
- multiple damage types
- mixed damage

Examples:

variant 1:
dots

variant 2:
scratches + stain

variant 3:
dirt + smut + sprinkle

---

# 11. DAMAGE SEVERITY

Support configurable damage severity:

- light
- medium
- heavy

Do not invent severity parameters before inspecting FilmDamageSimulator.

Use the simulator's actual scale/strength parameters where possible.

The three levels should produce visibly different damage quantities/severity.

For example, conceptually:

LIGHT
small/limited damaged area

MEDIUM
moderate damaged area

HEAVY
substantial damaged area

The exact numerical parameters must come from the repository's implementation or be experimentally calibrated.

---

# 12. INDIVIDUAL ARTIFACT EXPERIMENTS

The system must support explicit artifact selection.

Example CLI:

python generate_dataset.py --damage-types dots

python generate_dataset.py --damage-types scratches

python generate_dataset.py --damage-types dirt,scratches

python generate_dataset.py --damage-types dirt,dots,scratches,smut,spot,sprinkle,stain

Also support:

--damage-types all

and:

--damage-types mixed

where `mixed` randomly selects a subset of the seven allowed artifact types.

The selected artifact types must be recorded in metadata.

---

# 13. METADATA

Every damaged sample must have metadata.

Example:

{
    "source_image": "buildings/image_001.jpg",
    "clean_image": "...",
    "damaged_image": "...",
    "mask": "...",
    "class": "buildings",
    "split": "train",
    "damage_types": [
        "dots",
        "scratches",
        "stain"
    ],
    "severity": "medium",
    "seed": 12345,
    "image_size": [256, 256]
}

Also record, where available:

- artifact count
- simulator parameters
- scale
- strength
- selected synthetic asset
- generation timestamp
- variant number

---

# 14. DATASET DIRECTORY STRUCTURE

Use a structure such as:

restoration_dataset/

├── train/
│   ├── buildings/
│   │   ├── clean/
│   │   ├── damaged/
│   │   └── masks/
│   │
│   └── street/
│       ├── clean/
│       ├── damaged/
│       └── masks/
│
├── val/
│   ├── buildings/
│   │   ├── clean/
│   │   ├── damaged/
│   │   └── masks/
│   │
│   └── street/
│       ├── clean/
│       ├── damaged/
│       └── masks/
│
├── test/
│   ├── buildings/
│   │   ├── clean/
│   │   ├── damaged/
│   │   └── masks/
│   │
│   └── street/
│       ├── clean/
│       ├── damaged/
│       └── masks/
│
└── metadata/
    ├── train.json
    ├── val.json
    └── test.json

Maintain exact pairing between:

clean
damaged
mask

---

# 15. MASK FORMAT

Inspect FilmDamageSimulator to determine the exact meaning of its masks.

Do not assume the mask convention.

If the repository uses:

255 = undamaged
0 = damaged

preserve that convention or explicitly convert it into the representation required by the restoration model.

Document the convention.

The final restoration pipeline must know exactly:

- which pixels are damaged
- which pixels are clean/background

Do not silently invert masks.

---

# 16. VISUAL SANITY CHECK

Before generating the full dataset, create debug visualizations.

For each artifact:

dirt
dots
scratches
smut
spot
sprinkle
stain

generate one example showing:

CLEAN | MASK | DAMAGED

Also create mixed-damage examples.

Save:

debug_samples/
    dirt.png
    dots.png
    scratches.png
    smut.png
    spot.png
    sprinkle.png
    stain.png
    mixed.png

The developer agent must inspect these outputs and verify that the correct artifact is actually being generated.

Do NOT proceed to full dataset generation until this is verified.

---

# 17. RESTORATION MODEL SOURCE

Use:

https://github.com/daniela997/DustScratchRemoval

as the restoration-model reference implementation.

First inspect:

- notebooks/
- model architecture
- training code
- loss functions
- transfer-learning implementation
- preprocessing
- inference
- checkpointing

The repository specifically addresses dust/scratch artifact restoration using a U-Net architecture and transfer learning, with multiple VGG16-based perceptual loss variants and an SSIM-enhanced variant.

Adapt this implementation to our synthetic dataset rather than blindly copying the original experiment setup.

---

# 18. RESTORATION INPUT / OUTPUT

Primary experiment:

INPUT:
3-channel RGB damaged image

OUTPUT:
3-channel RGB restored image

TARGET:
3-channel RGB clean image

The restoration network must reconstruct the clean RGB image.

The damage mask can be retained for:

- analysis
- visualization
- optional auxiliary experiments

but the primary restoration experiment should be:

DAMAGED RGB IMAGE → CLEAN RGB IMAGE

Do not make the restoration model a colorization model.

---

# 19. MASK-AWARE OPTIONAL EXPERIMENT

After the RGB-only baseline is working, optionally implement:

INPUT:
RGB damaged image + 1-channel damage mask

OUTPUT:
RGB clean image

This should be considered a separate experiment.

The primary baseline should remain:

3-channel damaged RGB → 3-channel clean RGB

This makes the comparison more meaningful and prevents the project from becoming dependent on an externally supplied perfect damage mask.

---

# 20. RESTORATION TRAINING

Use PyTorch or the framework used by DustScratchRemoval, after inspecting the repository.

Training must support:

- GPU
- configurable image size
- configurable batch size
- configurable learning rate
- configurable number of epochs
- checkpoints
- best checkpoint
- final checkpoint
- resume training
- validation
- reproducible seed

Do not load the entire dataset into RAM.

Use Dataset/DataLoader or the equivalent appropriate mechanism.

---

# 21. LOSSES

Start with the losses used by DustScratchRemoval.

Inspect the repository and identify the exact implementation of:

- image reconstruction loss
- perceptual loss
- content/style components
- SSIM loss

Do not invent a new loss before understanding the original implementation.

Create configurable loss modes, such as:

1. reconstruction baseline
2. perceptual loss model 1
3. perceptual loss model 2
4. perceptual + SSIM model

The exact weights should initially follow the original repository where technically appropriate, then be made configurable for experiments.

Do not include any colorization loss.

---

# 22. TRANSFER LEARNING

Inspect how DustScratchRemoval performs transfer learning.

Determine:

- pretrained network
- frozen layers
- trainable layers
- initialization
- fine-tuning strategy

Reproduce the useful transfer-learning setup where compatible.

Do not claim that a particular pretrained model or layer-freezing setup exists until you verify it from the repository code.

---

# 23. TRAINING EXPERIMENTS

Create reproducible experiments.

At minimum:

EXPERIMENT 1:
RGB damaged → RGB clean
baseline reconstruction loss

EXPERIMENT 2:
RGB damaged → RGB clean
perceptual loss

EXPERIMENT 3:
RGB damaged → RGB clean
perceptual + SSIM loss

OPTIONAL:

EXPERIMENT 4:
RGB damaged + mask → RGB clean

The goal is to determine whether the restoration approach improves as the loss/model configuration becomes more appropriate for film artifacts.

---

# 24. EVALUATION METRICS

Evaluate against the original clean image.

Required metrics:

- PSNR
- SSIM
- MAE
- MSE

For every test image calculate:

A. DAMAGED vs CLEAN

B. RESTORED vs CLEAN

This enables direct measurement of restoration improvement.

Create:

| Metric | Damaged vs Clean | Restored vs Clean | Improvement |
|--------|------------------|-------------------|-------------|
| PSNR   |                  |                   |             |
| SSIM   |                  |                   |             |
| MAE    |                  |                   |             |
| MSE    |                  |                   |             |

For PSNR/SSIM:

higher is generally better.

For MAE/MSE:

lower is better.

---

# 25. CLASS-WISE EVALUATION

Although the Intel class is not the restoration target, analyze whether scene type influences restoration.

Report separately:

- buildings
- street
- overall

Example:

| Class | PSNR Before | PSNR After | SSIM Before | SSIM After |
|-------|-------------|------------|-------------|------------|
| Buildings | | | | |
| Street | | | | |
| Overall | | | | |

---

# 26. DAMAGE-TYPE EVALUATION

Run separate test subsets for:

1. dirt
2. dots
3. scratches
4. smut
5. spot
6. sprinkle
7. stain
8. mixed

Report:

| Damage Type | PSNR Before | PSNR After | SSIM Before | SSIM After | MAE Before | MAE After |
|-------------|-------------|------------|-------------|------------|-------------|-----------|
| Dirt | | | | | | |
| Dots | | | | | | |
| Scratches | | | | | | |
| Smut | | | | | | |
| Spot | | | | | | |
| Sprinkle | | | | | | |
| Stain | | | | | | |
| Mixed | | | | | | |

This is an important part of the project because the seven artifact types are explicitly part of the experimental design.

---

# 27. QUALITATIVE RESULTS

Create restoration visualizations.

For each selected sample show:

CLEAN
→ DAMAGED
→ MASK
→ RESTORED

Generate examples from:

- buildings
- street

Generate examples for:

- light
- medium
- heavy
- mixed damage

Also create side-by-side crops where useful to show whether:

- edges were preserved
- textures were preserved
- scratches disappeared
- spots/dots disappeared
- stains were removed
- new artifacts were introduced

Save to:

results/
    qualitative/
    crops/
    metrics/
    checkpoints/

---

# 28. BEFORE / AFTER COMPARISON

For each restoration result, compute:

damage severity / degradation

versus

restoration quality.

The project should demonstrate that:

Original clean image
→ synthetic damage
→ restoration

produces a restored output quantitatively closer to the original clean image.

Do not judge restoration only visually.

Use quantitative metrics.

---

# 29. DATA LEAKAGE TEST

Create an automated test that checks:

- no source clean image occurs in multiple splits
- no source image variant crosses splits
- test images are never used for training
- validation images are never used for training
- multiple damaged variants from the same clean source stay together

This test must pass before training.

---

# 30. COMMAND-LINE INTERFACE

Create simple commands such as:

Dataset split:

python split_dataset.py 
    --input ./intel 
    --output ./dataset 
    --seed 42

Generate one artifact:

python generate_damage.py 
    --input ./dataset/train 
    --output ./restoration_dataset 
    --damage-types scratches 
    --variants 1

Generate mixed damage:

python generate_damage.py 
    --input ./dataset/train 
    --output ./restoration_dataset 
    --damage-types mixed 
    --variants 3

Train:

python train.py 
    --data ./restoration_dataset 
    --epochs 50 
    --batch-size 8 
    --img-size 256 
    --lr 1e-4

Evaluate:

python evaluate.py 
    --data ./restoration_dataset 
    --checkpoint ./checkpoints/best.pth

Inference:

python infer.py 
    --image ./damaged.png 
    --checkpoint ./checkpoints/best.pth 
    --output ./restored.png

---

# 31. CONFIGURATION

Create:

config.yaml

Example:

dataset:
  input_root:
  output_root:
  classes:
    - buildings
    - street
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42

damage:
  types:
    - dirt
    - dots
    - scratches
    - smut
    - spot
    - sprinkle
    - stain
  variants_per_image: 3
  severity:
  probabilities:
  seed:

restoration:
  image_size: 256
  input_channels: 3
  output_channels: 3
  use_mask: false

training:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0001
  loss:
  checkpoint_dir:

---

# 32. PROJECT STRUCTURE

Keep the project modular.

Recommended:

project/
│
├── config.yaml
├── requirements.txt
├── README.md
│
├── external/
│   ├── FilmDamageSimulator/
│   └── DustScratchRemoval/
│
├── data/
│   ├── dataset.py
│   ├── split.py
│   └── augmentation.py
│
├── damage/
│   ├── simulator_adapter.py
│   ├── artifact_selector.py
│   ├── generator.py
│   └── visualization.py
│
├── restoration/
│   ├── model.py
│   ├── losses.py
│   ├── dataset.py
│   └── training.py
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluate.py
│   └── damage_type_eval.py
│
├── inference/
│   └── infer.py
│
├── scripts/
│   ├── split_dataset.py
│   ├── generate_damage.py
│   └── visualize_samples.py
│
├── checkpoints/
├── debug_samples/
└── results/

Do not create one giant Python file.

Keep both original repositories identifiable.

Prefer:

- git submodules
- local cloned repositories
- clearly separated adapters

rather than copying the entire repositories into the project.

---

# 33. DO NOT AUTOMATICALLY CHANGE THE ORIGINAL REPOSITORIES

Treat both repositories as reference/external components.

Do not make destructive modifications to:

FilmDamageSimulator

or

DustScratchRemoval

unless there is a clearly documented reason.

Instead, create adapters/wrappers in this project.

Example:

damage/simulator_adapter.py

and:

restoration/model.py

This keeps the academic project reproducible and makes it clear which work originates from the existing repositories and which work is our adaptation.

---

# 34. PHASED IMPLEMENTATION

## PHASE 1 — INSPECT EVERYTHING

Do NOT modify anything yet.

Inspect:

FILM DAMAGE REPOSITORY:
- README
- TECHNICAL
- damage_generator
- synthetic
- relevant mask generation
- damage application
- command-line interfaces

RESTORATION REPOSITORY:
- README
- notebooks
- model architecture
- training
- losses
- transfer learning
- inference
- preprocessing
- checkpoints

INTEL DATASET:
- directory structure
- class names
- image counts
- dimensions
- formats

Report:

1. repository structures
2. relevant files
3. relevant functions/classes
4. how the two repositories can be connected
5. exact artifact mappings
6. required dependency versions
7. compatibility problems
8. proposed adapter design

DO NOT write implementation code yet.

---

## PHASE 2 — DATASET SPLITTING

Implement:

- Intel loader
- buildings/street filtering
- 70/15/15 split
- reproducible seed
- leakage checking

Output class counts.

Do NOT generate damage yet.

---

## PHASE 3 — DAMAGE INTEGRATION

Connect FilmDamageSimulator.

Generate ONLY:

- dirt
- dots
- scratches
- smut
- spot
- sprinkle
- stain

Generate one example of each.

Produce:

CLEAN | MASK | DAMAGED

Verify the artifact visually and through metadata.

Do not generate the complete dataset yet.

---

## PHASE 4 — SYNTHETIC DATASET

Generate:

clean
damaged
mask
metadata

for a small subset first.

Test:

- pairing
- masks
- artifact selection
- severity
- metadata
- deterministic generation
- split isolation

Only then enable full dataset generation.

---

## PHASE 5 — RESTORATION ADAPTATION

Integrate DustScratchRemoval.

Start with:

DAMAGED RGB → CLEAN RGB

Do not use colorization.

Reproduce/adapt the U-Net and relevant transfer-learning/loss components.

Create a small sanity-check training run.

Verify:

- forward pass
- loss
- backpropagation
- checkpoint saving
- validation
- inference

---

## PHASE 6 — FULL TRAINING

Run the baseline restoration experiment.

Then evaluate the different loss configurations based on the original DustScratchRemoval approach.

Do not start all expensive experiments simultaneously.

Save experiment configurations and results.

---

## PHASE 7 — EVALUATION

Calculate:

- PSNR
- SSIM
- MAE
- MSE

for:

damaged vs clean

and:

restored vs clean

Evaluate:

- overall
- buildings
- street
- each artifact type
- mixed damage

---

## PHASE 8 — QUALITATIVE ANALYSIS

Generate:

CLEAN | DAMAGED | MASK | RESTORED

for representative test images.

Create examples for different:

- classes
- artifact types
- severities

---

## PHASE 9 — FINAL PROJECT CLEANUP

Produce:

- README
- setup instructions
- dataset-generation instructions
- training instructions
- evaluation instructions
- inference instructions
- experiment configuration
- final results tables
- qualitative examples

Clearly document:

FILM DAMAGE SIMULATOR
= synthetic degradation

DUST SCRATCH REMOVAL
= restoration model

INTEL IMAGE CLASSIFICATION
= clean source images

NO COLORIZATION
= explicitly outside project scope.

---

# 35. IMPORTANT CODING-AGENT RULES

Do not assume an API exists.

Inspect actual source code before using a function.

Do not fabricate filenames.

Do not fabricate artifact directories.

Do not fabricate CLI flags.

Do not replace repository functionality with random OpenCV damage generation unless absolutely necessary.

Do not overwrite the original Intel dataset.

Do not overwrite clean images.

Do not silently skip failed images.

Log failures.

Use progress bars for large operations.

Test on a tiny subset before expensive operations.

Do not launch a long training run automatically.

Never mix train/validation/test samples.

Do not evaluate on training data and report it as test performance.

Do not claim a model is "better" without quantitative comparison.

Do not introduce colorization.

Do not add GAN training.

Do not add unnecessary architectures unless explicitly needed.

When adapting code from the repositories, keep the original source clearly identifiable and document the modifications.

The final project should be reproducible and appropriate for an academic image-restoration project.

START WITH PHASE 1 ONLY.

Do not implement the entire project in the first response.