# MAGNA — Segmentation training + Optuna tuning

Small library to train semantic / instance segmentation models and to optimize their hyperparameters (including augmentation parameters) using Optuna.  
Supported architectures (and "bundles" = model + backbone) are implemented as trainer classes under `trainer/`.

Contents
- Overview
- Supported architectures & bundles
- How it works (high-level)
- Quickstart (API + example)
- Dataset format
- Hyperparameter optimization (Optuna)
- Logging & artifacts (MLflow + Triton config)
- Extending the library
- Files & key symbols (links)

Overview
--------
MAGNA provides:
- Trainer classes that wrap model building, dataset preprocessing, training and metric computation.
- Two base trainer families:
  - Transformer-based HF models (SegFormer, Mask2Former) via [`trainer/transformers_base.py`](trainer/transformers_base.py) (class [`TransformerBase`](trainer/transformers_base.py))
  - PyTorch-native segmentation nets (DeepLabV3, FCN) via [`trainer/torch_base.py`](trainer/torch_base.py) (class [`TorchBase`](trainer/torch_base.py))
- Optuna-driven hyperparameter search implemented in the base classes.
- MLflow logging and TorchScript model artifact logging with a generated Triton `config.pbtxt`.

Supported architectures & bundles
---------------------------------
- SegFormer (HuggingFace SegFormer)
  - Bundles:
    - nvidia/segformer-b0-finetuned-ade-512-512
    - nvidia/segformer-b1-finetuned-ade-512-512
    - nvidia/segformer-b2-finetuned-ade-512-512
    - nvidia/segformer-b3-finetuned-ade-512-512
    - nvidia/segformer-b4-finetuned-ade-512-512
  - Trainer: [`SegFormerTrainer`](trainer/segformer.py)

- Mask2Former (Mask2Former)
  - Bundles:
    - facebook/mask2former-swin-tiny-ade-semantic
    - facebook/mask2former-swin-small-ade-semantic
    - facebook/mask2former-swin-base-ade-semantic
    - facebook/mask2former-swin-large-ade-semantic
  - Trainer: [`MaskFormerTrainer`](trainer/maskformer.py)

- DeepLabV3 (PyTorch torchvision)
  - Bundles:
    - deeplabv3_resnet50
    - deeplabv3_resnet101
  - Trainer: [`DeepLabv3`](trainer/deeplabv3.py)

- FCN (PyTorch torchvision)
  - Bundles:
    - fcn_resnet50
    - fcn_resnet101
  - Trainer: [`FCN`](trainer/fcn.py)
  
- Detectron2 (Detectron implementation)
  - Bundles:
    - mask_rcnn_r50_c4_1x
    - mask_rcnn_r50_c4_3x
    - mask_rcnn_r50_dc5_1x
    - mask_rcnn_r50_dc5_3x
    - mask_rcnn_r50_fpn_1x
    - mask_rcnn_r50_fpn_1x_giou
    - mask_rcnn_r50_fpn_3x
    - mask_rcnn_r101_c4_3x
    - mask_rcnn_r101_dc5_3x
    - mask_rcnn_r101_fpn_3x
    - mask_rcnn_x101_32x8d_fpn_3x
    - cascade_mask_rcnn_r50_fpn_1x
    - cascade_mask_rcnn_r50_fpn_3x
    - misc_mask_rcnn_r50_fpn_3x_gn
    - misc_scratch_mask_rcnn_r50_fpn_9x_gn
  - Trainer: [`Detectron`](trainer/detectron2_base.py)


How it works (high-level)
-------------------------
1. Prepare a HF Dataset (or any dataset compatible with the preprocessing) that contains columns `image_path` and `label_path`. See [`test_trainer.get_dummy_data`](test_trainer.py#get_dummy_data).
2. Create a trainer instance for the desired architecture (one of the classes above).
3. Call `.prepare(train_data, val_data)` to map/convert dataset entries into tensors suitable for training (augmentations applied during mapping).
4. Call `.initialize()` to build the model and setup training (HuggingFace `Trainer` or internal training loop).
5. Either:
   - Run `.run(train_data, val_data)` to perform a single training run, or
   - Run `.optimize(train_data, val_data, n_trials=...)` to run Optuna hyperparameter optimization (calls `optuna` study and trains a fresh model per trial).

Quickstart — example
--------------------
See the runnable example at [`test_trainer.py`](test_trainer.py). Minimal usage pattern:

```python
# Example usage (see test_trainer.py for full example)
from trainer.fcn import FCN
from datasets import Dataset
# prepare HF Dataset with image_path/label_path...
trainer = FCN(config)
trainer.prepare(train_dataset, val_dataset)
trainer.initialize()
trainer.fit(run_name="train_run")           # single training run -> MLflow logged
# OR run hyperparameter search:
trainer.optimize(train_dataset, val_dataset, n_trials=10)
```

Files / key classes & methods
-----------------------------
- Core abstract base: [`BaseTrainer`](trainer.py) — high-level orchestration (methods: [`BaseTrainer.prepare`](trainer.py), [`BaseTrainer.initialize`](trainer.py), [`BaseTrainer.run`](trainer.py), [`BaseTrainer.optimize`](trainer.py))
- Transformer HF base: [`TransformerBase`](trainer/transformers_base.py) — implements HF `Trainer` wiring, Optuna helpers and HF-specific preprocess (see `build_augmentations`, `optuna_hp_space`, `optimize_architecture`).
- PyTorch base: [`TorchBase`](trainer/torch_base.py) — similar to TransformerBase but for pure PyTorch models; implements augmentation helpers and Optuna flow.
- Trainers:
  - [`SegFormerTrainer`](trainer/segformer.py) — SegFormer HF wrapper
  - [`MaskFormerTrainer`](trainer/maskformer.py) — Mask2Former HF wrapper
  - [`DeepLabv3`](trainer/deeplabv3.py) — DeepLabv3 PyTorch wrapper
  - [`FCN`](trainer/fcn.py) — FCN PyTorch wrapper
  - [`Detectron`](trainer/detectron2_base.py) — Detectron2 wrapper
- Example runner: [`test_trainer.py`](test_trainer.py) — shows `get_dummy_data()` and a sample config


Config reference (fields & semantics)
------------------------------------

Example configs in the repo demonstrate two common usages: optimization (ranges) and fixed training.

Key config fields:
- model: high-level architecture string used by caller to pick trainer class (e.g. "Segformer", "Mask2Former", "DeepLabv3", "FCN"). The code picks the trainer class externally (see test_trainer.py).
- model_name (list of str): the exact bundle identifier / backbone to load. Examples:
  - HuggingFace: nvidia/segformer-b2-finetuned-ade-512-512 or facebook/mask2former-swin-base-ade-semantic
  - torchvision: fcn_resnet50, deeplabv3_resnet101
- experiment: name of the experiment run that will be shown in MLflow.
- training_name (str): optional string which will be the name of the training.
- mlflow_ip (str): IP Address of MLFlow in order to log correctly.
- num_classes (int): number of semantic classes including background (background should be encoded as index 0).
- labels (dict): relation between categorical and nominal classes to train. There should be always the class background as: 0: "background" 
- train_split (float): Train validation split,
- cvat_task_ids (list of ints): Task numbers to be downloaded from CVAT,
- learning_rate: dict with keys:
  - value: list with either 1 value (fixed training) or 2 values [min, max] (Optuna range)
- batch_size: dict similar to learning_rate (value can be list of ints)
- epochs: dict similar to learning_rate
- augmentations: dict mapping augmentation parameter names to dicts:
  - e.g. "aug_hflip": {"value": [0, 0.5], "steps": 0.1}
  - value can be a single element list (fixed) or two-element list [min,max] (optimize)
  - supported augmentation keys: aug_hflip, aug_vflip, aug_scale, aug_brightness, aug_saturation, aug_gaussianblur, aug_motionblur, aug_gaussiannoise, aug_isonoise
- device: "cuda" or "cpu"

Intended semantics (optimization vs fixed training)
- Optimization config (example): for each hyperparam you want to search, set "value": [min, max]. The base classes implement optuna_hp_space(trial) to sample from those ranges (continuous uniform for floats, discrete choices for integers). Augmentation parameters are sampled and applied via build_augmentations for each trial.
- Training config (example): set "value" lists to single fixed values (e.g., "learning_rate": {"value": [3e-3]}). The trainer uses those single values directly for TrainingArguments.

Special notes
- Background label index 0 is ignored in IoU computation (ignore_index=0).
- When a numeric field contains two values, it's treated as a range for Optuna; when it has one the value is fixed.
- "model" is for external selection of which trainer class to instantiate; "model_name" determines the actual backbone / pretrained bundle passed to the model constructor.
- The test_trainer.py example demonstrates both patterns:
  - optimization config: ranges provided for learning_rate, augmentations, etc.
  - training config: fixed single values for learning_rate, batch_size, epochs, augmentations.
- main.py file includes an example of API
- Dockerfile is included in order to deploy it in a VM.


Dataset format
--------------
- Each sample must include:
  - `image_path` — path to the RGB image
  - `label_path` — path to the segmentation mask (integer mask)
- The example helper is [`get_dummy_data`](test_trainer.py#get_dummy_data) which builds a HF Dataset storing only paths (no eager image loading).
- During preprocessing the trainers:
  - Load images with PIL
  - Convert masks to integer arrays
  - Fix labels outside range by setting them to background (index 0)
  - Apply Albumentations augmentation pipeline (train-only)
  - Normalize images with ImageNet mean/std when using torchvision nets or AutoImageProcessor for HF models
- CVAT_Dataset utils can be used to extract CVAT tasks.
```python
    magna_utils.cvat_dataset.download_cvat_tasks(config.cvat_task_ids, output_dir="/home/ubuntu/tmp")
    magna_utils.cvat_dataset.extract_annotations_to_masks("/home/ubuntu/tmp", "/home/ubuntu/tmp/labels")
    train_data, val_data = get_dummy_data("/home/ubuntu/tmp")
```

Augmentations
-------------
- Augmentations are built via `Albumentations` in the base classes:
  - [`TransformerBase.build_augmentations`](trainer/transformers_base.py)
  - [`TorchBase.build_augmentations`](trainer/torch_base.py)
- Augmentation hyperparameters are supported in the config under the `augmentations` key and are sampled during Optuna trials.
- Detectron augmentations are custom made and can be found in ['detectron2_base.py'](trainer/detectron2_base.py)

Metrics & evaluation
--------------------
- Mean IoU is computed via the `evaluate` package (`mean_iou`).
- Each trainer implements `compute_metrics(...)` (see implementations in:
  - [`trainer/segformer.py`](trainer/segformer.py) — `SegFormerTrainer.compute_metrics`
  - [`trainer/maskformer.py`](trainer/maskformer.py) — `MaskFormerTrainer.compute_metrics`
  - [`trainer/deeplabv3.py`](trainer/deeplabv3.py) — `DeepLabv3.compute_metrics`
  - [`trainer/fcn.py`](trainer/fcn.py) — `FCN.compute_metrics`
  - [`trainer/detectron2_base.py`](trainer/detectron2_base.py) — `DetectronBase.compute_metrics`
)
- Metrics logged by MLflow callbacks: `LogMetricsCallback`, `BestModelCallback` (see [`trainer/transformers_base.py`](trainer/transformers_base.py) and [`trainer/torch_base.py`](trainer/torch_base.py)).

Hyperparameter optimization (Optuna)
-----------------------------------
- Optuna search space constructed from config ranges. The base classes contain:
  - `optuna_hp_space(trial)` and `optimize_architecture(n_trials)` implementations in [`TransformerBase`](trainer/transformers_base.py) and [`TorchBase`](trainer/torch_base.py).
- During each trial:
  - augmentation params are sampled and applied
  - a fresh model is built and trained for the trial
  - evaluation metric `eval_mean_iou` is returned to Optuna
- After optimization basic visualizations are written to `temp_models/optimization_history.html` and `temp_models/param_importances.html`.

Logging & artifacts (MLflow)
---------------------------
- MLflow is configured inside [`BaseTrainer.__init__`](trainer.py):
  - `mlflow.set_tracking_uri(config["mlflow_ip"])`
  - `mlflow.set_experiment(config.get("experiment", "experiment"))`
- On training end:
  - Models are traced with TorchScript and logged via MLflow (`mlflow.pytorch.log_model`) from callbacks (`LogModelCallback` in both base files).
  - A minimal Triton `config.pbtxt` is generated and logged to `model/data/config.pbtxt` as an artifact.
  - Models are also optimized and exported to:
    - ONNX
    - TensorRT

Extending the library
---------------------
- Add a new architecture:
  1. Derive a class from [`TransformerBase`](trainer/transformers_base.py) (HF-based) or [`TorchBase`](trainer/torch_base.py) (PyTorch-native).
  2. Implement `build_model()`, `prepare_data()`, `compute_metrics()` and `load_checkpoint()` as required.
  3. Add unit test / example usage similar to [`test_trainer.py`](test_trainer.py).

Running the example
-------------------
- Ensure you have required packages (transformers, datasets, albumentations, optuna, evaluate, mlflow, torch, torchvision).
- Start MLflow server (if desired) or adjust tracking URI in [`trainer.py`](trainer.py).
- Run the example in repository root:
```sh
python test_trainer.py
```
This will use the config inside `test_trainer.py` and demonstrate training or Optuna optimization.

Notes & gotchas
---------------
- The code assumes GPU availability when device is `cuda`. Adjust `device` in configs for CPU runs.
- HF models use `AutoImageProcessor` to produce `pixel_values` and `labels`. PyTorch wrappers normalize manually.
- Background label is encoded as index `0` and is ignored in IoU computation (`ignore_index=0`).
- The Optuna optimization may run into CUDA OOM; the base classes catch OOM and assign a bad score to that trial.

Files & entrypoints (quick links)
-------------------------------
- [trainer.py](trainer.py) — [`BaseTrainer`](trainer.py)
- [trainer/transformers_base.py](trainer/transformers_base.py) — [`TransformerBase`](trainer/transformers_base.py)
- [trainer/torch_base.py](trainer/torch_base.py) — [`TorchBase`](trainer/torch_base.py)
- [trainer/segformer.py](trainer/segformer.py) — [`SegFormerTrainer`](trainer/segformer.py)
- [trainer/maskformer.py](trainer/maskformer.py) — [`MaskFormerTrainer`](trainer/maskformer.py)
- [trainer/deeplabv3.py](trainer/deeplabv3.py) — [`DeepLabv3`](trainer/deeplabv3.py)
- [trainer/fcn.py](trainer/fcn.py) — [`FCN`](trainer/fcn.py)
- [trainer/fcn.py](trainer/detectron_base.py) — [`Detectron`](trainer/fcn.py)
- [test_trainer.py](test_trainer.py) — example runner, [`get_dummy_data`](test_trainer.py)
