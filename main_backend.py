import os
from datasets import Dataset
import magna_utils.cvat_dataset
from magna_utils.cvat_utils import get_task_labels_metadata

from trainer.maskformer import MaskFormerTrainer
from trainer.segformer import SegFormerTrainer
from trainer.deeplabv3 import DeepLabv3
from trainer.fcn import FCN

try:
    from trainer.detectron2_base import DetectronBase
except ImportError:
    DetectronBase = None


def get_dummy_data(root_dir: str, train_split=0.8):
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ])

    samples = []
    for f in image_files:
        name, ext = os.path.splitext(f)

        if ext.lower() != ".png":
            label_name = name + ".png"
        else:
            label_name = f

        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            continue

        samples.append({
            "image_path": os.path.join(image_dir, f),
            "label_path": label_path,
        })

    if not samples:
        raise ValueError("No se han encontrado pares image_path / label_path válidos")

    dataset = Dataset.from_list(samples)

    split_idx = int(len(dataset) * train_split)
    split_idx = max(1, min(split_idx, len(dataset) - 1)) if len(dataset) > 1 else len(dataset)

    if len(dataset) == 1:
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset = dataset.select(range(split_idx))
        val_dataset = dataset.select(range(split_idx, len(dataset)))

    return train_dataset, val_dataset


def get_trainer(config):
    model = config.get("model")

    if model == "SegFormer":
        return SegFormerTrainer(config)
    elif model == "MaskFormer":
        return MaskFormerTrainer(config)
    elif model == "FCN":
        return FCN(config)
    elif model == "DeepLabV3":
        return DeepLabv3(config)
    elif model == "Detectron":
        if DetectronBase is None:
            raise ImportError("Detectron2 no está instalado")
        return DetectronBase(config)
    else:
        raise ValueError(f"Modelo no soportado: {model}")


def detect_task_metadata(task_ids):
    shape_labels = get_task_labels_metadata(task_ids)

    if not shape_labels:
        raise ValueError("No se han podido detectar labels de segmentación en las tasks indicadas")

    labels = {0: "background"}
    for idx, label_name in enumerate(shape_labels, start=1):
        labels[idx] = label_name

    metadata = {
        "labels": labels,
        "num_classes": len(labels),
    }
    return metadata


def prepare_data(config):
    task_ids = config.get("cvat_task_ids", [])
    train_split = config.get("train_split", 0.8)

    if not task_ids:
        raise ValueError("Debes proporcionar cvat_task_ids")

    if "labels" not in config or "num_classes" not in config:
        metadata = detect_task_metadata(task_ids)
        config["labels"] = metadata["labels"]
        config["num_classes"] = metadata["num_classes"]

    training_name = config.get("training_name", "run")
    output_dir = f"./tmp/{training_name}"
    labels_dir = os.path.join(output_dir, "labels")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Descargando datos CVAT en {output_dir}...")

    magna_utils.cvat_dataset.download_cvat_tasks(
       task_ids,
       output_dir=output_dir
    )

    magna_utils.cvat_dataset.extract_annotations_to_masks(
        temp_dir=output_dir,
        output_dir=labels_dir,
        task_ids=task_ids,
    )

    train_data, val_data = get_dummy_data(output_dir, train_split)

    return train_data, val_data


def run_training_pipeline(config):
    print("=== INICIANDO TRAINING ===")

    train_data, val_data = prepare_data(config)
    trainer = get_trainer(config)

    trainer.run(
        train_data=train_data,
        val_data=val_data
    )

    print("=== TRAINING FINALIZADO ===")


def run_optimization_pipeline(config):
    print("=== INICIANDO OPTUNA ===")

    train_data, val_data = prepare_data(config)
    trainer = get_trainer(config)

    n_trials = config.get("n_trials", 10)

    trainer.optimize(
        train_data=train_data,
        val_data=val_data,
        n_trials=n_trials
    )

    print("=== OPTIMIZACIÓN FINALIZADA ===")