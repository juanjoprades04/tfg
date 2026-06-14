from typing import Any, Dict

import torch
import numpy as np
import evaluate
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101

from trainer.torch_base import TorchBase


class FCNWrapper(nn.Module):
    def __init__(self, config, model_name, device="cpu"):
        super().__init__()

        self.device = torch.device(
            "cuda" if str(device) == "cuda" and torch.cuda.is_available() else "cpu"
        )

        if model_name == "fcn_resnet50":
            self.model = fcn_resnet50(weights="DEFAULT")
        elif model_name == "fcn_resnet101":
            self.model = fcn_resnet101(weights="DEFAULT")
        else:
            raise ValueError(f"Model {model_name} declared does not exist")

        # Replace classifier head
        in_channels_main = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(
            in_channels_main,
            config["num_classes"],
            kernel_size=1
        )

        if self.model.aux_classifier is not None:
            in_channels_aux = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(
                in_channels_aux,
                config["num_classes"],
                kernel_size=1
            )

        self.num_classes = config["num_classes"]

        self.to(self.device)

    def forward(self, pixel_values=None, x=None, labels=None):
        if pixel_values is not None:
            inputs = pixel_values
        elif x is not None:
            inputs = x
        else:
            raise ValueError("No input tensor provided")

        inputs = inputs.to(self.device)

        outputs = self.model(inputs)["out"]

        loss = None
        if labels is not None:
            labels = labels.to(outputs.device)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)

        return {"logits": outputs, "loss": loss}


class FCN(TorchBase):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_labels = config["num_classes"]
        self.model_name = None

    def build_model(self, model_name=None) -> nn.Module:
        if model_name is None:
            self.model_name = self.config["model_name"][0]
        else:
            self.model_name = model_name

        model = FCNWrapper(
            config=self.config,
            model_name=self.model_name,
            device=self.device
        )

        print(f"[FCN] Using device: {model.device}")
        return model

    def prepare_data(self, train_data: Any, val_data: Any) -> None:

        target_size = tuple(self.config.get("input_size", [512, 512]))

        def preprocess(batch):
            image_pil = Image.open(batch["image_path"]).convert("RGB")
            mask_pil = Image.open(batch["label_path"])

            # Resize
            image_pil = image_pil.resize(target_size, Image.BILINEAR)
            mask_pil = mask_pil.resize(target_size, Image.NEAREST)

            image = np.array(image_pil)
            mask = np.array(mask_pil, dtype=np.int64)
            valid_values = sorted([v for v in np.unique(mask) if v != 0])

            remapped_mask = np.zeros_like(mask, dtype=np.int64)

            for new_id, old_id in enumerate(valid_values, start=1):
                remapped_mask[mask == old_id] = new_id

            mask = remapped_mask

            mask[mask >= self.config["num_classes"]] = 0

            if batch.get("__is_train__", False):
                augmented = self.augment(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

            mask_tensor = torch.tensor(mask, dtype=torch.long)

            return {
                "pixel_values": image_tensor,
                "labels": mask_tensor
            }

        train_data = train_data.add_column("__is_train__", [True] * len(train_data))
        val_data = val_data.add_column("__is_train__", [False] * len(val_data))

        self.train_dataset = train_data.map(preprocess)
        self.val_dataset = val_data.map(preprocess)

        self.train_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
        self.val_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    def predict(self, inputs: Any) -> Any:
        pass

    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError("load_checkpoint no implementado para FCN")

    def compute_metrics(self, eval_pred):

        self.metric = evaluate.load("mean_iou")

        with torch.no_grad():
            logits, labels = eval_pred

            logits_tensor = torch.from_numpy(logits)
            labels_tensor = torch.from_numpy(labels)

            pred_labels = torch.argmax(logits_tensor, dim=1).numpy()
            labels_np = labels_tensor.numpy()

            metrics = self.metric.compute(
                predictions=pred_labels,
                references=labels_np,
                num_labels=self.num_labels,
                ignore_index=255,
                reduce_labels=False
            )

            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()

            per_class_iou = np.array(metrics.get("per_category_iou")[1:], dtype=float)
            valid_iou = per_class_iou[~np.isnan(per_class_iou)]

            if len(valid_iou) == 0:
                metrics["mean_iou"] = 0.0
            else:
                metrics["mean_iou"] = float(valid_iou.mean())

            return metrics
