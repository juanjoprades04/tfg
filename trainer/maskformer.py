from typing import Any, Dict
from transformers import MaskFormerConfig, Mask2FormerForUniversalSegmentation, AutoImageProcessor
import mlflow
import torch
import numpy as np
import evaluate
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
from trainer.transformers_base import TransformerBase
import tempfile
import os


class MaskFormerTrainer(TransformerBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = self.config["model_name"][0]

    def build_model(self, model_name=None) -> Any:
        if model_name is None:
            self.model_name = self.config["model_name"][0]
        else:
            self.model_name = model_name

        self.num_labels = self.config["num_classes"]

        config_mask = MaskFormerConfig.from_pretrained(self.model_name)
        config_mask.id2label = self.config["labels"]
        config_mask.label2id = {v: k for k, v in config_mask.id2label.items()}
        config_mask.num_labels = self.num_labels
        config_mask.ignore_value = 0

        config_mask.backbone_config.id2label = self.config["labels"]
        config_mask.backbone_config.label2id = {v: k for k, v in config_mask.id2label.items()}
        config_mask.backbone_config.num_labels = self.num_labels

        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_name,
            ignore_mismatched_sizes=True,
            config=config_mask
        )

        model = model.to(self.device)
        return model

    def prepare_data(self, train_data: Any, val_data: Any) -> None:
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            do_reduce_labels=False,
            reduce_labels=False,
            )

        def preprocess(batch):
            image = Image.open(batch["image_path"]).convert("RGB")
            mask = np.array(Image.open(batch["label_path"]), dtype=np.int64)
            valid_values = sorted([v for v in np.unique(mask) if v != 0])

            remapped_mask = np.zeros_like(mask, dtype=np.int64)

            for new_id, old_id in enumerate(valid_values, start=1):
                remapped_mask[mask == old_id] = new_id

            mask = remapped_mask

            mask[mask >= self.config["num_classes"]] = 0
            image_np = np.array(image)

            if batch.get("__is_train__", False):
                augmented = self.augment(image=image_np, mask=mask)
                image_np = augmented["image"]
                mask = augmented["mask"]

            inputs = self.processor(
                images=image_np,
                segmentation_maps=mask,
                return_tensors="pt"
            )

            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "mask_labels": inputs["mask_labels"][0],
                "class_labels": inputs["class_labels"][0],
            }

        train_data = train_data.add_column("__is_train__", [True] * len(train_data))
        val_data = val_data.add_column("__is_train__", [False] * len(val_data))

        self.train_dataset = train_data.map(preprocess)
        self.val_dataset = val_data.map(preprocess)

        self.train_dataset.set_format(type="torch", columns=["pixel_values", "mask_labels", "class_labels"])
        self.val_dataset.set_format(type="torch", columns=["pixel_values", "mask_labels", "class_labels"])

    def predict(self, inputs: Any) -> Any:
        pass

    def load_checkpoint(self, path: str) -> None:
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(path)

    def postprocess(self, eval_pred):
        raw_preds, raw_labels = eval_pred
        class_q_logits = torch.tensor(raw_preds[0])
        mask_q_logits = torch.tensor(raw_preds[1])

        mask_labels = torch.tensor(raw_labels[0])

        class OutputWrapper:
            def __init__(self, cq, mq):
                self.class_queries_logits = cq
                self.masks_queries_logits = mq

        model_outputs = OutputWrapper(class_q_logits, mask_q_logits)
        target_sizes = [lbl.shape[-2:] for lbl in mask_labels]

        results = self.processor.post_process_semantic_segmentation(
            model_outputs,
            target_sizes=target_sizes
        )
        return results

    def compute_metrics(self, eval_pred):
        self.metric = evaluate.load("mean_iou")

        with torch.no_grad():
            raw_preds, raw_labels = eval_pred
            class_q_logits = torch.tensor(raw_preds[0])
            mask_q_logits = torch.tensor(raw_preds[1])

            mask_labels = torch.tensor(raw_labels[0])

            class OutputWrapper:
                def __init__(self, cq, mq):
                    self.class_queries_logits = cq
                    self.masks_queries_logits = mq

            model_outputs = OutputWrapper(class_q_logits, mask_q_logits)
            target_sizes = [lbl.shape[-2:] for lbl in mask_labels]

            results = self.processor.post_process_semantic_segmentation(
                model_outputs,
                target_sizes=target_sizes
            )

            pred_labels = results
            labels_tensor = mask_labels.argmax(dim=1)

            metrics = self.metric.compute(
                predictions=pred_labels,
                references=labels_tensor,
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

    def plot_val_result(self, pixel_values, preds, labels):
        image = pixel_values[0].cpu()
        pred_mask = preds[0].cpu()
        gt_mask = labels[0].cpu() if labels is not None else None

        image = (image - image.min()) / (image.max() - image.min())

        if gt_mask is not None and gt_mask.ndim == 3:
            gt_mask = gt_mask.argmax(dim=0)

        fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(12, 4))

        axes[0].imshow(image.permute(1, 2, 0))
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="tab20")
        axes[1].set_title("Prediction (Best Model)")
        axes[1].axis("off")

        if gt_mask is not None:
            axes[2].imshow(gt_mask, cmap="tab20")
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "best_model_prediction.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close(fig)

            mlflow.log_artifact(path, artifact_path="best_model_segmentation")