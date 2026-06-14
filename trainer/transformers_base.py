from typing import Any, Dict
from transformers import AutoImageProcessor, TrainerCallback, TrainingArguments, Trainer
from trainer import BaseTrainer
import mlflow
import torch
import numpy as np
import evaluate
import albumentations as A
import optuna
import matplotlib.pyplot as plt
from abc import abstractmethod
import gc
import time
import os

try:
    import tensorrt as trt
except ImportError:
    trt = None


class TransformerBase(BaseTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.label_ids = None
        self.train_dataset = None
        self.val_dataset = None
        self.training_args = None
        self.hf_trainer = None
        self.metric = None
        self.processor = None
        self.augment = self.build_augmentations()

    def build_augmentations(self, aug_params=None):
        aug_config = self.config.get("augmentations", {})

        def get_value(x):
            if isinstance(x, dict) and "value" in x:
                return x["value"][0]
            return 0.0

        aug_params = aug_params or {k: get_value(v) for k, v in aug_config.items()}

        return A.Compose(
            [
                A.HorizontalFlip(p=aug_params.get("aug_hflip", 0.0)),
                A.VerticalFlip(p=aug_params.get("aug_vflip", 0.0)),
                A.RandomScale(scale_limit=0.2, p=aug_params.get("aug_scale", 0.0)),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=aug_params.get("aug_brightness", 0.0),
                ),
                A.HueSaturationValue(
                    hue_shift_limit=0,
                    sat_shift_limit=20,
                    val_shift_limit=0,
                    p=aug_params.get("aug_saturation", 0.0),
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=aug_params.get("aug_gaussianblur", 0.0)),
                A.MotionBlur(blur_limit=7, p=aug_params.get("aug_motionblur", 0.0)),
                A.GaussNoise(var_limit=(10.0, 50.0), p=aug_params.get("aug_gaussiannoise", 0.0)),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=aug_params.get("aug_isonoise", 0.0)),
            ]
        )

    @abstractmethod
    def build_model(self) -> Any:
        pass

    @abstractmethod
    def prepare_data(self, train_data: Any, val_data: Any) -> None:
        pass

    def setup_training(self) -> None:
        epochs = self.config["epochs"]["value"][0]
        lr = self.config["learning_rate"]["value"][0]
        batch_size = self.config["batch_size"]["value"][0]
        dir_model = f"models/{self.model_name}"

        training_args = TrainingArguments(
            output_dir=dir_model,
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_total_limit=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            eval_accumulation_steps=5,
            load_best_model_at_end=True,
            use_cpu=(self.device == "cpu"),
        )

        self.hf_trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                LogMetricsCallback,
                BestModelCallback(
                    metric_name="eval_mean_iou",
                    greater_is_better=True,
                )
            ],
        )

    def fit(self, run_name: str) -> Dict[str, Any]:
        with mlflow.start_run(run_name=run_name):
            self.hf_trainer.train()
            self.log_train_aug_params()
            mlflow.log_param("model_name", self.model_name)
            print("Parameters logged successfully")

        mlflow.end_run()

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        pass

    def optuna_hp_space(self, trial):
        hp = {}

        lr_range = self.config.get("learning_rate", {}).get("value", [1e-4])
        hp["learning_rate"] = (
            trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)
            if len(lr_range) > 1 else lr_range[0]
        )

        batch_range = self.config.get("batch_size", {}).get("value", [8])
        hp["per_device_train_batch_size"] = (
            trial.suggest_int("per_device_train_batch_size", batch_range[0], batch_range[1])
            if len(batch_range) > 1 else batch_range[0]
        )

        epochs_range = self.config.get("epochs", {}).get("value", [20])
        hp["num_train_epochs"] = (
            trial.suggest_int("num_train_epochs", epochs_range[0], epochs_range[1])
            if len(epochs_range) > 1 else epochs_range[0]
        )

        model_range = self.config.get("model_name", [])
        if len(model_range) == 0:
            raise ValueError("Error in model_name")
        hp["model_name"] = (
            trial.suggest_categorical("model_name", model_range)
            if len(model_range) > 1 else model_range[0]
        )

        aug_config = self.config.get("augmentations", {})
        aug_params = {}
        for key, val in aug_config.items():
            val_range = val.get("value", [0.0])
            aug_params[key] = (
                trial.suggest_float(key, val_range[0], val_range[1])
                if len(val_range) > 1 else val_range[0]
            )

        return hp, aug_params

    def optimize_architecture(self, name, n_trials=100):
        def objective(trial):
            run_name = f"{name}_{trial.number}"
            with mlflow.start_run(run_name=run_name):
                try:
                    if torch.cuda.is_available() and self.device == "cuda":
                        mlflow.log_param("gpu_name", torch.cuda.get_device_name(torch.cuda.current_device()))
                    else:
                        mlflow.log_param("gpu_name", "cpu")

                    hp, aug_params = self.optuna_hp_space(trial)
                    self.augment = self.build_augmentations(aug_params)
                    model = self.build_model(hp["model_name"])

                    self.prepare_data(self.train_data, self.val_data)

                    print(f"\nTrial {trial.number} hyperparameters:")
                    print({**hp, **aug_params})

                    for k, v in {**hp, **aug_params}.items():
                        mlflow.log_param(k, v)

                    training_args = TrainingArguments(
                        output_dir=f"temp_models/{trial.number}",
                        learning_rate=hp["learning_rate"],
                        num_train_epochs=hp["num_train_epochs"],
                        per_device_train_batch_size=hp["per_device_train_batch_size"],
                        per_device_eval_batch_size=hp["per_device_train_batch_size"],
                        save_strategy="no",
                        evaluation_strategy="no",
                        logging_strategy="no",
                        load_best_model_at_end=False,
                        use_cpu=(self.device == "cpu"),
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=self.train_dataset,
                        eval_dataset=self.val_dataset,
                        compute_metrics=self.compute_metrics,
                    )

                    trainer.train()
                    metrics = trainer.evaluate()

                    print(f"\nTrial {trial.number} metrics:")
                    print({**metrics})
                    score = metrics.get("eval_mean_iou", 0.0)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[OOM] Trial {trial.number} failed due to insufficient GPU memory. Assigning score of -1")
                        metrics = {"eval_mean_iou": -1}
                        score = -1
                    else:
                        raise

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()

                self.log_metrics_during_optimization(metrics)
                mlflow.end_run()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        optuna.visualization.plot_optimization_history(study).write_html("temp_models/optimization_history.html")
        optuna.visualization.plot_param_importances(study).write_html("temp_models/param_importances.html")

        print("Best hyperparameters found:")
        print(study.best_params)
        print("Best value:", study.best_value)
        return study.best_params

    def log_metrics_during_optimization(self, metrics):
        mlflow.log_metric("eval_loss", metrics.get("eval_loss"))
        mlflow.log_metric("eval_mean_iou", metrics.get("eval_mean_iou"))
        mlflow.log_metric("eval_overall_accuracy", metrics.get("eval_overall_accuracy"))

        per_category_iou = metrics.get("eval_per_category_iou", [])[1:]
        per_category_acc = metrics.get("eval_per_category_accuracy", [])[1:]

        for i, iou in enumerate(per_category_iou, start=1):
            if iou is not None and not (iou != iou):
                mlflow.log_metric(f"eval_per_category_iou_class_{i}", iou)

        for i, acc in enumerate(per_category_acc, start=1):
            if acc is not None and not (acc != acc):
                mlflow.log_metric(f"eval_per_category_accuracy_class_{i}", acc)

    @abstractmethod
    def compute_metrics(self, eval_pred):
        pass


class LogModelCallback(TrainerCallback):
    def __init__(self, processor_name_or_path: str, model_name: str):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(processor_name_or_path)
        self.model_name = model_name

    def _make_dummy_input(self, batch_size: int = 1, height: int = 512, width: int = 512):
        dummy_image = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8).numpy()
        inputs = self.processor(images=dummy_image, return_tensors="pt")
        return inputs["pixel_values"]

    def _onnx_to_tensorrt(self, input_tensor, onnx_path: str, engine_path: str):
        if trt is None:
            print("TensorRT no está instalado. Se omite export a TensorRT.")
            return

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser:

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

            input_name = "seg_in"
            N, C, H, W = input_tensor.shape
            profile = builder.create_optimization_profile()
            profile.set_shape(input_name, min=(N, C, H, W), opt=(N, C, H, W), max=(N, C, H, W))
            config.add_optimization_profile(profile)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(parser.get_error(i))
                    raise RuntimeError("ONNX parsing failed")

            engine = builder.build_serialized_network(network, config)
            if engine is None:
                raise RuntimeError("TensorRT engine build failed")

            with open(engine_path, "wb") as f:
                f.write(engine)

    class _TraceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            try:
                output = self.model(pixel_values).logits
            except Exception:
                out = self.model(pixel_values)
                output = (out.class_queries_logits, out.masks_queries_logits)
            return output

    def on_train_end(self, args, state, control, model, **kwargs):
        dummy_input = self._make_dummy_input(batch_size=1)

        model.eval()
        model_cpu = model.to("cpu")

        model_cpu.save_pretrained("./my_model")
        mlflow.log_artifact("./my_model", artifact_path="model")
        mlflow.pytorch.log_model(model_cpu, artifact_path="model")
        print("✓ PyTorch model logged to MLflow")

        wrapped_model = self._TraceWrapper(model_cpu)

        traced_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
        torch_path = "model.pt"
        traced_model.save(torch_path)
        mlflow.log_artifact(torch_path, artifact_path="base")
        print("✓ TorchScript model traced and logged")

        onnx_path = "model.onnx"
        output_names = ["seg_out"] if self.model_name == "SegFormer" else ["class_logits", "mask_logits"]

        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            opset_version=17,
            do_constant_folding=True,
            input_names=["seg_in"],
            output_names=output_names,
            dynamic_axes=None
)

        mlflow.log_artifact(onnx_path, artifact_path="onnx")
        print("✓ ONNX model traced and logged")

        if trt is not None:
            engine_path = "model.plan"
            self._onnx_to_tensorrt(dummy_input, onnx_path, engine_path)
            if os.path.exists(engine_path):
                mlflow.log_artifact(engine_path, artifact_path="tensorrt")
                print("✓ TensorRT model traced and logged")
        else:
            print("TensorRT no disponible, se omite ese artefacto.")

        H_in = dummy_input.shape[2]
        W_in = dummy_input.shape[3]
        num_channels = model.config.num_labels

        input_name = "seg_in"
        output_name = "seg_out"

        if self.model_name == "SegFormer":
            H_out = H_in // 4
            W_out = W_in // 4
        elif self.model_name == "MaskFormer":
            H_out = H_in
            W_out = W_in
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")

        config_text = f"""
name: "segmenter"
platform: "pytorch_libtorch"
max_batch_size: 1

input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [{dummy_input.shape[1]}, {H_in}, {W_in}]
  }}
]

output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [{num_channels}, {H_out}, {W_out}]
  }}
]
"""
        mlflow.log_text(config_text.strip(), "model/data/config.pbtxt")

        print("✓ Triton config.pbtxt logged to MLflow")
        print("Training finished.")


class LogMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(state.epoch)

        mlflow.log_metric("eval_loss", metrics.get("eval_loss"), epoch)
        mlflow.log_metric("eval_mean_iou", metrics.get("eval_mean_iou"), epoch)
        mlflow.log_metric("eval_overall_accuracy", metrics.get("eval_overall_accuracy"), epoch)

        per_category_iou = metrics.get("eval_per_category_iou", [])[1:]
        per_category_acc = metrics.get("eval_per_category_accuracy", [])[1:]

        for i, iou in enumerate(per_category_iou, start=1):
            if iou is not None and not (iou != iou):
                mlflow.log_metric(f"eval_per_category_iou_class_{i}", iou, epoch)

        for i, acc in enumerate(per_category_acc, start=1):
            if acc is not None and not (acc != acc):
                mlflow.log_metric(f"eval_per_category_accuracy_class_{i}", acc, epoch)

        print(f"Logged metrics for epoch {epoch}")


class BestModelCallback(TrainerCallback):
    def __init__(self, metric_name="eval_loss", greater_is_better=False, mlflow_client=None, run_id=None):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_score = None
        self.best_epoch = None
        self.best_metrics = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric_name not in metrics:
            return

        current_value = metrics[self.metric_name]

        if self.best_score is None:
            self.best_score = current_value
            self.best_epoch = int(state.epoch)
            self.best_metrics = metrics
            return

        is_better = current_value > self.best_score if self.greater_is_better else current_value < self.best_score

        if is_better:
            self.best_score = current_value
            self.best_epoch = int(state.epoch)
            self.best_metrics = metrics

    def on_train_end(self, args, state, control, **kwargs):
        print("\n========== BEST MODEL INFO ==========")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best {self.metric_name}: {self.best_score}")
        print(f"Full metrics: {self.best_metrics}")
        print("=====================================\n")

        mlflow.log_metric(f"best_{self.metric_name}", self.best_score)
        mlflow.log_params({"best_epoch": self.best_epoch})
        mlflow.log_metric("best_eval_loss", self.best_metrics.get("eval_loss"), self.best_epoch)
        mlflow.log_metric("best_eval_mean_iou", self.best_metrics.get("eval_mean_iou"), self.best_epoch)
        mlflow.log_metric("best_eval_overall_accuracy", self.best_metrics.get("eval_overall_accuracy"), self.best_epoch)

        per_category_iou = self.best_metrics.get("eval_per_category_iou", [])[1:]
        per_category_acc = self.best_metrics.get("eval_per_category_accuracy", [])[1:]

        for i, iou in enumerate(per_category_iou, start=1):
            if iou is not None and not (iou != iou):
                mlflow.log_metric(f"best_eval_per_category_iou_class_{i}", iou, self.best_epoch)

        for i, acc in enumerate(per_category_acc, start=1):
            if acc is not None and not (acc != acc):
                mlflow.log_metric(f"best_eval_per_category_accuracy_class_{i}", acc, self.best_epoch)


class LogBestSegmentationCallback(TrainerCallback):
    def __init__(self, transformer_base):
        self.transformer_base = transformer_base

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        dataloader = kwargs["eval_dataloader"]

        model.eval()

        batch = next(iter(dataloader))
        pixel_values = batch["pixel_values"].to(model.device)
        labels = batch.get("labels", None)
        if labels is not None:
            labels = labels.to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        end_time = time.time()

        inference_time_ms = (end_time - start_time) * 1000
        mlflow.log_metric("inference_time_ms", inference_time_ms)

        if hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
        else:
            if not callable(self.transformer_base.postprocess):
                raise RuntimeError(
                    "Mask2Former outputs detected but no callable `postprocess` method found on callback."
                )

            if labels is None:
                labels = batch.get("mask_labels", None)
                labels = labels.to(model.device)

            eval_pred = (
                (
                    outputs.class_queries_logits.detach().cpu().numpy(),
                    outputs.masks_queries_logits.detach().cpu().numpy(),
                ),
                (
                    labels.detach().cpu().numpy(),
                    [],
                ),
            )
            preds = self.transformer_base.postprocess(eval_pred)
            preds = torch.stack(preds)

        self.transformer_base.plot_val_result(pixel_values, preds, labels)
