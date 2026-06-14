from abc import ABC, abstractmethod
from typing import Any, Dict
import mlflow
import torch


class BaseTrainer(ABC):
    """
    Abstract Trainer class that unifies training logic across
    Transformer (HuggingFace), PyTorch, and Keras frameworks.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_data = None
        self.val_data = None
        self.metrics = {}

        if "model_name" not in config:
            raise ValueError("Parameter model_name does not exist in config.")

        self.device = config.get("device", "cpu")

        if "mlflow_ip" not in config:
            raise ValueError("MLFlow IP is not defined in config.")

        mlflow.set_tracking_uri(config["mlflow_ip"])
        mlflow.set_experiment(config.get("experiment", "experiment"))

    @abstractmethod
    def build_model(self) -> Any:
        pass

    @abstractmethod
    def prepare_data(self, train_data: Any, val_data: Any) -> None:
        pass

    @abstractmethod
    def setup_training(self) -> None:
        pass

    @abstractmethod
    def fit(self, run_name: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        pass

    @abstractmethod
    def optimize_architecture(self, run_name: str, n_trials: int):
        pass

    def get_objective_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get("eval_mean_iou", float("inf"))

    def initialize(self):
        print("[BaseTrainer] Building model...")
        self.model = self.build_model()

        print("[BaseTrainer] Setting up training...")
        self.setup_training()

    def prepare(self, train_data: Any, val_data: Any):
        print("[BaseTrainer] Preparing data...")
        self.train_data = train_data
        self.val_data = val_data
        self.prepare_data(train_data, val_data)

    def run(self, train_data: Any, val_data: Any) -> Dict[str, float]:
        self.prepare(train_data, val_data)
        self.initialize()

        print("[BaseTrainer] Starting training...")
        training_name = self.config.get("training_name", "train")
        metrics = self.fit(training_name)

        print("[BaseTrainer] Training complete.")
        return metrics

    def log_train_aug_params(self):
        for aug_name, aug_config in self.config["augmentations"].items():
            value = aug_config.get("value")
            if isinstance(value, list):
                value = value[0]
            mlflow.log_param(aug_name, value)

        if torch.cuda.is_available() and self.device == "cuda":
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            mlflow.log_param("gpu_name", "cpu")

    def optimize(self, train_data: Any, val_data: Any, n_trials=100):
        self.prepare(train_data, val_data)
        self.initialize()

        training_name = self.config.get("training_name", "trial")
        best = self.optimize_architecture(training_name, n_trials)

        print("[BaseTrainer] Optimization complete.")
        return best