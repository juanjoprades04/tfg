from typing import Any, Dict
from transformers import Mask2FormerForUniversalSegmentation,AutoImageProcessor, TrainerCallback
from trainer import BaseTrainer
import math
from transformers import TrainingArguments
from transformers import Trainer
import mlflow
import torch
import numpy as np
import evaluate
from PIL import Image
import albumentations as A
import optuna
import matplotlib.pyplot as plt
from abc import abstractmethod
import gc
import detectron2
import detectron2.engine
import detectron2.config
import detectron2.modeling
import detectron2.solver
import detectron2.checkpoint
import detectron2.data.transforms as T
import detectron2.data
import detectron2.evaluation
import trainer.detectron_transforms
from pycocotools import mask as mask_utils
from skimage import measure
from detectron2.structures import BoxMode
import cv2
from pycocotools import mask as maskUtils

import torch
import numpy as np
import evaluate
import os
import tempfile
import contextlib


class DetectronBase(BaseTrainer):
    """
    Trainer for HuggingFace SegFormer models.
    Inherits from your generic BaseTrainer.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        print(detectron2)
        self.model_name = self.config["model_name"][0]
        self.args_detectron = detectron2.engine.default_argument_parser().parse_args()
        self.convert_model_string_to_args()
        
        self.cfg_detectron = self.setup_detectron(self.args_detectron)

        self.epochs = self.config["epochs"]["value"][0]
        self.lr = self.config["learning_rate"]["value"][0]
        self.batch_size = self.config["batch_size"]["value"][0]
        
        self.label_ids = None
        self.train_dataset = None
        self.val_dataset = None
        self.training_args = None
        self.hf_trainer = None  # HuggingFace Trainer object
        self.metric = None
        self.processor = None
        
        # -----------------------------------------
        #  Define augmentations
        # -----------------------------------------
        # Build augmentations dynamically
        self.augment = self.build_augmentations()


    # --------------------------------------------------
    # REQUIRED ABSTRACT METHOD IMPLEMENTATIONS
    # --------------------------------------------------
    def setup_detectron(self, args):
        cfg = detectron2.config.get_cfg()
        cfg.merge_from_file(args.config_file)
        #TODO: Unhardcode
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        detectron2.engine.default_setup(
            cfg, args
        )  # if you don't like any of the default setup, write your own setup code
        return cfg
        

    def convert_model_string_to_args(self):
        """
        Translate model_name string into Detectron2 config path
        for official instance segmentation configs.
        """
        base_path = "./trainer/"
        model_name = self.model_name.lower()

        MODEL_CONFIG_MAP = {
            # --------------------------------------------------
            # ResNet-50
            # --------------------------------------------------
            "mask_rcnn_r50_c4_1x":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",

            "mask_rcnn_r50_c4_3x":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",

            "mask_rcnn_r50_dc5_1x":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",

            "mask_rcnn_r50_dc5_3x":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",

            "mask_rcnn_r50_fpn_1x":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",

            "mask_rcnn_r50_fpn_1x_giou":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_giou.yaml",

            "mask_rcnn_r50_fpn_3x":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",

            # --------------------------------------------------
            # ResNet-101
            # --------------------------------------------------
            "mask_rcnn_r101_c4_3x":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",

            "mask_rcnn_r101_dc5_3x":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",

            "mask_rcnn_r101_fpn_3x":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",

            # --------------------------------------------------
            # ResNeXt
            # --------------------------------------------------
            "mask_rcnn_x101_32x8d_fpn_3x":
                base_path + "configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",

            # ---------------------------------
            # Cascade / Misc instance segmentation models
            # (these live in configs/Misc)
            # ---------------------------------
            "cascade_mask_rcnn_r50_fpn_3x":
                base_path + "configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",

            "cascade_mask_rcnn_r50_fpn_1x":
                base_path + "configs/Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml",
                
            # You can optionally add more cascade / misc variants 
            # supported in model zoo:
            "misc_mask_rcnn_r50_fpn_3x_syncbn":
                base_path + "configs/Misc/mask_rcnn_R_50_FPN_3x_syncbn.yaml",

            "misc_mask_rcnn_r50_fpn_3x_gn":
                base_path + "configs/Misc/mask_rcnn_R_50_FPN_3x_gn.yaml",

            # Scratch variants
            "misc_scratch_mask_rcnn_r50_fpn_3x_gn":
                base_path + "configs/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml",

            "misc_scratch_mask_rcnn_r50_fpn_9x_gn":
                base_path + "configs/Misc/scratch_mask_rcnn_R_50_FPN_9x_gn.yaml",
        }

        if model_name not in MODEL_CONFIG_MAP:
            raise ValueError(
                f"Unknown instance segmentation model '{model_name}'.\n"
                f"Available models: {list(MODEL_CONFIG_MAP.keys())}"
            )

        self.args_detectron.config_file = MODEL_CONFIG_MAP[model_name]
   
    def binary_mask_to_rle(self,binary_mask):
        rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('ascii')
        return rle
    
    def hf_to_detectron_dicts(self, hf_dataset):
        dataset_dicts = []
        for idx, sample in enumerate(hf_dataset):
            image_path = sample["image_path"]
            mask_path = sample["label_path"]

            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Image not found: {image_path}")
            height, width = img.shape[:2]

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            record = {
                "file_name": image_path,
                "image_id": idx,
                "height": height,
                "width": width,
                "annotations": [],
            }

            # Iterate over all classes except background (0)
            for class_id, class_name in self.config["labels"].items():
                if class_id == 0:
                    continue
                # Binary mask for this class
                binary_mask = (mask == class_id).astype(np.uint8)
                if binary_mask.sum() == 0:
                    continue

                # Optional: split connected components if multiple instances
                labeled_mask = measure.label(binary_mask)
                for instance_id in range(1, labeled_mask.max() + 1):
                    instance_mask = (labeled_mask == instance_id).astype(np.uint8)
                    ys, xs = np.where(instance_mask)
                    if len(xs) == 0 or len(ys) == 0:
                        continue
                    x0, x1 = xs.min(), xs.max()
                    y0, y1 = ys.min(), ys.max()

                    annotation = {
                        "segmentation": self.binary_mask_to_rle(instance_mask),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "bbox_mode": detectron2.structures.BoxMode.XYWH_ABS,
                        "category_id": class_id -1, #Doing this because coco json need classes from 0 to N, so we remove backgorund class
                    }
                    record["annotations"].append(annotation)

            dataset_dicts.append(record)
        return dataset_dicts
    
    def build_model(self) -> Any:
        model = detectron2.modeling.build_model(self.cfg_detectron)
        return model
    
    def prepare_data(self, train_data: Any, val_data: Any) -> None:
        self.train_data = train_data
        self.val_data = val_data
        # This is needed to build the object
        self.num_samples_train = len(train_data)

        # Class mapping from your config
        class_names = [v for k, v in sorted(self.config["labels"].items()) if k != 0]
        #
        # Register datasets
        try:
            detectron2.data.DatasetCatalog.register("my_train", lambda: self.hf_to_detectron_dicts(train_data))
            detectron2.data.MetadataCatalog.get("my_train").set(thing_classes=class_names)

            detectron2.data.DatasetCatalog.register("my_val", lambda: self.hf_to_detectron_dicts(val_data))
            detectron2.data.MetadataCatalog.get("my_val").set(thing_classes=class_names)
        except:
            pass        
        # Update config
        self.cfg_detectron.defrost()
        self.cfg_detectron.DATASETS.TRAIN = ("my_train",)
        self.cfg_detectron.DATASETS.TEST = ("my_val",)
        self.cfg_detectron.SOLVER.IMS_PER_BATCH = self.batch_size
        self.cfg_detectron.SOLVER.CLIP_GRADIENTS.ENABLED = True
        self.cfg_detectron.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
        self.cfg_detectron.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 5.0
        self.cfg_detectron.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
        self.cfg_detectron.freeze()
        self.data_loader = detectron2.data.build_detection_train_loader(self.cfg_detectron,
        mapper=detectron2.data.DatasetMapper(self.cfg_detectron, is_train=True, augmentations=self.augment, instance_mask_format="bitmask"))
        
        val_dataset = detectron2.data.DatasetCatalog.get("my_val")
        mapper = detectron2.data.DatasetMapper(cfg=self.cfg_detectron, is_train=True, augmentations=[], instance_mask_format="bitmask")
        mapped_val_dataset = detectron2.data.MapDataset(val_dataset, mapper)
        self.data_loader_val = torch.utils.data.DataLoader(
            mapped_val_dataset,
            batch_size=1,
            shuffle=False,   # deterministic
            drop_last=False, # last batch can be smaller
            collate_fn=detectron2.data.build.trivial_batch_collator,
        )

    def build_augmentations(self, aug_params=None):
        """
        Build Albumentations Compose based on config or provided hyperparameters.
        If aug_params is None, use defaults from self.config.
        """
        aug_config = self.config.get("augmentations", {})

        # Helper function: safely extract a float value
        def get_value(x):
            if isinstance(x, dict) and "value" in x:
                return x["value"][0]  # first value if it's a list
            return 0.0  # default if not present

        # Use hyperparams if provided, else config defaults
        aug_params = aug_params or {k: get_value(v) for k, v in aug_config.items()}


        return [
            T.RandomFlip(prob=aug_params.get("aug_hflip", 0.0), horizontal=True, vertical=False),
            T.RandomFlip(prob=aug_params.get("aug_vflip", 0.0), horizontal=False, vertical=True),
            trainer.detectron_transforms.RandomScale(scale_limit=1.2, prob=aug_params.get("aug_scale", 0.0)),
            T.RandomApply(T.RandomContrast(intensity_min=0.6, intensity_max=1.6), prob=aug_params.get("aug_brightness", 0.0)), #The values work differently that Albumentations
            T.RandomApply(T.RandomBrightness(intensity_min=0.6, intensity_max=1.6), prob=aug_params.get("aug_brightness", 0.0)), #The values work differently that Albumentations
            trainer.detectron_transforms.RandomHSV(
                hue_shift_limit=20,
                sat_shift_limit=30, 
                val_shift_limit=20, 
                prob=aug_params.get("aug_saturation", 0.0),
            ),
            trainer.detectron_transforms.RandomGaussianBlur(blur_limit=(3, 7), prob=aug_params.get("aug_gaussianblur", 0.0)),
            trainer.detectron_transforms.RandomMotionBlur(blur_limit=7, prob=aug_params.get("aug_motionblur", 0.0)),
            trainer.detectron_transforms.RandomGaussianNoise(std_range=(0.05, 0.1), prob=aug_params.get("aug_gaussiannoise", 0.0)),
            trainer.detectron_transforms.RandomISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.5), prob=aug_params.get("aug_isonoise", 0.0))        
        ]

    def setup_training(self) -> None:
        # Create Trainer args, define self.hf_trainer = Trainer(...)
        self.model.train()
        

        self.cfg_detectron.defrost()
        self.cfg_detectron.SOLVER.BASE_LR = self.lr
        num_batches_per_epoch = math.ceil(self.num_samples_train / self.batch_size)
        self.cfg_detectron.TEST.EVAL_PERIOD = num_batches_per_epoch
        self.cfg_detectron.SOLVER.MAX_ITER = num_batches_per_epoch * self.epochs
        self.cfg_detectron.SOLVER.STEPS = (
            int(0.6 * self.cfg_detectron.SOLVER.MAX_ITER),
            int(0.8 * self.cfg_detectron.SOLVER.MAX_ITER),
        )
        self.cfg_detectron.SOLVER.WARMUP_ITERS = int(0.02 * self.cfg_detectron.SOLVER.MAX_ITER)
        self.cfg_detectron.SOLVER.CHECKPOINT_PERIOD = num_batches_per_epoch * 5
        self.cfg_detectron.freeze()
        
        self.optimizer = detectron2.solver.build_optimizer(self.cfg_detectron, self.model)
        self.scheduler = detectron2.solver.build_lr_scheduler(self.cfg_detectron, self.optimizer)

        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(
            self.model, self.cfg_detectron.OUTPUT_DIR, optimizer=self.optimizer, scheduler=self.scheduler
        )
        self.start_iter = (
            self.checkpointer.resume_or_load(self.cfg_detectron.MODEL.WEIGHTS, resume=False).get("iteration", -1) + 1
        )
        self.max_iter = self.cfg_detectron.SOLVER.MAX_ITER

        self.periodic_checkpointer = detectron2.checkpoint.PeriodicCheckpointer(
            self.checkpointer, self.cfg_detectron.SOLVER.CHECKPOINT_PERIOD, max_iter=self.max_iter
        )

        self.writers = detectron2.engine.default_writers(self.cfg_detectron.OUTPUT_DIR, self.max_iter)
        self.num_classes = len(self.config["labels"].items())-1

    def fit(self, run_name: str, active_run=False) -> Dict[str, Any]:
        # Call self.hf_trainer.train()
        logval_metrics = LogMetricsCallback()
        bestmodel_callback = BestModelCallback()
        model_logger = LogModelCallback(self.cfg_detectron)
        if not active_run:
            run_ctx = mlflow.start_run(run_name=run_name)
        else:
            run_ctx = contextlib.nullcontext()
            
        with run_ctx:
            with detectron2.utils.events.EventStorage(self.start_iter) as storage:
                epoch = 0
                for data, iteration in zip(self.data_loader, range(self.start_iter, self.max_iter)):
                    self.model.train()
                    storage.iter = iteration

                    loss_dict = self.model(data)
                    losses = sum(loss_dict.values())
                    assert torch.isfinite(losses).all(), loss_dict

                    loss_dict_reduced = {k: v.item() for k, v in detectron2.utils.comm.reduce_dict(loss_dict).items()}
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    if detectron2.utils.comm.is_main_process():
                        storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()
                    storage.put_scalar("lr", self.optimizer.param_groups[0]["lr"], smoothing_hint=False)
                    self.scheduler.step()

                    
                    if (
                        self.cfg_detectron.TEST.EVAL_PERIOD > 0
                        and (iteration + 1) % self.cfg_detectron.TEST.EVAL_PERIOD == 0
                        and iteration != 0
                    ):
                        results_val = self.do_test()
                        logval_metrics.on_evaluate(epoch, results_val)
                        bestmodel_callback.on_evaluate(epoch, self.checkpointer, results_val)
                        # Compared to "train_net.py", the test results are not dumped to EventStorage
                        detectron2.utils.comm.synchronize()
                        epoch += 1

                    if iteration - self.start_iter > 5 and (
                        (iteration + 1) % 20 == 0 or iteration == self.max_iter - 1
                    ):
                        for writer in self.writers:
                            writer.write()
                    self.periodic_checkpointer.step(iteration)
                bestmodel_callback.on_train_end()
                model_logger.on_train_end()
                self.plot_val_result_detectron2()
            if not active_run:    
                self.log_train_aug_params()

        mlflow.end_run()
        
    def do_test(self):     
        self.model.train()  # REQUIRED to compute losses
        losses_sum = {}
        num_batches = 0
        
        # Load validation dataset
        val_dataset = detectron2.data.DatasetCatalog.get("my_val")
        # Load mIoU metric
        metric = evaluate.load("mean_iou")

        # TODO: Check this
        #device = next(self.model.parameters()).device  # ensure tensors on correct device
        # --------- validation loss ----------
        with detectron2.utils.events.EventStorage(start_iter=0):
            with torch.no_grad():
                for batch in self.data_loader_val:
                    # batch is a list of dicts, just like training
                    loss_dict = self.model(batch)  # returns dict of losses

                    # Reduce across processes (for multi-GPU) if needed
                    loss_dict = detectron2.utils.comm.reduce_dict(loss_dict)

                    # Sum losses
                    for k, v in loss_dict.items():
                        losses_sum[k] = losses_sum.get(k, 0.0) + v.item()

                    num_batches += 1

        val_losses = {k: v / num_batches for k, v in losses_sum.items()}
        val_loss = sum(val_losses.values())
        
        self.model.eval()

        for record in val_dataset:
            # --- Load image ---
            # Detectron2 stores file path in "file_name"
            image_path = record["file_name"]
            import cv2
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H, W, _ = image.shape

            # --- Run model ---
            with torch.no_grad():
                # Convert to tensor CxHxW, add batch dim
                image_tensor = torch.as_tensor(image.transpose(2,0,1)).to(self.model.device).float()
                inputs = [{"image": image_tensor, "height": H, "width": W}]
                outputs_list = self.model(inputs)
            outputs = outputs_list[0]
            # --- Convert predicted instances to semantic mask ---
            pred_mask = np.zeros((H, W), dtype=np.int32)
            if len(outputs["instances"]) > 0:
                pred_masks = outputs["instances"].pred_masks.cpu().numpy()
                pred_classes = outputs["instances"].pred_classes.cpu().numpy()
                for m, c in zip(pred_masks, pred_classes):
                    pred_mask[m] = c + 1  # +1 if background=0

            # --- Convert GT instances to semantic mask ---
            gt_mask = np.zeros((H, W), dtype=np.int32)
            for ann in record.get("annotations", []):
                mask_rle = ann["segmentation"]
                mask = maskUtils.decode(mask_rle)
                if mask.ndim == 3:  # sometimes decode returns HxWx1
                    mask = mask[:, :, 0]
                gt_mask[mask == 1] = ann["category_id"] + 1

            # --- Add batch to metric ---
            metric.add_batch(predictions=[pred_mask], references=[gt_mask])

        # --- Compute final mIoU ---
        results = metric.compute(
            num_labels=self.num_classes + 1,  # +1 for background
            ignore_index=0  # ignore background
        )
        results['mean_iou'] = np.float64(np.mean(results['per_category_iou'][1:]))
        results['val_loss'] = val_loss
        results['val_losses'] = val_losses
        
        print(f"Validation mIoU: {results['mean_iou']:.4f}")
        return results

    def plot_val_result_detectron2(self, dataset_name="my_val"):
        best_model_src = os.path.join(
        self.cfg_detectron.OUTPUT_DIR,
        "model_best.pth"
         )
        # Fix this
        w_before = self.model.roi_heads.box_predictor.cls_score.weight.clone()
        self.checkpointer.load(best_model_src)
        w_after = self.model.roi_heads.box_predictor.cls_score.weight.clone()
        
        print((w_before - w_after).abs().max())
        self.model.eval()

        # ---- Take first validation sample ----
        dataset = detectron2.data.DatasetCatalog.get(dataset_name)
        record = dataset[0]

        image_path = record["file_name"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape

        # ---- Run inference ----
        with torch.no_grad():
            image_tensor = torch.as_tensor(image.transpose(2, 0, 1)).float().to(self.model.device)
            inputs = [{"image": image_tensor, "height": H, "width": W}]
            outputs = self.model(inputs)[0]

        # ---- Predicted semantic mask ----
        pred_mask = np.zeros((H, W), dtype=np.int32)
        if "instances" in outputs and len(outputs["instances"]) > 0:
            instances = outputs["instances"].to("cpu")
            pred_masks = instances.pred_masks.numpy()
            pred_classes = instances.pred_classes.numpy()

            for m, c in zip(pred_masks, pred_classes):
                pred_mask[m] = c + 1  # background = 0

        # ---- Ground-truth semantic mask ----
        gt_mask = np.zeros((H, W), dtype=np.int32)
        for ann in record.get("annotations", []):
            mask_rle = ann["segmentation"]
            mask = maskUtils.decode(mask_rle)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            gt_mask[mask == 1] = ann["category_id"] + 1

        # ---- Plot ----
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="tab20")
        axes[1].set_title("Prediction (Best Model)")
        axes[1].axis("off")

        axes[2].imshow(gt_mask, cmap="tab20")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        # ---- Log to MLflow ----
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "best_model_prediction.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close(fig)

            mlflow.log_artifact(path, artifact_path="best_model_segmentation")

    #@abstractmethod
    def predict(self, inputs: Any) -> Any:
        # Use self.hf_trainer.predict(inputs)
        pass
    
    #@abstractmethod
    def load_checkpoint(self, path: str) -> None:
        #self.model = SegformerForSemanticSegmentation.from_pretrained(path)
        pass
       
    def optuna_hp_space(self, trial):
        # 1️⃣ Sample hyperparameters
        hp = {}

        # Learning rate
        lr_range = self.config.get("learning_rate", {}).get("value", [1e-4])
        if len(lr_range) > 1:
            hp["learning_rate"] = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)
        else:
            hp["learning_rate"] = lr_range[0]

        # Batch size
        batch_range = self.config.get("batch_size", {}).get("value", [8])
        if len(batch_range) > 1:
            hp["per_device_train_batch_size"] = trial.suggest_int("per_device_train_batch_size", batch_range[0], batch_range[1])
        else:
            hp["per_device_train_batch_size"] = batch_range[0]

        # Epochs
        epochs_range = self.config.get("epochs", {}).get("value", [20])
        if len(epochs_range) > 1:
            hp["num_train_epochs"] = trial.suggest_int("num_train_epochs", epochs_range[0], epochs_range[1])
        else:
            hp["num_train_epochs"] = epochs_range[0]

        # Model range
        model_range = self.config.get("model_name", [])
        if len(model_range) == 0:
            raise ValueError("Error in model_name")
        elif len(model_range) > 1:
            hp["model_name"] = trial.suggest_categorical("model_name", model_range)
        else:
            hp["model_name"] = model_range[0]

        # Augmentation parameters
        aug_config = self.config.get("augmentations", {})
        aug_params = {}
        for key, val in aug_config.items():
            val_range = val.get("value", [0.0])
            if len(val_range) > 1:
                aug_params[key] = trial.suggest_float(key, val_range[0], val_range[1])
            else:
                aug_params[key] = val_range[0]
        return hp, aug_params
    
    def get_aug_params_from_trial(self, trial):
        """
        Extract augmentation hyperparameters from an Optuna trial.
        These are applied to dataset preprocessing, not TrainingArguments.
        """
        aug_params = {}
        aug_config = self.config.get("augmentations", {})
        for key, val in aug_config.items():
            val_range = val.get("value", [0.0])
            if len(val_range) > 1:
                aug_params[key] = trial.suggest_float(key, val_range[0], val_range[1])
            else:
                aug_params[key] = val_range[0]
        return aug_params
    
    def optimize_architecture(self, name, n_trials=100):
        """
        Run hyperparameter optimization, including augmentation parameters.
        """

        def objective(trial):
            run_name = f"{name}_{trial.number}"
            with mlflow.start_run(run_name=run_name):
                try:

                    # Sample hyperparameters
                    hp, aug_params = self.optuna_hp_space(trial)

                    self.model_name = hp["model_name"]
                    self.args_detectron = detectron2.engine.default_argument_parser().parse_args()
                    self.convert_model_string_to_args()
                    self.cfg_detectron = self.setup_detectron(self.args_detectron)

                    # 2️⃣ Apply sampled augmentations
                    self.augment = self.build_augmentations(aug_params)

                    # 3️⃣ Reset / build a fresh model for this trial
                    model = self.build_model()

                    # 🔹 Log the parameters being used in this trial
                    print(f"\nTrial {trial.number} hyperparameters:")
                    print({**hp, **aug_params})
                    tmp = {**hp, **aug_params}
                    
                    for k, v in tmp.items():
                        mlflow.log_param(k, v)
                    self.lr = hp["learning_rate"]
                    self.epochs = hp["num_train_epochs"]
                    self.batch_size = hp["per_device_train_batch_size"]
                    self.prepare_data(self.train_data, self.val_data)
                    self.setup_training()                    
                    self.fit(run_name, active_run=True)

                    # 6️⃣ Evaluate and return metric for Optuna
                    metrics = self.do_test()
                    print(f"\nTrial {trial.number} metrics:")
                    print({**metrics})
                    score = metrics.get("mean_iou", 0.0)
                except RuntimeError as e:
                    # --------------------
                    #     CUDA OOM !!
                    # --------------------
                    if "out of memory" in str(e).lower():
                        print(f"[OOM] Trial {trial.number} failed due to insufficient GPU memory. Assigning score of -1")
                        metrics["mean_iou"] = -1
                        # Assign the worst possible score
                        score = -1
                    else:
                        print(e)
                        raise  # Not memory-related → rethrow
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()       # <--- mandatory!
                gc.collect()
                
            mlflow.end_run()
            return score

        # 7️⃣ Run Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        optuna.visualization.plot_optimization_history(study).write_html(f"temp_models/optimization_history.html")
        optuna.visualization.plot_param_importances(study).write_html(f"temp_models/param_importances.html")

        print("Best hyperparameters found:")
        print(study.best_params)
        print("Best value:", study.best_value)
        return study.best_params

class LogModelCallback():
    """
    A clean HuggingFace Trainer callback that logs the trained model to MLflow
    using TorchScript tracing, with a dummy input generated via AutoImageProcessor.
    Also generates Triton config.pbtxt for deployment.
    """
    def __init__(self, cfg_detectron):
        self.cfg_detectron = cfg_detectron
        
    def on_train_end(self):
        """
        Logs the trained model using TorchScript + MLflow, and generates Triton config.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # -----------------------
            # Copy model_best.pth to temp dir
            # -----------------------
            best_model_src = os.path.join(self.cfg_detectron.OUTPUT_DIR, "model_best.pth")
            best_model_dst = os.path.join(tmpdir, "model_best.pth")
            if not os.path.exists(best_model_src):
                raise FileNotFoundError(f"{best_model_src} does not exist! Check your checkpointer.")
            import shutil
            shutil.copy(best_model_src, best_model_dst)

            # -----------------------
            # Save config
            # -----------------------
            cfg_path = os.path.join(tmpdir, "config.yaml")
            with open(cfg_path, "w") as f:
                f.write(self.cfg_detectron.dump())

            # -----------------------
            # Log to MLflow
            # -----------------------
            mlflow.log_artifact(best_model_dst, artifact_path="model")
            mlflow.log_artifact(cfg_path, artifact_path="model")
      
class LogMetricsCallback():
    """
    A callback that logs evaluation metrics at the end of each evaluation using MLflow.
    Only foreground classes are logged for per-category metrics.
    """

    def on_evaluate(self, epoch, metrics=None):
        """
        Called after every evaluation.
        Metrics from compute_metrics are passed as `metrics`.
        """
        # Log general metrics
        mlflow.log_metric("eval_loss", metrics.get("val_loss"), epoch)
        mlflow.log_metric("eval_mean_iou", metrics.get("mean_iou"), epoch)
        mlflow.log_metric("eval_overall_accuracy", metrics.get("overall_accuracy"), epoch)
 
        # Log per-class metrics, skipping background (index 0)
        per_category_iou = metrics.get("per_category_iou", [])[1:]
        per_category_acc = metrics.get("per_category_accuracy", [])[1:]

        for i, iou in enumerate(per_category_iou, start=1):
            if iou is not None and not (iou != iou):  # skip NaN
                mlflow.log_metric(f"eval_per_category_iou_class_{i}", iou, epoch)

        for i, acc in enumerate(per_category_acc, start=1):
            if acc is not None and not (acc != acc):  # skip NaN
                mlflow.log_metric(f"eval_per_category_accuracy_class_{i}", acc, epoch)

        print(f"Logged metrics for epoch {epoch}")
        
       
class BestModelCallback():
    """
    Tracks the best epoch and metrics saved by Trainer.
    Works independently of the built-in HF 'load_best_model_at_end'.
    """

    def __init__(self, metric_name="val_loss", greater_is_better=False):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_score = None
        self.best_epoch = None
        self.best_metrics = None

    def on_evaluate(self, epoch, checkpointer, metrics=None):
        """Called at every eval step (usually end of epoch)."""

        if metrics is None or self.metric_name not in metrics:
            return

        current_value = metrics[self.metric_name]

        # First time
        if self.best_score is None:
            self.best_score = current_value
            self.best_epoch = epoch
            self.best_metrics = metrics
            return

        # Better metric?
        is_better = (
            current_value > self.best_score if self.greater_is_better else current_value < self.best_score
        )

        if is_better:
            self.best_score = current_value
            self.best_epoch = epoch
            self.best_metrics = metrics
            checkpointer.save("model_best")

    def on_train_end(self):
        """Print/log best result at the end."""

        print("\n========== BEST MODEL INFO ==========")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best {self.metric_name}: {self.best_score}")
        print(f"Full metrics: {self.best_metrics}")
        print("=====================================\n")


        mlflow.log_metric(f"best_{self.metric_name}", self.best_score)
        mlflow.log_params({"best_epoch": self.best_epoch})
        
        # Log general metrics
        mlflow.log_metric("best_eval_loss", self.best_metrics.get("val_loss"), self.best_epoch)
        mlflow.log_metric("best_eval_mean_iou", self.best_metrics.get("mean_iou"), self.best_epoch)
        mlflow.log_metric("best_eval_overall_accuracy", self.best_metrics.get("overall_accuracy"), self.best_epoch)

        # Log per-class metrics, skipping background (index 0)
        per_category_iou = self.best_metrics.get("per_category_iou", [])[1:]
        per_category_acc = self.best_metrics.get("per_category_accuracy", [])[1:]

        for i, iou in enumerate(per_category_iou, start=1):
            if iou is not None and not (iou != iou):  # skip NaN
                mlflow.log_metric(f"best_eval_per_category_iou_class_{i}", iou, self.best_epoch)

        for i, acc in enumerate(per_category_acc, start=1):
            if acc is not None and not (acc != acc):  # skip NaN
                mlflow.log_metric(f"best_eval_per_category_accuracy_class_{i}", acc, self.best_epoch)
