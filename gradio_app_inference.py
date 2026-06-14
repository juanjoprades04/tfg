import json
import os
import threading
import traceback
import gradio as gr
from dotenv import load_dotenv

from main_backend import (
    run_training_pipeline,
    run_optimization_pipeline,
    detect_task_metadata,
)
from magna_utils.magna_mlflow import list_experiments, list_runs, get_run_url
from inference_utils import DEFAULT_MODEL_DIR, run_orange_inference

load_dotenv()

MODEL_BUNDLES = {
    "SegFormer": [
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "nvidia/segformer-b1-finetuned-ade-512-512",
        "nvidia/segformer-b2-finetuned-ade-512-512",
    ],
    "MaskFormer": [
        "facebook/mask2former-swin-tiny-ade-semantic",
        "facebook/mask2former-swin-small-ade-semantic",
    ],
    "FCN": [
        "fcn_resnet50",
        "fcn_resnet101",
    ],
    "DeepLabV3": [
        "deeplabv3_resnet50",
        "deeplabv3_resnet101",
    ],
}

CSS = """
.gradio-container {
    background: #f7f7f7 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
}
.app { background: #f7f7f7 !important; }

.gft-header {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 6px;
    padding: 18px 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 14px;
}
.gft-logo {
    font-size: 1.35rem;
    font-weight: 800;
    color: #111111;
    letter-spacing: 0.5px;
}
.gft-vline { width: 1px; height: 28px; background: #dddddd; }
.gft-title {
    color: #111111 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}
.gft-sub {
    color: #888888 !important;
    font-size: 0.75rem !important;
    margin: 2px 0 0 !important;
    letter-spacing: 0.3px;
}
.gft-spacer { flex: 1; }
.gft-badge {
    background: #111111;
    color: #ffffff !important;
    padding: 3px 10px;
    border-radius: 3px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.pipeline-bar {
    background: #ffffff !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    align-items: center !important;
    gap: 8px !important;
    box-shadow: none !important;
    margin-bottom: 14px;
}
#pipeline-status { flex: 1 !important; min-width: 0 !important; }
.pipeline-label {
    color: #aaaaaa !important;
    font-size: 0.67rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-right: 6px;
}
.agent-step {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 12px;
    border-radius: 3px;
    font-size: 0.73rem;
    font-weight: 600;
}
.step-pending {
    background: #f2f2f2;
    color: #bbbbbb !important;
    border: 1px solid #e5e5e5;
}
.step-running {
    background: #111111;
    color: #ffffff !important;
    border: 1px solid #111111;
    animation: step-pulse 1.4s ease-in-out infinite;
}
.step-running::before {
    content: '';
    display: inline-block;
    width: 5px;
    height: 5px;
    background: #ffffff;
    border-radius: 50%;
    animation: blink 1s infinite;
}
.step-done {
    background: #ffffff;
    color: #111111 !important;
    border: 1px solid #111111;
}
.step-done::before {
    content: '✓';
    font-size: 0.7rem;
    font-weight: 900;
}
.step-arrow {
    color: #cccccc !important;
    font-size: 0.75rem;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
@keyframes step-pulse { 0%,100%{opacity:1} 50%{opacity:.8} }

.block {
    background: #ffffff !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}

label {
    color: #444444 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

input[type="text"], textarea {
    background: #fafafa !important;
    color: #111111 !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 4px !important;
    font-size: 0.84rem !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #111111 !important;
    background: #ffffff !important;
    box-shadow: none !important;
    outline: none !important;
}

button {
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
}
button.primary, .gr-button-primary {
    background: #111111 !important;
    color: #ffffff !important;
    border: 2px solid #111111 !important;
    padding: 10px 18px !important;
}
button.primary:hover {
    background: #333333 !important;
    border-color: #333333 !important;
}
button.secondary, .gr-button-secondary {
    background: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #cccccc !important;
    padding: 7px 12px !important;
}
button.secondary:hover {
    border-color: #111111 !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f5f5f5; }
::-webkit-scrollbar-thumb { background: #cccccc; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #888; }
"""

job_state = {
    "running": False,
    "done": False,
    "error": None,
    "stage": "idle",
    "percent": 0,
    "logs": "",
    "result": "",
    "mlflow": {},
}


class LiveLogger:
    def __init__(self):
        self.buffer = []

    def write(self, text):
        if text:
            self.buffer.append(text)
            job_state["logs"] = "".join(self.buffer)

    def flush(self):
        pass


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def apply_cvat_env(cvat_url_input=None, cvat_username_input=None, cvat_password_input=None):
    if cvat_url_input and cvat_url_input.strip():
        os.environ["CVAT_URL"] = cvat_url_input.strip()

    if cvat_username_input and cvat_username_input.strip():
        os.environ["CVAT_USERNAME"] = cvat_username_input.strip()

    if cvat_password_input and cvat_password_input.strip():
        os.environ["CVAT_PASSWORD"] = cvat_password_input.strip()


def get_latest_run_info(experiment_name):
    try:
        experiments = list_experiments()
        if not experiments:
            return {"error": "No se pudieron obtener experimentos de MLflow"}

        experiment = next((e for e in experiments if e["name"] == experiment_name), None)
        if experiment is None:
            return {"error": f"No existe el experimento '{experiment_name}' en MLflow"}

        runs = list_runs(experiment["experiment_id"])
        if not runs:
            return {"error": "No hay runs registradas todavía para este experimento"}

        latest_run = sorted(runs, key=lambda x: x["run_start_time"], reverse=True)[0]
        run_url = get_run_url(latest_run["run_id"])

        return {
            "experiment_id": experiment["experiment_id"],
            "run_id": latest_run["run_id"],
            "status": latest_run["run_status"],
            "start_time": latest_run["run_start_time"],
            "metrics": latest_run["run_metrics"],
            "params": latest_run["run_params"],
            "url": run_url,
        }
    except Exception as e:
        return {"error": f"Error consultando MLflow: {e}"}


def build_param_config(mode, fixed_value, min_value, max_value, cast_fn=float):
    if mode == "train":
        return {"value": [cast_fn(fixed_value)]}

    min_v = cast_fn(min_value)
    max_v = cast_fn(max_value)
    if min_v > max_v:
        raise ValueError(f"Rango inválido: mínimo {min_v} mayor que máximo {max_v}")
    return {"value": [min_v, max_v]}


def build_augmentations_config(
    mode,
    aug_hflip_fixed, aug_hflip_min, aug_hflip_max,
    aug_vflip_fixed, aug_vflip_min, aug_vflip_max,
    aug_scale_fixed, aug_scale_min, aug_scale_max,
    aug_brightness_fixed, aug_brightness_min, aug_brightness_max,
    aug_saturation_fixed, aug_saturation_min, aug_saturation_max,
    aug_gaussianblur_fixed, aug_gaussianblur_min, aug_gaussianblur_max,
    aug_motionblur_fixed, aug_motionblur_min, aug_motionblur_max,
    aug_gaussiannoise_fixed, aug_gaussiannoise_min, aug_gaussiannoise_max,
    aug_isonoise_fixed, aug_isonoise_min, aug_isonoise_max,
):
    return {
        "aug_hflip": build_param_config(mode, aug_hflip_fixed, aug_hflip_min, aug_hflip_max, float),
        "aug_vflip": build_param_config(mode, aug_vflip_fixed, aug_vflip_min, aug_vflip_max, float),
        "aug_scale": build_param_config(mode, aug_scale_fixed, aug_scale_min, aug_scale_max, float),
        "aug_brightness": build_param_config(mode, aug_brightness_fixed, aug_brightness_min, aug_brightness_max, float),
        "aug_saturation": build_param_config(mode, aug_saturation_fixed, aug_saturation_min, aug_saturation_max, float),
        "aug_gaussianblur": build_param_config(mode, aug_gaussianblur_fixed, aug_gaussianblur_min, aug_gaussianblur_max, float),
        "aug_motionblur": build_param_config(mode, aug_motionblur_fixed, aug_motionblur_min, aug_motionblur_max, float),
        "aug_gaussiannoise": build_param_config(mode, aug_gaussiannoise_fixed, aug_gaussiannoise_min, aug_gaussiannoise_max, float),
        "aug_isonoise": build_param_config(mode, aug_isonoise_fixed, aug_isonoise_min, aug_isonoise_max, float),
    }


def build_config(
    mode,
    model,
    model_name_single,
    model_name_multi,
    experiment,
    training_name,
    mlflow_input,
    cvat_url_input,
    cvat_username_input,
    cvat_password_input,
    train_split,
    cvat_task_ids,
    device,
    lr_fixed, lr_min, lr_max,
    batch_fixed, batch_min, batch_max,
    epochs_fixed, epochs_min, epochs_max,
    n_trials,
    aug_hflip_fixed, aug_hflip_min, aug_hflip_max,
    aug_vflip_fixed, aug_vflip_min, aug_vflip_max,
    aug_scale_fixed, aug_scale_min, aug_scale_max,
    aug_brightness_fixed, aug_brightness_min, aug_brightness_max,
    aug_saturation_fixed, aug_saturation_min, aug_saturation_max,
    aug_gaussianblur_fixed, aug_gaussianblur_min, aug_gaussianblur_max,
    aug_motionblur_fixed, aug_motionblur_min, aug_motionblur_max,
    aug_gaussiannoise_fixed, aug_gaussiannoise_min, aug_gaussiannoise_max,
    aug_isonoise_fixed, aug_isonoise_min, aug_isonoise_max,
):
    mlflow_uri = mlflow_input.strip() if mlflow_input and mlflow_input.strip() else os.getenv("MLFLOW_TRACKING_URI")

    if not mlflow_uri:
        raise ValueError("No se ha indicado MLflow URI ni existe MLFLOW_TRACKING_URI en .env")

    task_ids = parse_int_list(cvat_task_ids)
    if not task_ids:
        raise ValueError("Debes indicar al menos una CVAT task")

    if mode == "train":
        model_names = [model_name_single]
    else:
        model_names = model_name_multi if model_name_multi else [model_name_single]

    return {
        "model": model,
        "model_name": model_names,
        "experiment": experiment,
        "training_name": training_name,
        "mlflow_ip": mlflow_uri,
        "cvat_url": cvat_url_input.strip() if cvat_url_input and cvat_url_input.strip() else None,
        "cvat_username": cvat_username_input.strip() if cvat_username_input and cvat_username_input.strip() else None,
        "cvat_password": cvat_password_input.strip() if cvat_password_input and cvat_password_input.strip() else None,
        "train_split": float(train_split),
        "cvat_task_ids": task_ids,
        "learning_rate": build_param_config(mode, lr_fixed, lr_min, lr_max, float),
        "batch_size": build_param_config(mode, batch_fixed, batch_min, batch_max, int),
        "epochs": build_param_config(mode, epochs_fixed, epochs_min, epochs_max, int),
        "augmentations": build_augmentations_config(
            mode,
            aug_hflip_fixed, aug_hflip_min, aug_hflip_max,
            aug_vflip_fixed, aug_vflip_min, aug_vflip_max,
            aug_scale_fixed, aug_scale_min, aug_scale_max,
            aug_brightness_fixed, aug_brightness_min, aug_brightness_max,
            aug_saturation_fixed, aug_saturation_min, aug_saturation_max,
            aug_gaussianblur_fixed, aug_gaussianblur_min, aug_gaussianblur_max,
            aug_motionblur_fixed, aug_motionblur_min, aug_motionblur_max,
            aug_gaussiannoise_fixed, aug_gaussiannoise_min, aug_gaussiannoise_max,
            aug_isonoise_fixed, aug_isonoise_min, aug_isonoise_max,
        ),
        "device": device,
        "n_trials": int(n_trials),
    }


def render_pipeline_html():
    stage = job_state["stage"]
    steps = [
        ("Labels", "detecting_labels"),
        ("Data", "preparing_data"),
        ("Run", "training" if stage == "training" else "optimizing"),
        ("MLflow", "querying_mlflow"),
        ("Done", "finished"),
    ]

    order = ["detecting_labels", "preparing_data", "training", "optimizing", "querying_mlflow", "finished"]

    def cls(step_name):
        if stage == "finished":
            return "step-done"
        if step_name == stage:
            return "step-running"
        if stage in order and step_name in order and order.index(step_name) < order.index(stage):
            return "step-done"
        return "step-pending"

    html = '<div class="pipeline-label">Pipeline</div>'
    for i, (label, step_name) in enumerate(steps):
        html += f'<span class="agent-step {cls(step_name)}">{label}</span>'
        if i < len(steps) - 1:
            html += '<span class="step-arrow">›</span>'
    return html


def training_worker(config, mode):
    logger = LiveLogger()

    try:
        job_state["running"] = True
        job_state["done"] = False
        job_state["error"] = None
        job_state["stage"] = "starting"
        job_state["percent"] = 0
        job_state["logs"] = ""
        job_state["result"] = ""
        job_state["mlflow"] = {}

        import contextlib

        with contextlib.redirect_stdout(logger), contextlib.redirect_stderr(logger):
            logger.write("=== INICIO DEL JOB ===\n")
            safe_config = dict(config)
            if safe_config.get("cvat_password"):
                safe_config["cvat_password"] = "********"
            logger.write(json.dumps(safe_config, indent=2, ensure_ascii=False) + "\n\n")

            apply_cvat_env(
                config.get("cvat_url"),
                config.get("cvat_username"),
                config.get("cvat_password"),
            )

            job_state["stage"] = "detecting_labels"
            job_state["percent"] = 5
            logger.write("Detectando labels y número de clases desde CVAT...\n")

            metadata = detect_task_metadata(config["cvat_task_ids"])
            config["labels"] = metadata["labels"]
            config["num_classes"] = metadata["num_classes"]

            logger.write(f"Labels detectadas: {json.dumps(metadata['labels'], ensure_ascii=False)}\n")
            logger.write(f"Número de clases detectado: {metadata['num_classes']}\n\n")

            job_state["stage"] = "preparing_data"
            job_state["percent"] = 10
            logger.write("Preparando datos desde CVAT...\n")

            if mode == "train":
                job_state["stage"] = "training"
                job_state["percent"] = 40
                logger.write("Lanzando entrenamiento...\n")
                run_training_pipeline(config)
                job_state["result"] = "✅ Training completado"
            else:
                job_state["stage"] = "optimizing"
                job_state["percent"] = 40
                logger.write("Lanzando optimización con Optuna...\n")
                run_optimization_pipeline(config)
                job_state["result"] = "✅ Optimización completada"

            job_state["stage"] = "querying_mlflow"
            job_state["percent"] = 90
            logger.write("Consultando última run en MLflow...\n")
            job_state["mlflow"] = get_latest_run_info(config["experiment"])

            job_state["stage"] = "finished"
            job_state["percent"] = 100
            job_state["done"] = True
            logger.write("=== JOB FINALIZADO ===\n")

    except Exception as e:
        job_state["error"] = str(e)
        job_state["result"] = f"❌ Error: {e}"
        job_state["stage"] = "error"
        job_state["percent"] = 100
        job_state["logs"] += "\n" + traceback.format_exc()
    finally:
        job_state["running"] = False


def start_job(
    mode,
    model,
    model_name_single,
    model_name_multi,
    experiment,
    training_name,
    mlflow_input,
    cvat_url_input,
    cvat_username_input,
    cvat_password_input,
    train_split,
    cvat_task_ids,
    device,
    lr_fixed, lr_min, lr_max,
    batch_fixed, batch_min, batch_max,
    epochs_fixed, epochs_min, epochs_max,
    n_trials,
    aug_hflip_fixed, aug_hflip_min, aug_hflip_max,
    aug_vflip_fixed, aug_vflip_min, aug_vflip_max,
    aug_scale_fixed, aug_scale_min, aug_scale_max,
    aug_brightness_fixed, aug_brightness_min, aug_brightness_max,
    aug_saturation_fixed, aug_saturation_min, aug_saturation_max,
    aug_gaussianblur_fixed, aug_gaussianblur_min, aug_gaussianblur_max,
    aug_motionblur_fixed, aug_motionblur_min, aug_motionblur_max,
    aug_gaussiannoise_fixed, aug_gaussiannoise_min, aug_gaussiannoise_max,
    aug_isonoise_fixed, aug_isonoise_min, aug_isonoise_max,
):
    if job_state["running"]:
        return (
            "⚠️ Ya hay un job en ejecución",
            f"{job_state['stage']} ({job_state['percent']}%)",
            job_state["logs"],
            json.dumps(job_state["mlflow"], indent=2, ensure_ascii=False) if job_state["mlflow"] else "",
            "",
            render_pipeline_html(),
        )

    config = build_config(
        mode,
        model,
        model_name_single,
        model_name_multi,
        experiment,
        training_name,
        mlflow_input,
        cvat_url_input,
        cvat_username_input,
        cvat_password_input,
        train_split,
        cvat_task_ids,
        device,
        lr_fixed, lr_min, lr_max,
        batch_fixed, batch_min, batch_max,
        epochs_fixed, epochs_min, epochs_max,
        n_trials,
        aug_hflip_fixed, aug_hflip_min, aug_hflip_max,
        aug_vflip_fixed, aug_vflip_min, aug_vflip_max,
        aug_scale_fixed, aug_scale_min, aug_scale_max,
        aug_brightness_fixed, aug_brightness_min, aug_brightness_max,
        aug_saturation_fixed, aug_saturation_min, aug_saturation_max,
        aug_gaussianblur_fixed, aug_gaussianblur_min, aug_gaussianblur_max,
        aug_motionblur_fixed, aug_motionblur_min, aug_motionblur_max,
        aug_gaussiannoise_fixed, aug_gaussiannoise_min, aug_gaussiannoise_max,
        aug_isonoise_fixed, aug_isonoise_min, aug_isonoise_max,
    )

    safe_config = dict(config)
    if safe_config.get("cvat_password"):
        safe_config["cvat_password"] = "********"

    thread = threading.Thread(target=training_worker, args=(config, mode), daemon=True)
    thread.start()

    return (
        "🚀 Job lanzado",
        f"{job_state['stage']} ({job_state['percent']}%)",
        "",
        "",
        json.dumps(safe_config, indent=2, ensure_ascii=False),
        render_pipeline_html(),
    )


def refresh_status():
    mlflow_data = ""
    if job_state["mlflow"]:
        mlflow_data = json.dumps(job_state["mlflow"], indent=2, ensure_ascii=False)

    if job_state["error"]:
        status = f"❌ {job_state['error']}"
    elif job_state["running"]:
        status = f"⏳ Ejecutando - etapa: {job_state['stage']}"
    elif job_state["done"]:
        status = job_state["result"] or "✅ Finalizado"
    else:
        status = "Idle"

    progress = f"{job_state['stage']} ({job_state['percent']}%)"
    return status, progress, job_state["logs"], mlflow_data, render_pipeline_html()


def update_bundles(model):
    bundles = MODEL_BUNDLES[model]
    return (
        gr.Dropdown(choices=bundles, value=bundles[0]),
        gr.CheckboxGroup(choices=bundles, value=[bundles[0]]),
    )


def toggle_optimize_fields(mode):
    is_optimize = mode == "optimize"

    return (
        gr.update(visible=not is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),

        gr.update(visible=not is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),

        gr.update(visible=not is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),

        gr.update(visible=not is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),

        gr.update(visible=not is_optimize),
        gr.update(visible=not is_optimize),
        gr.update(visible=not is_optimize),
        gr.update(visible=not is_optimize),
        gr.update(visible=not is_optimize),
        gr.update(visible=not is_optimize),
        gr.update(visible=not is_optimize),
        gr.update(visible=not is_optimize),
        gr.update(visible=not is_optimize),

        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
        gr.update(visible=is_optimize),
    )


def detect_labels_ui(cvat_task_ids, cvat_url_input, cvat_username_input, cvat_password_input):
    try:
        apply_cvat_env(cvat_url_input, cvat_username_input, cvat_password_input)

        task_ids = parse_int_list(cvat_task_ids)
        if not task_ids:
            return "Sin tasks", "Sin detectar"

        metadata = detect_task_metadata(task_ids)
        labels_text = json.dumps(metadata["labels"], indent=2, ensure_ascii=False)
        return labels_text, str(metadata["num_classes"])
    except Exception as e:
        return f"Error detectando labels: {e}", "Error"


def make_aug_block(title, fixed_default=0.0, min_default=0.0, max_default=0.5):
    with gr.Group():
        gr.Markdown(f"### {title}")
        fixed = gr.Number(value=fixed_default, label=f"{title} (train)", visible=True)
        with gr.Row():
            min_v = gr.Number(value=min_default, label=f"{title} min (optimize)", visible=False)
            max_v = gr.Number(value=max_default, label=f"{title} max (optimize)", visible=False)
    return fixed, min_v, max_v


def run_inference_ui(image, model_dir, min_area, distance_m, horizontal_fov_deg):
    try:
        if image is None:
            return (
                "❌ Sube una imagen antes de ejecutar inferencia.",
                0,
                "0.00 %",
                "0.0 cm",
                "0.0 cm",
                "0.0 cm",
                None,
                None,
                "{}",
            )

        min_area = int(min_area) if min_area is not None else 25
        distance_m = float(distance_m) if distance_m is not None else 2.0
        horizontal_fov_deg = float(horizontal_fov_deg) if horizontal_fov_deg is not None else 70.0
        model_dir = model_dir.strip() if model_dir and model_dir.strip() else DEFAULT_MODEL_DIR

        result = run_orange_inference(
            image=image,
            model_dir=model_dir,
            min_area=min_area,
            distance_m=distance_m,
            horizontal_fov_deg=horizontal_fov_deg,
        )

        stats = result["stats"]
        components_preview = result["components"][:50]
        details = {
            "model_dir": model_dir,
            "min_area_px": min_area,
            "orange_count": result["count"],
            "orange_pixels": stats["orange_pixels"],
            "coverage_pct": round(stats["coverage_pct"], 4),
            "distance_m_assumed": distance_m,
            "horizontal_fov_deg_assumed": horizontal_fov_deg,
            "cm_per_px_estimated": round(stats["cm_per_px_estimated"], 6),
            "mean_sphere_diameter_cm_estimated": round(stats["mean_diameter_cm_estimated"], 2),
            "min_sphere_diameter_cm_estimated": round(stats["min_diameter_cm_estimated"], 2),
            "max_sphere_diameter_cm_estimated": round(stats["max_diameter_cm_estimated"], 2),
            "mean_sphere_diameter_px": round(stats["mean_diameter_px"], 2),
            "diameter_method": stats.get("diameter_method"),
            "components_preview": components_preview,
        }

        status = "✅ Inferencia completada"
        return (
            status,
            result["count"],
            f'{stats["coverage_pct"]:.2f} %',
            f'{stats["mean_diameter_cm_estimated"]:.1f} cm',
            f'{stats["min_diameter_cm_estimated"]:.1f} cm',
            f'{stats["max_diameter_cm_estimated"]:.1f} cm',
            result["mask_image"],
            result["overlay"],
            json.dumps(details, indent=2, ensure_ascii=False),
        )

    except Exception as e:
        return (
            f"❌ Error ejecutando inferencia: {e}",
            0,
            "0.00 %",
            "0 px",
            "0 px",
            "0 px",
            None,
            None,
            traceback.format_exc(),
        )


with gr.Blocks(title="Segmentation Training UI", css=CSS, theme=gr.themes.Base()) as demo:
    gr.HTML("""
    <div class="gft-header">
        <div class="gft-logo">GFT</div>
        <div class="gft-vline"></div>
        <div>
            <p class="gft-title">Segmentation Training UI</p>
            <p class="gft-sub">CVAT · Training · Optuna · MLflow</p>
        </div>
        <div class="gft-spacer"></div>
        <div class="gft-badge">BETA</div>
    </div>
    """)

    pipeline_status = gr.HTML(render_pipeline_html(), elem_classes=["pipeline-bar"], elem_id="pipeline-status")

    with gr.Tabs():
        with gr.Tab("Train / Optimize"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Configuración")

                    with gr.Row():
                        mode = gr.Radio(["train", "optimize"], value="train", label="Modo")
                        model = gr.Dropdown(list(MODEL_BUNDLES.keys()), value="SegFormer", label="Modelo")
                        device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device")

                    with gr.Row():
                        model_name_single = gr.Dropdown(
                            choices=MODEL_BUNDLES["SegFormer"],
                            value=MODEL_BUNDLES["SegFormer"][0],
                            label="Modelo / Bundle (train)"
                        )
                        model_name_multi = gr.CheckboxGroup(
                            choices=MODEL_BUNDLES["SegFormer"],
                            value=[MODEL_BUNDLES["SegFormer"][0]],
                            label="Bundles a optimizar (optimize)",
                            visible=False
                        )

                    with gr.Row():
                        experiment = gr.Textbox(value="experiment_1", label="Experiment")
                        training_name = gr.Textbox(value="run_1", label="Training name")

                    mlflow_input = gr.Textbox(
                        label="MLflow URI (opcional, si se deja vacío usa .env)",
                        placeholder="http://localhost:5000"
                    )

                    with gr.Accordion("CVAT connection", open=False):
                        cvat_url_input = gr.Textbox(
                            label="CVAT URL (opcional, si se deja vacío usa .env)",
                            placeholder="http://<cvat_host>"
                        )
                        cvat_username_input = gr.Textbox(
                            label="CVAT Username (opcional, si se deja vacío usa .env)",
                            placeholder="usuario"
                        )
                        cvat_password_input = gr.Textbox(
                            label="CVAT Password (opcional, si se deja vacío usa .env)",
                            type="password"
                        )

                    with gr.Row():
                        cvat_task_ids = gr.Textbox(value="43", label="CVAT Task IDs (separados por comas)")
                        train_split = gr.Number(value=0.8, label="Train split")

                    with gr.Row():
                        detect_labels_button = gr.Button("🔍 Detectar labels automáticamente", variant="secondary")
                        detected_num_classes = gr.Textbox(label="Num classes detectado", interactive=False)

                    detected_labels = gr.Code(label="Labels detectadas", language="json")

                    gr.Markdown("## Hiperparámetros")

                    with gr.Group():
                        gr.Markdown("### Learning rate")
                        lr_fixed = gr.Number(value=0.001, label="Learning rate (train)", visible=True)
                        with gr.Row():
                            lr_min = gr.Number(value=0.0001, label="Learning rate min (optimize)", visible=False)
                            lr_max = gr.Number(value=0.001, label="Learning rate max (optimize)", visible=False)

                    with gr.Group():
                        gr.Markdown("### Batch size")
                        batch_fixed = gr.Number(value=2, label="Batch size (train)", visible=True)
                        with gr.Row():
                            batch_min = gr.Number(value=1, label="Batch size min (optimize)", visible=False)
                            batch_max = gr.Number(value=4, label="Batch size max (optimize)", visible=False)

                    with gr.Group():
                        gr.Markdown("### Epochs")
                        epochs_fixed = gr.Number(value=10, label="Epochs (train)", visible=True)
                        with gr.Row():
                            epochs_min = gr.Number(value=1, label="Epochs min (optimize)", visible=False)
                            epochs_max = gr.Number(value=5, label="Epochs max (optimize)", visible=False)

                    n_trials = gr.Number(value=10, label="n_trials", visible=False)

                    with gr.Accordion("Augmentations", open=False):
                        aug_hflip_fixed, aug_hflip_min, aug_hflip_max = make_aug_block("Horizontal Flip", 0.0, 0.0, 0.5)
                        aug_vflip_fixed, aug_vflip_min, aug_vflip_max = make_aug_block("Vertical Flip", 0.0, 0.0, 0.5)
                        aug_scale_fixed, aug_scale_min, aug_scale_max = make_aug_block("Scale", 0.0, 0.0, 0.5)
                        aug_brightness_fixed, aug_brightness_min, aug_brightness_max = make_aug_block("Brightness", 0.0, 0.0, 0.5)
                        aug_saturation_fixed, aug_saturation_min, aug_saturation_max = make_aug_block("Saturation", 0.0, 0.0, 0.5)
                        aug_gaussianblur_fixed, aug_gaussianblur_min, aug_gaussianblur_max = make_aug_block("Gaussian Blur", 0.0, 0.0, 0.3)
                        aug_motionblur_fixed, aug_motionblur_min, aug_motionblur_max = make_aug_block("Motion Blur", 0.0, 0.0, 0.2)
                        aug_gaussiannoise_fixed, aug_gaussiannoise_min, aug_gaussiannoise_max = make_aug_block("Gaussian Noise", 0.0, 0.0, 0.3)
                        aug_isonoise_fixed, aug_isonoise_min, aug_isonoise_max = make_aug_block("ISO Noise", 0.0, 0.0, 0.2)

                    with gr.Row():
                        start_button = gr.Button("🚀 Ejecutar", variant="primary")
                        refresh_button = gr.Button("🔄 Refrescar estado", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown("## Estado y resultados")

                    status_box = gr.Textbox(label="Estado")
                    progress_box = gr.Textbox(label="Progreso / etapa")

                    with gr.Tabs():
                        with gr.Tab("Logs"):
                            logs_box = gr.Textbox(label="Logs", lines=24)

                        with gr.Tab("MLflow"):
                            mlflow_box = gr.Code(label="Última run en MLflow", language="json")

                        with gr.Tab("Config"):
                            config_preview = gr.Code(label="Config generado", language="json")



        with gr.Tab("Inference"):
            gr.Markdown("## Inferencia del modelo final")
            gr.Markdown(
                "Sube una imagen de un naranjo. El modelo final MaskFormer Tiny predice la máscara de naranjas "
                "calcula un conteo aproximado mediante componentes conectados y estima el diámetro esférico aparente de los frutos."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    inference_image = gr.Image(
                        type="pil",
                        label="Imagen de entrada"
                    )
                    inference_model_dir = gr.Textbox(
                        value=DEFAULT_MODEL_DIR,
                        label="Carpeta del modelo final"
                    )
                    with gr.Accordion("Parámetros avanzados", open=False):
                        inference_min_area = gr.Number(
                            value=25,
                            label="Área mínima para contar una naranja (px)"
                        )
                        inference_distance_m = gr.Number(
                            value=2.0,
                            label="Distancia aproximada al árbol (m)"
                        )
                        inference_fov_deg = gr.Number(
                            value=70.0,
                            label="Campo de visión horizontal estimado (º)"
                        )
                        gr.Markdown(
                            "Las medidas en cm son aproximadas y asumen que las naranjas están a la distancia indicada."
                        )

                    inference_button = gr.Button("🍊 Ejecutar inferencia", variant="primary")

                with gr.Column(scale=1):
                    inference_status = gr.Textbox(label="Estado")
                    inference_count = gr.Number(label="Naranjas detectadas", precision=0)
                    inference_coverage = gr.Textbox(label="Cobertura naranja")
                    inference_mean_area = gr.Textbox(label="Diámetro esférico medio estimado")
                    inference_min_area_detected = gr.Textbox(label="Diámetro esférico mínimo estimado")
                    inference_max_area_detected = gr.Textbox(label="Diámetro esférico máximo estimado")

            with gr.Row():
                inference_mask = gr.Image(label="Máscara predicha")
                inference_overlay = gr.Image(label="Overlay")

            inference_details = gr.Textbox(label="Detalles de inferencia", lines=12)
    model.change(update_bundles, inputs=model, outputs=[model_name_single, model_name_multi], api_name=False)

    mode.change(
        toggle_optimize_fields,
        inputs=mode,
        outputs=[
            model_name_single,
            model_name_multi,
            n_trials,

            lr_fixed,
            lr_min,
            lr_max,

            batch_fixed,
            batch_min,
            batch_max,

            epochs_fixed,
            epochs_min,
            epochs_max,

            aug_hflip_fixed,
            aug_vflip_fixed,
            aug_scale_fixed,
            aug_brightness_fixed,
            aug_saturation_fixed,
            aug_gaussianblur_fixed,
            aug_motionblur_fixed,
            aug_gaussiannoise_fixed,
            aug_isonoise_fixed,

            aug_hflip_min,
            aug_hflip_max,
            aug_vflip_min,
            aug_vflip_max,
            aug_scale_min,
            aug_scale_max,
            aug_brightness_min,
            aug_brightness_max,
            aug_saturation_min,
            aug_saturation_max,
            aug_gaussianblur_min,
            aug_gaussianblur_max,
            aug_motionblur_min,
            aug_motionblur_max,
            aug_gaussiannoise_min,
            aug_gaussiannoise_max,
            aug_isonoise_min,
            aug_isonoise_max,
        ],
        api_name=False,
    )

    detect_labels_button.click(
        fn=detect_labels_ui,
        inputs=[
            cvat_task_ids,
            cvat_url_input,
            cvat_username_input,
            cvat_password_input,
        ],
        outputs=[detected_labels, detected_num_classes],
        api_name=False
    )

    start_button.click(
        fn=start_job,
        inputs=[
            mode,
            model,
            model_name_single,
            model_name_multi,
            experiment,
            training_name,
            mlflow_input,
            cvat_url_input,
            cvat_username_input,
            cvat_password_input,
            train_split,
            cvat_task_ids,
            device,
            lr_fixed, lr_min, lr_max,
            batch_fixed, batch_min, batch_max,
            epochs_fixed, epochs_min, epochs_max,
            n_trials,
            aug_hflip_fixed, aug_hflip_min, aug_hflip_max,
            aug_vflip_fixed, aug_vflip_min, aug_vflip_max,
            aug_scale_fixed, aug_scale_min, aug_scale_max,
            aug_brightness_fixed, aug_brightness_min, aug_brightness_max,
            aug_saturation_fixed, aug_saturation_min, aug_saturation_max,
            aug_gaussianblur_fixed, aug_gaussianblur_min, aug_gaussianblur_max,
            aug_motionblur_fixed, aug_motionblur_min, aug_motionblur_max,
            aug_gaussiannoise_fixed, aug_gaussiannoise_min, aug_gaussiannoise_max,
            aug_isonoise_fixed, aug_isonoise_min, aug_isonoise_max,
        ],
        outputs=[status_box, progress_box, logs_box, mlflow_box, config_preview, pipeline_status],
        api_name=False
    )


    inference_button.click(
        fn=run_inference_ui,
        inputs=[
            inference_image,
            inference_model_dir,
            inference_min_area,
            inference_distance_m,
            inference_fov_deg,
        ],
        outputs=[
            inference_status,
            inference_count,
            inference_coverage,
            inference_mean_area,
            inference_min_area_detected,
            inference_max_area_detected,
            inference_mask,
            inference_overlay,
            inference_details,
        ],
        api_name=False
    )

    refresh_button.click(
        fn=refresh_status,
        inputs=[],
        outputs=[status_box, progress_box, logs_box, mlflow_box, pipeline_status],
        api_name=False
    )

    demo.load(
        fn=refresh_status,
        inputs=[],
        outputs=[status_box, progress_box, logs_box, mlflow_box, pipeline_status],
        every=2,
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)