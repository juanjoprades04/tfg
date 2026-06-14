import os
import shutil
import zipfile
from typing import List, Union, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from xml.etree import ElementTree as ET

from magna_utils.cvat_utils import get_cvat_client, get_task_labels_metadata


def download_cvat_tasks(task_ids: Union[int, List[int]], output_dir: str = "temp"):
    client = get_cvat_client()
    if client is None:
        raise RuntimeError("CVAT client is not available")

    if isinstance(task_ids, int):
        task_ids = [task_ids]

    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for task_id in task_ids:
        print(f"\nProcessing task {task_id}...")

        try:
            task = client.tasks.retrieve(task_id)

            zip_filename = f"task_{task_id}_dataset.zip"
            zip_path = os.path.join(output_dir, zip_filename)
            extract_to = os.path.join(output_dir, f"task_{task_id}_extracted")

            print("  Downloading dataset...")
            task.export_dataset(
                format_name="CVAT for images 1.1",
                filename=zip_path,
                include_images=True,
            )

            os.makedirs(extract_to, exist_ok=True)

            print("  Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

            print(f"  ✓ Task {task_id} extracted to: {extract_to}")

            if os.path.exists(zip_path):
                os.remove(zip_path)

            print(f"  Moving extracted images into {images_dir}...")

            for root, _, files in os.walk(extract_to):
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_exts:
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(images_dir, file)

                        if os.path.exists(dst_path):
                            base, ext = os.path.splitext(file)
                            counter = 1
                            while os.path.exists(dst_path):
                                dst_path = os.path.join(images_dir, f"{base}_{counter}{ext}")
                                counter += 1

                        shutil.move(src_path, dst_path)

            print("  ✓ Images moved successfully.")

        except Exception as e:
            print(f"  ✗ Error processing task {task_id}: {e}")
            continue

    print(f"\nAll tasks processed. Output directory: {output_dir}")


def load_cvat_xml(xml_path: str):
    tree = ET.parse(xml_path)
    return tree.getroot()


def build_label_mapping(task_ids: List[int]) -> Dict[str, int]:
    label_names = get_task_labels_metadata(task_ids)

    if not label_names:
        raise ValueError("No shape labels found in CVAT tasks")

    return {label_name: idx + 1 for idx, label_name in enumerate(label_names)}


def get_polygons_from_cvat_xml(xml_root, images_dir: str):
    polygons = []

    for image in xml_root.findall("image"):
        image_name = image.get("name")
        if not image_name:
            continue

        filename = os.path.basename(image_name)
        image_path = os.path.join(images_dir, filename)

        for polygon in image.findall(".//polygon"):
            polygons.append(
                {
                    "label": polygon.get("label", "").lower().strip(),
                    "points": polygon.get("points"),
                    "width": int(image.get("width")),
                    "height": int(image.get("height")),
                    "image_path": image_path,
                    "image_filename": filename,
                }
            )

    return pd.DataFrame(polygons)


def draw_polygon(mask: np.ndarray, points: str, label_value: int):
    pts = np.array(
        [tuple(map(lambda x: int(round(float(x))), p.split(","))) for p in points.split(";")],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], label_value)


def save_mask(array: np.ndarray, file_path: str):
    mask_image = Image.fromarray(np.uint8(array))
    mask_image.save(file_path)


def extract_annotations_to_masks(
    temp_dir: str,
    output_dir: str,
    task_ids: Union[int, List[int]],
):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(task_ids, int):
        task_ids = [task_ids]

    images_dir = os.path.join(temp_dir, "images")
    label_mapping = build_label_mapping(task_ids)

    print(f"Dynamic label mapping: {label_mapping}")

    xml_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))

    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {temp_dir}")

    image_masks = []

    for xml_file in xml_files:
        print(f"\nProcessing: {xml_file}")

        root = load_cvat_xml(xml_file)
        polygons_df = get_polygons_from_cvat_xml(root, images_dir)

        if polygons_df.empty:
            continue

        grouped_annotations = polygons_df.groupby("image_filename")

        for image_filename, group in grouped_annotations:
            image_shape = (group["height"].iloc[0], group["width"].iloc[0])
            mask = np.zeros(image_shape, dtype=np.uint8)

            for _, row in group.iterrows():
                label_name = row["label"]
                if label_name not in label_mapping:
                    continue
                draw_polygon(mask, row["points"], label_mapping[label_name])

            base_name = os.path.splitext(image_filename)[0]
            mask_path = os.path.join(output_dir, f"{base_name}.png")
            save_mask(mask, mask_path)

            image_masks.append(
                {
                    "image_path": os.path.join(images_dir, image_filename),
                    "label_path": mask_path,
                }
            )

    print(image_masks)
    return image_masks











