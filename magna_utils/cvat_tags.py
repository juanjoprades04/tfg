from cvat_sdk.api_client import models
from cvat_sdk.datasets.task_dataset import TaskDataset
from cvat_utils import get_cvat_client
import logging
from typing import Optional, List, Dict, Any, Tuple, Literal
from dataclasses import dataclass

# Configurar el nivel de registro para httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

AnnotationType = Literal["tag", "shape"]
ShapeType = Literal["rectangle", "polygon", "points"]


@dataclass
class Point:
    x: float
    y: float

class AnnotationManager:
    def __init__(self, task_id):
        self.client = get_cvat_client()
        self.dataset = TaskDataset(self.client, task_id, load_annotations=True)
        self.labels = self.dataset.labels
        self.task_id = task_id
    
    def add_tag_to_frame(self, frame, tag_name):
        """
        Add an image-level tag to a specific frame in a CVAT task.

        Args:
            client: CVAT client object
            task_id: int, task id
            frame: int, frame number
            tag_name: str, name of the tag to add

        Returns:
            None
        """

        client = self.client
        labels = self.labels
        task_id = self.task_id

        # 1️⃣ Find the label_id for the tag
        tag_label = None
        for lbl in labels:
            if lbl.get("type") == "tag" and lbl.get("name") == tag_name:
                tag_label = lbl
                break
        if tag_label is None:
            raise ValueError(f"Tag '{tag_name}' not found in task labels.")
        tag_id = tag_label["id"]

        # 2️⃣ Retrieve current annotations from the server
        ann_dict, _ = client.tasks.api.retrieve_annotations(task_id)
        print(ann_dict)

        # 3️⃣ Normalize existing tags to dicts (handle SDK objects or dicts)
        existing_tags = []
        for t in ann_dict.get("tags", []):
            if isinstance(t, dict):
                existing_tags.append(t)
            else:
                existing_tags.append({
                    "frame": t.frame,
                    "label_id": t.label_id,
                    "attributes": getattr(t, "attributes", [])
                })

        # 4️⃣ Avoid duplicate tags on the same frame
        if not any(t["frame"] == frame and t["label_id"] == tag_id for t in existing_tags):
            new_tag = {
                "frame": frame,
                "label_id": tag_id,
                "attributes": []
            }
            existing_tags.append(new_tag)

        # 5️⃣ Update annotations on the server
        client.tasks.api.update_annotations(
            task_id,
            task_annotations_update_request=models.LabeledDataRequest(
                shapes=ann_dict.get("shapes", []),
                tracks=ann_dict.get("tracks", []),
                tags=existing_tags,
                version=ann_dict.get("version", 0)
            )
        )

        logger.info(f"Added tag '{tag_name}' (id={tag_id}) to frame {frame}")

    def add_shape_to_frame(
        self,
        frame: int,
        label_name: str,
        shape_type: ShapeType,
        points: List[float],
        attributes: Optional[List[Dict[str, Any]]] = None,
        occluded: bool = False,
        z_order: int = 0
    ) -> None:
        """
        Add a shape annotation (bounding box, polygon, etc.) to a specific frame.

        Args:
            frame: Frame number to add the shape to
            label_name: Name of the label for this shape
            shape_type: Type of shape (rectangle, polygon, polyline, points, ellipse, cuboid, skeleton)
            points: Flat list of coordinates [x1, y1, x2, y2, ...] specific to the shape type
            attributes: Optional list of attribute dictionaries
            occluded: Whether the object is occluded
            z_order: Drawing order (higher values are drawn on top)

        Raises:
            ValueError: If the label name is not found or is not a shape label
        """
        # 1️⃣ Find the label_id for the shape
        shape_label: Optional[Dict[str, Any]] = None
        for lbl in self.labels:
            if lbl.get("type") in ["any", "rectangle", "polygon", "polyline", "points", "ellipse", "cuboid", "skeleton"] \
               and lbl.get("name") == label_name:
                shape_label = lbl
                break
        if shape_label is None:
            raise ValueError(f"Shape label '{label_name}' not found in task labels.")
        label_id: int = shape_label["id"]

        # 2️⃣ Retrieve current annotations
        ann_dict, _ = self.client.tasks.api.retrieve_annotations(self.task_id)

        # 3️⃣ Create new shape
        new_shape: Dict[str, Any] = {
            "frame": frame,
            "label_id": label_id,
            "type": shape_type,
            "points": points,
            "occluded": occluded,
            "z_order": z_order,
            "attributes": attributes or []
        }

        # 4️⃣ Add to existing shapes
        shapes: List[Dict[str, Any]] = ann_dict.get("shapes", [])
        shapes.append(new_shape)

        # 5️⃣ Update annotations on the server
        self.client.tasks.api.update_annotations(
            self.task_id,
            task_annotations_update_request=models.LabeledDataRequest(
                shapes=shapes,
                tracks=ann_dict.get("tracks", []),
                tags=ann_dict.get("tags", []),
                version=ann_dict.get("version", 0)
            )
        )

        logger.info(f"Added {shape_type} shape '{label_name}' to frame {frame}")

    def add_keypoint_to_frame(
        self,
        frame: int,
        label_name: str,
        keypoints: List[Tuple[float, float]],
        attributes: Optional[List[Dict[str, Any]]] = None,
        occluded: bool = False
    ) -> None:
        """
        Add keypoint annotations to a specific frame.

        Args:
            frame: Frame number to add keypoints to
            label_name: Name of the skeleton/points label
            keypoints: List of (x, y) coordinate tuples
            attributes: Optional list of attribute dictionaries
            occluded: Whether the keypoints are occluded
        """
        # Flatten keypoints to the format expected by CVAT
        points: List[float] = [coord for kp in keypoints for coord in kp]
        
        self.add_shape_to_frame(
            frame=frame,
            label_name=label_name,
            shape_type="points",
            points=points,
            attributes=attributes,
            occluded=occluded
        )

    def remove_tag_from_frame(self, frame, tag_name):
        """
        Remove an image-level tag from a specific frame in a CVAT task.

        Args:
            client: CVAT client object
            dataset: TaskDataset object
            frame: int, frame number
            tag_name: str, name of the tag to remove

        Returns:
            None
        """
        client = self.client
        labels = self.labels
        task_id = self.task_id

        # 1️⃣ Find the label_id for the tag
        tag_label = None
        for lbl in labels:
            if lbl.get("type") == "tag" and lbl.get("name") == tag_name:
                tag_label = lbl
                break
        if tag_label is None:
            raise ValueError(f"Tag '{tag_name}' not found in task labels.")
        tag_id = tag_label["id"]

        # 2️⃣ Retrieve current annotations from the server
        ann_dict, _ = client.tasks.api.retrieve_annotations(task_id)

        # 3️⃣ Normalize existing tags to dicts
        existing_tags = []
        for t in ann_dict.get("tags", []):
            if isinstance(t, dict):
                existing_tags.append(t)
            else:
                existing_tags.append({
                    "frame": t.frame,
                    "label_id": t.label_id,
                    "attributes": getattr(t, "attributes", [])
                })

        # 4️⃣ Filter out the tag for the given frame
        updated_tags = [
            t for t in existing_tags
            if not (t["frame"] == frame and t["label_id"] == tag_id)
        ]

        # 5️⃣ Update annotations on the server
        client.tasks.api.update_annotations(
            task_id,
            task_annotations_update_request=models.LabeledDataRequest(
                shapes=ann_dict.get("shapes", []),
                tracks=ann_dict.get("tracks", []),
                tags=updated_tags,
                version=ann_dict.get("version", 0)
            )
        )

        logger.info(f"Removed tag '{tag_name}' (id={tag_id}) from frame {frame}")




tagger = AnnotationManager(654)

# tagger.add_tag_to_frame(frame=9, tag_name="False Positive")
# tagger.add_tag_to_frame(frame=10, tag_name="False Negative")
tagger.add_shape_to_frame(frame = 10, label_name = "weld", shape_type = "polygon",
                          points = [660.00,558.00,647.00,530.00,
                                    636.00,521.00,144.00,526.00,
                                    82.00,532.00,18.00,531.00,
                                    0.00,533.00,0.00,739.00,
                                    50.00,737.00,88.00,731.00,
                                    124.00,729.00,458.00,728.00,
                                    518.00,725.00,562.00,727.00,
                                    604.00,724.00,671.00,726.00,679.00,716.00,
                                    679.00,708.00,658.00,664.00,662.00,643.00,
                                    657.00,588.00,661.00,573.00])
# tagger.remove_tag_from_frame(frame = 9, tag_name = "bad frame")
# tagger.remove_tag_from_frame(frame = 10, tag_name = "wrong joint")
    

