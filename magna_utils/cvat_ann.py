from typing import Optional, List, Dict, Any, Tuple, Literal
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from magna_utils import config
from magna_utils.cvat_utils import get_cvat_client, get_all_labels_from_task, add_label_to_project
import logging
import zipfile
import random
import os

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

AnnotationType = Literal["tag", "shape"]
ShapeType = Literal["box", "polygon", "polyline", "points", "ellipse", "cuboid", "skeleton"]


class XMLAnnotationManager:
    def __init__(self, project_id: int, task_id: int, xml_path: Optional[str] = None) -> None:
        """
        Initialize the XML Annotation Manager.

        Args:
            task_id: CVAT task ID
            xml_path: Path to existing XML file. If None, will download from CVAT.
        """
        self.client = get_cvat_client()
        self.task_id: int = task_id
        self.project_id: int = project_id
        self.xml_path: str = xml_path or f"task_{task_id}_annotations.xml"

        if xml_path and Path(xml_path).exists():
            self.tree: ET.ElementTree = ET.parse(xml_path)
        else:
            self._download_annotations()

        self.root: ET.Element = self.tree.getroot()

    def _download_annotations(self) -> None:
        """Download annotations from CVAT in XML format."""
        logger.info(f"Downloading annotations for task {self.task_id}")
        os.mkdir("temp/add_labels")
        try:
            task = self.client.tasks.retrieve(self.task_id)
            task.export_dataset(
                    format_name="CVAT for images 1.1",
                    filename=f'temp/add_labels/{self.task_id}.zip',
                    include_images=False,
                )

            with zipfile.ZipFile(f'temp/add_labels/{self.task_id}.zip', "r") as zip_ref:
                    zip_ref.extractall(f'temp/add_labels/{self.task_id}')

            self.tree = ET.parse(f'temp/add_labels/{self.task_id}/annotations.xml')
            logger.info(f"Annotations saved")

        except Exception as e:
            logger.error(f"Task {self.task_id} does not exist.")

    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save the modified XML annotations to file.

        Args:
            output_path: Path to save XML file. If None, overwrites the original.
        """
        save_path = output_path or self.xml_path
        self.tree.write(save_path, encoding='unicode', xml_declaration=True)
        logger.info(f"Annotations saved to {save_path}")

    def upload_annotations(self) -> None:
        """Upload the modified XML annotations back to CVAT."""
        import inspect
        print(inspect.signature(self.client.tasks.api.create_annotations))
        logger.info(f"Uploading annotations to task {self.task_id}")

        self.client.tasks.retrieve(self.task_id).import_annotations(
            format_name="CVAT 1.1",
            filename="temp/add_labels/modified_annotations.xml"
        )

        logger.info("Annotations uploaded successfully")

    def _get_image_element(self, frame: int) -> ET.Element:
        """
        Get <image> element for the specified frame.

        Args:
            frame: Frame number

        Returns:
            The image element for the specified frame
        """
        # Search for existing image element
        for image in self.root.findall('image'):
            if int(image.get('id', -1)) == frame:
                return image


    def add_tag_to_frame(
        self,
        frame: int,
        label: str,
        attributes: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a tag to a specific frame in the XML.

        Args:
            frame: Frame number
            label: Label name for the tag
            attributes: Optional dictionary of attribute name-value pairs
        """
        task_labels = get_all_labels_from_task(self.task_id, "tag")

        label_names = [label["name"] for label in task_labels]


        if label not in label_names:
            random_label_color = self.random_hex_color()
            add_label_to_project(self.project_id, label, random_label_color, "tag")

        image = self._get_image_element(frame)

        # Check if tag already exists
        for tag in image.findall('tag'):
            if tag.get('label') == label:
                logger.warning(f"Tag '{label}' already exists on frame {frame}")
                return

        # Create tag element
        tag = ET.SubElement(image, 'tag')
        tag.set('label', label)

        # Add attributes if provided
        if attributes:
            for attr_name, attr_value in attributes.items():
                attr = ET.SubElement(tag, 'attribute')
                attr.set('name', attr_name)
                attr.text = attr_value

        logger.info(f"Added tag '{label}' to frame {frame}")

    def add_points_to_frame(
        self,
        frame: int,
        label: str,
        points: List[Tuple[float, float]],
        attributes: Optional[Dict[str, str]] = None,
        occluded: bool = False,
        z_order: int = 0
    ) -> None:
        """
        Add points annotation to a specific frame in the XML.

        Args:
            frame: Frame number
            label: Label name for the points
            points: List of (x, y) coordinate tuples
            attributes: Optional dictionary of attribute name-value pairs
            occluded: Whether the points are occluded
            z_order: Drawing order (higher values on top)
        """
        task_labels = get_all_labels_from_task(self.task_id, "points")

        label_names = [label["name"] for label in task_labels]

        if label not in label_names:
            random_label_color = self.random_hex_color()
            add_label_to_project(project_id = self.project_id, label_name = label,
                                 color = random_label_color, label_type = "points")

        image = self._get_image_element(frame)

        # Create points element
        points_elem = ET.SubElement(image, 'points')
        points_elem.set('label', label)
        points_elem.set('occluded', '1' if occluded else '0')
        points_elem.set('z_order', str(z_order))

        # Format points as "x1,y1;x2,y2;x3,y3;..."
        points_str = ';'.join(f'{x:.2f},{y:.2f}' for x, y in points)
        points_elem.set('points', points_str)

        # Add attributes if provided
        if attributes:
            for attr_name, attr_value in attributes.items():
                attr = ET.SubElement(points_elem, 'attribute')
                attr.set('name', attr_name)
                attr.text = attr_value

        logger.info(f"Added points '{label}' with {len(points)} keypoints to frame {frame}")

    def add_polygon_to_frame(
        self,
        frame: int,
        label: str,
        points: List[Tuple[float, float]],
        attributes: Optional[Dict[str, str]] = None,
        occluded: bool = False,
        z_order: int = 0
    ) -> None:
        """
        Add polygon annotation to a specific frame in the XML.

        Args:
            frame: Frame number
            label: Label name for the polygon
            points: List of (x, y) coordinate tuples defining the polygon vertices
            attributes: Optional dictionary of attribute name-value pairs
            occluded: Whether the polygon is occluded
            z_order: Drawing order (higher values on top)
        """
        task_labels = get_all_labels_from_task(self.task_id, "polygon")

        label_names = [label["name"] for label in task_labels]

        if label not in label_names:
            random_label_color = self.random_hex_color()
            add_label_to_project(project_id = self.project_id, label_name = label,
                                 color = random_label_color, label_type = "polygon")

        image = self._get_image_element(frame)

        # Create polygon element
        polygon = ET.SubElement(image, 'polygon')
        polygon.set('label', label)
        polygon.set('occluded', '1' if occluded else '0')
        polygon.set('z_order', str(z_order))

        # Format points as "x1,y1;x2,y2;x3,y3;..."
        points_str = ';'.join(f'{x:.2f},{y:.2f}' for x, y in points)
        polygon.set('points', points_str)

        # Add attributes if provided
        if attributes:
            for attr_name, attr_value in attributes.items():
                attr = ET.SubElement(polygon, 'attribute')
                attr.set('name', attr_name)
                attr.text = attr_value

        logger.info(f"Added polygon '{label}' with {len(points)} vertices to frame {frame}")

    def add_box_to_frame(
        self,
        frame: int,
        label: str,
        xtl: float,
        ytl: float,
        xbr: float,
        ybr: float,
        attributes: Optional[Dict[str, str]] = None,
        occluded: bool = False,
        z_order: int = 0
    ) -> None:
        """
        Add bounding box annotation to a specific frame in the XML.

        Args:
            frame: Frame number
            label: Label name for the box
            xtl: X coordinate of top-left corner
            ytl: Y coordinate of top-left corner
            xbr: X coordinate of bottom-right corner
            ybr: Y coordinate of bottom-right corner
            attributes: Optional dictionary of attribute name-value pairs
            occluded: Whether the box is occluded
            z_order: Drawing order (higher values on top)
        """

        task_labels = get_all_labels_from_task(self.task_id, "box")
        logger.info(task_labels)

        label_names = [label["name"] for label in task_labels]
        logger.info(label_names)
        logger.info(label not in label_names)

        if label not in label_names:
            random_label_color = self.random_hex_color()
            add_label_to_project(project_id = self.project_id, label_name = label,
                                 color = random_label_color, label_type = "box")

        image = self._get_image_element(frame)

        # Create box element
        box = ET.SubElement(image, 'box')
        box.set('label', label)
        box.set('occluded', '1' if occluded else '0')
        box.set('z_order', str(z_order))
        box.set('xtl', f'{xtl:.2f}')
        box.set('ytl', f'{ytl:.2f}')
        box.set('xbr', f'{xbr:.2f}')
        box.set('ybr', f'{ybr:.2f}')

        # Add attributes if provided
        if attributes:
            for attr_name, attr_value in attributes.items():
                attr = ET.SubElement(box, 'attribute')
                attr.set('name', attr_name)
                attr.text = attr_value

        logger.info(f"Added box '{label}' to frame {frame}")


    def list_all_labels(self) -> Dict[str, List[str]]:
        """
        List all unique labels in the annotations by type.

        Returns:
            Dictionary mapping annotation types to lists of label names
        """
        labels: Dict[str, set] = {
            'tags': set(),
            'boxes': set(),
            'polygons': set(),
            'points': set()
        }

        for image in self.root.findall('image'):
            for tag in image.findall('tag'):
                labels['tags'].add(tag.get('label'))
            for box in image.findall('box'):
                labels['boxes'].add(box.get('label'))
            for polygon in image.findall('polygon'):
                labels['polygons'].add(polygon.get('label'))
            for points in image.findall('points'):
                labels['points'].add(points.get('label'))

        return {k: sorted(list(v)) for k, v in labels.items()}

    @staticmethod
    def random_hex_color():
        return f"#{random.randint(0, 0xFFFFFF):06X}"

    def add_annotations(self, ann_data):
        for data in ann_data:
            frame = data.get("frame")
            tags = data.get("tags")
            polygons = data.get("polygons")
            points = data.get("points")

            if tags:
                for tag in tags:
                    self.add_tag_to_frame(
                    frame=frame,
                    label=tag
                )
            else:
                logger.info("No tags found for this frame.")
                continue

            if polygons:
                for polygon in polygons:
                    label = polygon.get("label")
                    poly_points = polygon.get("points")
                    self.add_polygon_to_frame(
                        frame=frame,
                        label=label,
                        points=poly_points,
                        attributes={},
                        occluded=False,
                        z_order=1
                    )
            else:
                logger.info("No polygons found for this frame.")
                continue

            if points:
                for point in points:
                    label = point.get("label")
                    points = point.get("points")
                    self.add_points_to_frame(
                        frame=frame,
                        label=label,
                        points=points,
                        attributes={},
                        occluded=False,
                        z_order=1
                    )
            else:
                logger.info("No points found for this frame.")
                continue


if __name__ == "__main__":

    # Initialize manager (downloads XML from CVAT)
    manager = XMLAnnotationManager(task_id=655, project_id = 60)
    temp_dir = Path("temp/add_labels")
    dummy_data = [
                    {
                        "frame": 5,
                        "tags": ["false positive", "incorrect points", "incorrect mask"],
                        "polygons":[
                            {
                                "label":"weld",
                                "points":[(461, 285), (445, 296), (437, 310), (436, 325), (430, 357), (412, 407), (403, 437), (390, 464), (378, 482), (371, 498), (368, 516), (361, 543), (360, 571), (358, 588), (362, 601), (371, 608), (380, 615), (394, 619), (419, 613), (452, 610), (499, 611), (534, 613), (571, 615), (600, 615), (616, 611), (633, 597), (619, 592), (604, 578), (597, 568), (595, 552), (591, 536), (587, 525), (580, 509), (562, 489), (552, 468), (535, 432), (520, 401), (507, 380), (494, 363), (477, 347), (471, 334), (467, 320), (463, 305)]
                            }
                        ],
                        "points":[
                                {
                                    "label": "point_1",
                                    "points":[(630.54,595.57)]
                                },
                                {
                                    "label": "point_2",
                                    "points":[(461.70,285.80)]
                                }
                            ],
                    },
                     {
                        "frame": 17,
                        "tags": ["false positive", "incorrect points", "incorrect mask"],
                        "polygons":[
                            {
                                "label":"weld",
                                "points":[(461, 285), (445, 296), (437, 310), (436, 325), (430, 357), (412, 407), (403, 437), (390, 464), (378, 482), (371, 498), (368, 516), (361, 543), (360, 571), (358, 588), (362, 601), (371, 608), (380, 615), (394, 619), (419, 613), (452, 610), (499, 611), (534, 613), (571, 615), (600, 615), (616, 611), (633, 597), (619, 592), (604, 578), (597, 568), (595, 552), (591, 536), (587, 525), (580, 509), (562, 489), (552, 468), (535, 432), (520, 401), (507, 380), (494, 363), (477, 347), (471, 334), (467, 320), (463, 305)]
                            }
                        ],
                        "points":[
                                {
                                    "label": "point_1",
                                    "points":[(630.54,595.57)]
                                },
                                {
                                    "label": "point_2",
                                    "points":[(461.70,285.80)]
                                }
                            ],
                    }
                ]

    try:
        manager.add_annotations(dummy_data)


        # # Add a bounding box
        # manager.add_box_to_frame(
        #     frame=5,
        #     label="ROI",
        #     xtl=50.0, ytl=100.0,
        #     xbr=200.0, ybr=400.0
        # )

    except Exception as e:
        logger.error(f"Error:{e}")


    finally:

        # Save modified XML
        manager.save(f"{temp_dir}/modified_annotations.xml")

        # Upload back to CVAT
        manager.upload_annotations()

        # List all labels
        all_labels = manager.list_all_labels()
        logger.info(f'The task contains the following_labels: {all_labels}')

        # delete temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


###
# parse json to list of frames [{frame: 0, tags:[], polygons:[], points:[], box: []}]