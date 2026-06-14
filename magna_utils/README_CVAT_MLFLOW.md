# MAGNA Utilities for CVAT & MLflow

## Table of Contents
- [General Information](#general-information)
  - [CVAT](#cvat)
- [CVAT Utilities](#cvat-utilities-cvat_utilspy)
  - [Project and Task Creation](#project-and-task-creation)
    - [`create_cvat_project`](#create_cvat_project)
    - [`create_cvat_task_from_cloud_storage`](#create_cvat_task_from_cloud_storage)
  - [Get Project & Task Details](#get-project--task-details)
    - [`get_task_list`](#get_task_list)
    - [`get_project_list`](#get_project_list)
    - [`get_all_labels_from_project`](#get_all_labels_from_project)
    - [`get_all_labels_from_task`](#get_all_labels_from_task)
  - [Task and Label Management](#task-and-label-management)
    - [`delete_tasks`](#delete_tasks)
    - [`delete_task`](#delete_task)
    - [`add_label_to_project`](#add_label_to_project)
    - [`delete_label_from_project`](#delete_label_from_project)
  - [Usage Examples](#usage-examples)
   - [Typical End-to-End Workflow](#typical-end-to-end-workflow)

- [CVAT Dataset Download and Annotation Processing](#cvat-dataset-download-and-annotation-processing)
  - [Label Mapping](#label-mapping)
  - [Dataset Download Utilities](#dataset-download-utilities)
  - [`download_cvat_tasks`](#download_cvat_tasks)
  - [`load_cvat_xml`](#load_cvat_xml)
  - [`get_polygons_from_cvat_xml`](#get_polygons_from_cvat_xml)
  - [`draw_polygon`](#draw_polygon)
  - [`save_mask`](#save_mask)
  - [`extract_annotations_to_masks`](#extract_annotations_to_masks)
  - [`delete_folder`](#delete_folder)

- [MLflow Utilities](#mlflow-utilities)
  - [`list_experiments`](#list_experiments)
  - [`list_model_versions`](#list_model_versions)
  - [`list_models`](#list_models)
  - [`list_runs`](#list_runs)
  - [`get_mlflow_run`](#get_mlflow_run)
  - [`get_run_url`](#get_run_url)
  - [`delete_run`](#delete_run)
  
  
---

## General Information

### CVAT
CVAT is a tool for managing image datasets and creating high-quality annotations for computer vision tasks. It supports multiple annotation types, including:

- Bounding boxes
- Polygons
- Masks
- Points

Annotations can be organized, edited, and exported in various formats for downstream AI model training.

---

## CVAT Utilities (`cvat_utils.py`)

### Project and Task Creation

#### `create_cvat_project`
Creates a new CVAT project with the following default labels:

- **False Positive**
- **False Negative**

**Parameters**
- `project_name` (`str`): Name of the project to be created.

**Returns**
- `project_id` (`int`): ID of the newly created CVAT project.

---

#### `create_cvat_task_from_cloud_storage`
Creates a CVAT task using files stored in a pre-configured cloud storage (e.g., S3).

**Parameters**
- `project_id` (`int`): Project id where the task will be included.
- `task_name` (`str`): Name of the task.
- `cloud_storage` (`int`): Cloud storage id
- `s3_file_keys` (`List[str]`): Relative paths to files within the cloud storage.

**Returns**
- `project_id` (`int`): ID of the CVAT project associated with the task.
- `task_id` (`int`): ID of the newly created CVAT task.

---
### Get Project & Task Details

#### `get_task_list`
Returns a list of tasks with the following details:
- **task_id**
- **task_name**
- **created_date**
- **last_modified_date**
- **number_of_jobs**
- **labels**

**Parameters**
- `task_id` (`int`): Name of the project to be created.

**Returns**
- `task_list` (`list`): A list of task details

---

#### `get_project_list`
Return a list of dictionaries with project details: 
- **project_id**
- **project_name**
- **created_date**
- **last_modified_date**
- **number_of_tasks**
- **labels**

**Parameters**
None

**Returns**
- `project_list` (`list`): A list of task details

---

#### `get_all_labels_from_project`
Retrieves all labels associated with a given CVAT project.  
Optionally filters labels by annotation type.

**Parameters**
- `project_id` (`int`): ID of the CVAT project.
- `label_type` (`str`, optional):  
  Label type to filter by (e.g., `"polygon"`, `"tag"`).  
  If `None`, all labels are returned.

**Returns**
- `labels` (`List[Label]`): List of CVAT label objects.
- Returns `None` if the CVAT client is not initialized.

---

#### `get_all_labels_from_task`
Retrieves all labels associated with a given CVAT project.  
Optionally filters labels by annotation type.

**Parameters**
- `task_id` (`int`): ID of the CVAT project.
- `label_type` (`str`, optional):  
  Label type to filter by (e.g., `"polygon"`, `"tag"`).  
  If `None`, all labels are returned.

**Returns**
- `labels` (`List[Label]`): List of CVAT label objects.
- Returns `None` if the CVAT client is not initialized.

---

### Task and Label Management

#### `delete_tasks`
Deletes multiple CVAT tasks by their IDs.

**Parameters**
- `task_ids` (`List[int]`): List of CVAT task IDs to delete.

**Returns**
- None

**Notes**
- All specified tasks will be deleted in a single request.
- The operation is logged after completion.

---

#### `delete_task`
Deletes a single CVAT task by its ID.

**Parameters**
- `task_id` (`int`): ID of the CVAT task to delete.

**Returns**
- None

**Notes**
- The task is first retrieved and then removed.
- The operation is logged after deletion.

---

#### `add_label_to_project`
Adds a new label to an existing CVAT project.

**Parameters**
- `project_id` (`int`): ID of the CVAT project.
- `label_name` (`str`): Name of the label to create.
- `color` (`str`): Label color in hex format (e.g., `"#FF5733"`).
- `label_type` (`str`): Label type (e.g., `"polygon"`, `"rectangle"`, `"tag"`).
- `attributes` (`List`, optional): List of CVAT label attributes. Defaults to an empty list.

**Returns**
- None

**Notes**
- The label is added using a partial project update.
- Existing labels are preserved.

---

#### `delete_label_from_project`
Deletes a label from a CVAT project by label name.

**Parameters**
- `project_id` (`int`): ID of the CVAT project.
- `label_name` (`str`): Name of the label to delete.

**Returns**
- None

**Notes**
- The label is soft-deleted using a patched project update.
- If multiple labels share the same name, only the first match is deleted.

---
## Usage Examples

### Create a CVAT Project
```python
from cvat_utils import create_cvat_project

project_name = "weld_inspection_project"
project_id = create_cvat_project(project_name)

print(f"Created CVAT project with ID: {project_id}")
```

### Create a CVAT Task from Cloud Storage
```python
from cvat_utils import create_cvat_task_from_cloud_storage

s3_file_keys = [
    "datasets/welds/image_001.jpg",
    "datasets/welds/image_002.jpg",
    "datasets/welds/image_003.jpg",
]

project_id, task_id = create_cvat_task_from_cloud_storage(s3_file_keys)

print(f"Project ID: {project_id}")
print(f"Task ID: {task_id}")

```
### Get All Labels from a Project
```python
from cvat_utils import get_all_labels_from_project

project_id = 42
labels = get_all_labels_from_project(project_id)

for label in labels:
    print(label.name, label.type)
```
### Delete Multiple Tasks
```python
from cvat_utils import delete_tasks

task_ids = [101, 102, 103]
delete_tasks(task_ids)
```

---

## CVAT Dataset (`cvat_dataset.py`)

## CVAT Dataset Download and Annotation Processing

This module provides utilities to:
- Download CVAT task datasets
- Collect images into a unified directory
- Parse CVAT XML annotations
- Convert polygon annotations into segmentation masks
- Clean up temporary files

---

## Label Mapping

```python
label_mapping = {
    'weld': 1,
    'material1': 2,
    'material2': 3
}
```

Maps CVAT label names to integer values used in segmentation masks.

---

#### `download_cvat_tasks`
Downloads one or more CVAT task datasets, extracts them, and consolidates all images into a single directory.

**Parameters**
 - `task_ids` (`int | List[int]`):
CVAT task ID or list of task IDs to download.
 - `output_dir` (`str, optional`):
Base directory for downloaded data. Defaults to "temp".

**Directory Structure Created**

output_dir/
├── images/  # All extracted images
├── task_<id>_extracted/
│   └── annotations, metadata, etc.

**Behavior**
 - Downloads datasets in CVAT for images 1.1 format
 - Extracts ZIP files automatically
 - Moves all images into output_dir/images/
 - Automatically renames duplicate image filenames

**Returns**
 - None

---

#### `load_cvat_xml`
Loads a CVAT XML annotation file.

**Parameters**
 - `xml_path` (`str`): Path to a CVAT XML file.

**Returns**
 - `xml_root` (`xml.etree.ElementTree.Element`): Root element of the XML tree.

---

#### `get_polygons_from_cvat_xml`
Extracts polygon annotations from a CVAT XML file and returns them as a DataFrame.

**Parameters**
 - `xml_root` (`Element`): Parsed XML root element.

**Returns**
 - `polygons_df` (`pandas.DataFrame`) with columns:
  - `label` (`str`)
  - `points` (`str`)
  - `width` (`int`)
  - `height` (`int`)
  - `image_path` (`str`)

---

#### `draw_polygon`
Draws a filled polygon on a mask array using OpenCV.

**Parameters**
 - `mask` (`np.ndarray`): Target mask array.
 - `points` (`str`): Polygon points in CVAT format ("x1,y1;x2,y2;...").
 - `label` (`str`): Label name mapped via label_mapping.

**Returns**
 - None

---

#### `save_mask`
Saves a NumPy array as an image file.

**Parameters**
 - `array` (`np.ndarray`): Mask array to save.
 - `file_path` (`str`): Output image path.

**Returns**
 - None

---

#### `extract_annotations_to_masks`
Converts CVAT polygon annotations into segmentation masks.

**For each image:**
 - Collects all polygon annotations
 - Generates a single mask image
 - Each label is encoded using label_mapping

**Parameters**
 - `temp_dir` (`str, optional`):
Directory containing extracted CVAT datasets. Defaults to "temp".
 - `output_dir` (`str, optional`):
Directory where masks are saved. Defaults to "temp/labels".

**Output**
One mask image per source image:

temp/labels/
├── image_001.png
├── image_002.png

**Returns**
 - `image_masks` (`List[Dict]`):

```python
{
    "image_path": str,
    "label_path": str
}
```

---

#### `delete_folder`
Deletes a folder and all of its contents.

**Parameters**
 - `folder_path` (`str, optional`):
Path to the folder to delete. Defaults to "temp".

**Returns**
 - None

---
## Typical End-to-End Workflow
```
from cvat_utils import (
    download_cvat_tasks,
    extract_annotations_to_masks,
    delete_folder
)

# Step 1: Download CVAT tasks
download_cvat_tasks(task_ids=[101, 102], output_dir="temp")

# Step 2: Convert annotations to segmentation masks
image_mask_pairs = extract_annotations_to_masks(
    temp_dir="temp",
    output_dir="temp/labels"
)

# Step 3: Model Training

# Step 4: Cleanup temporary files
delete_folder("temp")
```

## Annotation Management

The `XMLAnnotationManager` class provides a high-level interface for downloading, loading, and managing CVAT XML annotations for a specific task and project.

It supports:
- Loading annotations from an existing XML file
- Downloading annotations directly from CVAT
- Parsing and exposing the XML root for downstream processing
- Modifying the downloaded XML by adding new annotations
- Uploading the XML to the task
---

### `XMLAnnotationManager`
Manages CVAT XML annotations for a given task.

#### Initialization
```python
XMLAnnotationManager(
    project_id: int,
    task_id: int,
    xml_path: Optional[str] = None
)
```
**Parameters**
 - `project_id` (`int`):
ID of the CVAT project.
 - `task_id` (`int`):
ID of the CVAT task.
 - `xml_path` (`str, optional`):
Path to an existing CVAT XML annotation file.
If not provided, annotations are downloaded automatically from CVAT.

**Behavior**
 - If xml_path exists, annotations are loaded from disk.
 - Otherwise, annotations are downloaded from CVAT in CVAT for images 1.1 format.
 - Extracted annotations are parsed into an XML tree and stored internally.

**Attributes**
 - `client`: Authenticated CVAT client.
 - `project_id` (`int`): CVAT project ID.
 - `task_id` (`int`): CVAT task ID.
 - `xml_path` (`str`): Path to the annotation XML file.
 - `tree` (`xml.etree.ElementTree.ElementTree`): Parsed XML tree.
 - `root` (`xml.etree.ElementTree.Element`): Root XML element.
---

### Internal Methods

#### `_download_annotations`
Downloads annotations from CVAT and parses the XML file.

**Behavior**
 - Exports task annotations as a ZIP file
 - Extracts the archive into a temporary directory
 - Loads annotations.xml into an XML tree

**Notes**
 - Images are not included in the export.
 - An error is logged if the task does not exist or the export fails
---
#### `save`
Save the modified XML annotations to file.

**Parameters**        
 - `output_path` (`str, optional`): Path to save XML file. If None, overwrites the original.

**Returns**
 - None
---

#### `_get_image_element`
Retrieves the `<image>` XML element corresponding to a specific frame.

**Parameters**
 - `frame` (`int`):
Frame number (image ID) to retrieve from the XML annotations.

**Returns**
 - `image` (`xml.etree.ElementTree.Element`):
The <image> element matching the specified frame ID.

**Behavior**
 - Iterates through all <image> elements in the XML root.
 - Matches the id attribute against the provided frame number.

**Returns** 
 - the first matching <image> element.

**Notes**
 - This method assumes the frame exists in the XML.
 - If no matching frame is found, the method returns None implicitly.

#### `add_tag_to_frame`
Adds a tag annotation to a specific frame in the CVAT XML.
If the tag label does not already exist in the task, it is automatically created
at the project level before being applied.

**Parameters**
 - `frame` (`int`):
Frame number to which the tag will be added.
 - `label` (`str`):
Name of the tag label.
 - `attributes` (`Dict[str, str], optional`):
Dictionary of attribute name–value pairs to attach to the tag.

**Behavior**
 - Retrieves existing tag labels for the task.
 - Automatically creates the tag label in the project if it does not exist.
 - Prevents duplicate tags on the same frame.
 - Adds optional attributes as XML sub-elements.

**Notes**
 - Logs a warning if the tag already exists on the frame.
 - Logs an info message when a tag is successfully added.
---

#### `add_points_to_frame`
Add points annotation to a specific frame in the XML.

**Parameters**
 - `frame` (`int`): Frame number
 - `label` (`str`): Label name for the points
 - `points` (`List[Tuple[float, float]]`): List of (x, y) coordinate tuples
 - `attributes` (`Optional[Dict[str, str]] = None`): dictionary of attribute name-value pairs
 - `occluded` (`bool = False`): Whether the points are occluded
 - `z_order` (`int = 0`): Drawing order (higher values on top)

**Returns**
 - None
---

#### `add_polygon_to_frame`

**Parameters**
 - `frame` (`int`): Frame number
 - `label` (`str`): Label name for the polygon
 - `points` (`List[Tuple[float, float]]`): List of (x, y) coordinate tuples defining the polygon vertices
 - `attributes` (`Optional[Dict[str, str]] = None`):  dictionary of attribute name-value pairs
 - `occluded` (`bool = False`): Whether the polygon is occluded
 - `z_order` (`int = 0`): Drawing order (higher values on top)

**Returns**
 - None
---

---
## Usage Example

```python
if __name__ == "__main__":

    # Initialize manager (downloads XML from CVAT)
    manager = XMLAnnotationManager(task_id=655, project_id = 60)
    temp_dir = Path("temp/add_labels")
    dummy_data = [
                    {
                        "frame": 13, 
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
                        "frame": 15, 
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

```


## MLflow Utilities

This module provides utilities to interact with MLflow for:
- Listing experiments and runs
- Inspecting registered models and versions
- Registering models from runs
- Managing model version tags and aliases
- Loading models by alias

---

#### `list_experiments`
Lists all MLflow experiments.

**Returns**
- `List[Dict]`: A list of experiments with the following fields:
  - `experiment_id`
  - `name`
  - `artifact_location`
  - `lifecycle_stage`
  - `tags`
  - `creation_time`
  - `last_update_time`

- Returns `None` if an error occurs.

---

#### `list_runs`
Lists all runs for a given experiment.

**Parameters**
 - `experiment_id` (`str`): MLflow experiment ID.

**Returns**
 - `List[Dict]`: A list of runs containing:
    - run_id
    - run_status
    - run_start_time
    - run_params
    - run_metrics
 - None if an error occurs.
 ---

#### `get_mlflow_run`
Retrieves details for a single MLflow run.

**Parameters**
 - `run_id` (`str`): MLflow run ID.

**Returns**
 - `Dict`: Run details including status, parameters, and metrics.
 - Returns None if an error occurs.
---

#### `list_models`
Lists all registered MLflow models and their versions.

**Returns**
 - `List[Dict]`: Each model contains:
    - model_name
    - description
    - tags
    - created_date
    - last_modified_date
    - version_details (list of versions)
     - Each version includes:
      - version
      - run_id
      - source
      - tags
      - created_date
      - last_modified_date
---

#### `register_model`
Registers a run as a new model version.

**Parameters**
 - `model_name` (`str`): Name of the registered model.
 - `run_id` (`str`): MLflow run ID to register.
 - `description` (`str, optional`): Model description.

**Behavior**
 - Creates the model if it does not already exist.
 - Registers a new version pointing to runs:/<run_id>/model.

**Returns**
 - `ModelVersion`: The newly created model version.
 - Returns None if an error occurs.
---

#### `list_model_versions`
Lists all versions of a registered model.

**Parameters**
 - `model_name` (`str`): Name of the registered model.

**Returns**
 - `List[Dict]`: Version details including:
    -version
    -aliases
    -tags
    -source
    -creation_timestamp
    -last_modified_date
---

#### `add_model_version_tags`
Adds tags to a specific model version.

**Parameters** 
 - `model_name` (`str`): Name of the registered model.
 - `version` (`str | int`): Model version.
 - `tags` (`Dict[str, str]`): Key–value tags to add.

**Returns**
 - `bool`: True if successful, False otherwise.
 ---

#### `set_model_alias`
Assigns an alias to a model version.

**Parameters**
 - `model_name` (`str`): Name of the registered model.
 - `version` (`str | int`): Model version.
 - `alias` (`str`): Alias name (e.g. production, champion).

**Returns**
 - `bool`: True if successful, False otherwise.
---

#### `remove_model_alias`
Removes an alias from a model.

**Parameters**
 - `model_name (str)`: Name of the registered model.
 - `alias (str)`: Alias to remove.

**Returns**
 - `bool`: True if successful, False otherwise.
 ---

#### `get_mlflow_run`
Retrieves an MLflow run by its run ID.

**Parameters**
 - `run_id (str)`: MLflow run ID.

**Returns**
 - `Dict`: contains the following details:
   - Experiment ID
   - Run metadata (status, name, timestamps)
   - Metrics
   - Parameters

---

#### `get_run_url`
Generates a direct URL to an MLflow run in the tracking UI.

**Parameters**
 - `run_id (str)`: MLflow run ID.
 - `mlflow_tracking_uri (str, optional)`: Base MLflow tracking URI.
    If not provided, the client’s configured tracking URI is used.

**Returns**
 - `str`: URL to the run in the MLflow UI.

**URL Format**
{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}

---

#### `delete_run`
Marks an MLflow run as deleted.

**Parameters**
`run_id (str)`: MLflow run ID to delete.

**Returns**
`None`

**Behavior**
* Soft-deletes the run (MLflow default behavior).
* Logs the deletion status.

**Notes**
* Requires a configured MLflow tracking URI.
* Deleted runs can still be recovered depending on backend configuration.
* Useful for debugging, cleanup, and experiment auditing workflows.

---

