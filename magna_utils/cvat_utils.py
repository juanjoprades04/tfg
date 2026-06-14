import os
import logging
import traceback

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from cvat_sdk.api_client import models

import magna_utils.config as config

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_cvat_client():
    """Return a configured CVAT client or None if connection fails."""
    try:
        cvat_url = os.getenv("CVAT_URL") or config.CVAT_URL
        cvat_port = os.getenv("CVAT_PORT") or getattr(config, "CVAT_PORT", None)
        cvat_username = os.getenv("CVAT_USERNAME") or config.CVAT_USERNAME
        cvat_password = os.getenv("CVAT_PASSWORD") or config.CVAT_PASSWORD

        if not cvat_url:
            raise ValueError("CVAT_URL is not defined")

        if not cvat_username:
            raise ValueError("CVAT_USERNAME is not defined")

        if not cvat_password:
            raise ValueError("CVAT_PASSWORD is not defined")

        if cvat_port:
            client = make_client(
                host=cvat_url,
                port=int(cvat_port),
                credentials=(cvat_username, cvat_password),
            )
            client.api_client.configuration.host = f"{cvat_url}:{cvat_port}"
        else:
            client = make_client(
                host=cvat_url,
                credentials=(cvat_username, cvat_password),
            )
            client.api_client.configuration.host = cvat_url

        return client

    except Exception as e:
        logging.error(f"❌ Error creating CVAT client: {e}")
        traceback.print_exc()
        return None


def get_all_labels_from_task(task_id, label_type=None):
    client = get_cvat_client()
    if client is None:
        return None

    labels = []
    page_number = 1

    try:
        while True:
            response = client.api_client.labels_api.list(
                page=page_number,
                task_id=task_id
            )[0]

            results = response.results

            for result in results:
                current_type = getattr(result, "type", None)

                if label_type is not None:
                    if current_type == label_type:
                        labels.append(result)
                else:
                    labels.append(result)

            if response.next:
                page_number += 1
            else:
                break

        return labels

    except Exception as e:
        logging.error(f"❌ Error getting labels from task {task_id}: {e}")
        traceback.print_exc()
        return None

    finally:
        client.close()


def get_task_labels_metadata(task_ids):
    """
    Devuelve labels de segmentación únicas de una o varias tasks.
    Ignora labels de tipo tag.
    """
    if isinstance(task_ids, int):
        task_ids = [task_ids]

    unique_labels = []
    seen = set()

    for task_id in task_ids:
        labels = get_all_labels_from_task(task_id, None)
        if not labels:
            continue

        for label in labels:
            label_name = getattr(label, "name", None)
            label_type = getattr(label, "type", None)

            if not label_name:
                continue

            if label_type == "tag":
                continue

            clean_name = label_name.strip().lower()

            if clean_name not in seen:
                seen.add(clean_name)
                unique_labels.append(clean_name)

    return unique_labels


def get_project_list():
    client = get_cvat_client()
    if client is None:
        return None

    try:
        project_list = []
        page_number = 1

        while True:
            response = client.projects.api.list(page=page_number)[0]
            results = response.results

            for result in results:
                project_details = {}
                project_id = result.get("id")
                project_details["project_id"] = project_id
                project_details["project_name"] = result.get("name")
                project_details["created_date"] = result.get("created_date")
                project_details["last_modified_date"] = result.get("updated_date")
                project_details["number_of_tasks"] = result["tasks"].get("count")
                project_list.append(project_details)

            if response.next:
                page_number += 1
            else:
                break

        return project_list

    except Exception as e:
        logging.error(f"❌ Error fetching project list: {e}")
        traceback.print_exc()
        return None

    finally:
        client.close()


def create_cvat_task_from_cloud_storage(
    project_id: int,
    task_name: str,
    cloud_storage_id: int,
    s3_file_keys: list[str],
):
    if not s3_file_keys:
        logging.warning("No S3 file keys provided for CVAT task creation; skipping.")
        return None

    client = get_cvat_client()
    if client is None:
        return None

    try:
        task_spec = {
            "name": task_name,
            "project_id": project_id,
        }

        data = dict(
            image_quality=100,
            cloud_storage_id=cloud_storage_id,
            server_files=s3_file_keys,
            storage=models.StorageType("cloud_storage"),
        )

        task = client.tasks.create_from_data(
            spec=task_spec,
            resource_type=ResourceType.SHARE,
            resources=s3_file_keys,
            data_params=data,
        )

        logging.info(f"✅ CVAT task '{task_name}' created successfully with ID {task.id}.")
        return project_id, task.id

    except Exception as e:
        logging.error(f"❌ Error creating CVAT task '{task_name}': {e}")
        traceback.print_exc()
        return None

    finally:
        client.close()
