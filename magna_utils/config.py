import os
from dotenv import load_dotenv

load_dotenv()

# CVAT Configuration
CVAT_TASK_NAME = os.getenv("CVAT_TASK_NAME")
CVAT_URL = os.getenv("CVAT_URL")
CVAT_PORT = os.getenv("CVAT_PORT", "8080")
CVAT_USERNAME = os.getenv("CVAT_USERNAME")
CVAT_PASSWORD = os.getenv("CVAT_PASSWORD")
CVAT_CLOUD_STORAGE_ID = os.getenv("CVAT_CLOUD_STORAGE_ID")

# MLFLOW Configuration
MLFLOW_TRACKING_URI = (
    os.getenv("MLFLOW_TRACKING_URI")
    or os.getenv("MLFLOW_URI")
)