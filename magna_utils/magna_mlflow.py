import mlflow
from mlflow.tracking import MlflowClient
from magna_utils import config
import logging
import traceback

# Configurar el nivel de registro para httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_mlflow_client():
    """Return a configured MLFlow client or None if connection fails."""
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

        client = MlflowClient()

        return client
    except Exception as e:
        logging.error(f"❌ Error creating MLFlow client: {e}")
        traceback.print_exc()
        return None

def list_experiments ():

    """Return a list of experiments"""

    client = get_mlflow_client()

    try:
        # List all experiments
        experiments = client.search_experiments()

        mlflow_experiments = []

        for exp in experiments:
            experiment = {}
            experiment['experiment_id'] = exp.experiment_id
            experiment['name'] = exp.name
            experiment['artifact_location'] = exp.artifact_location
            experiment['lifecycle_stage'] = exp.lifecycle_stage
            experiment['tags'] = exp.tags
            experiment['creation_time'] = exp.creation_time
            experiment['last_update_time'] = exp.last_update_time

            mlflow_experiments.append(experiment)

        return mlflow_experiments

    except Exception as e:
        logging.error(f"❌ Error retrieving experiments: {e}")
        traceback.print_exc()
        return None


def list_runs (experiment_id):

    """
    Return a list of runs from a specified experiment_id.

    Parameters
    ----------
    experiment_id : str
        The experiment_id
    """

    client = get_mlflow_client()

    try:
        # List runs for a specific experiment
        runs = client.search_runs(experiment_ids=[experiment_id])

        run_details = []

        # Iterate through runs
        for run in runs:
            run_detail = {}
            run_detail["run_id"] = run.info.run_id
            run_detail["run_status"] = run.info.status
            run_detail["run_start_time"] = run.info.start_time
            run_detail["run_params"] = run.data.params
            run_detail["run_metrics"] = run.data.metrics
            run_details.append(run_detail)

        return run_details

    except Exception as e:
        logging.error(f"❌ Error retrieving runs: {e}")
        traceback.print_exc()
        return None

def get_mlflow_run(run_id):

    client = get_mlflow_client()

    try:
        # List runs for a specific experiment
        run = client.get_run(run_id)
        run_detail = {}
        run_detail["experiment_id"] = run.info.experiment_id
        run_detail["run_id"] = run.info.run_id
        run_detail["run_name"] = run.info.run_name
        run_detail["run_status"] = run.info.status
        run_detail["start_time"] = run.info.start_time
        run_detail["end_time"] = run.info.end_time
        run_detail["run_metrics"] = run.data.metrics
        run_detail["run_params"] = run.data.params
        return run_detail
    except Exception as e:
        logging.error(f"❌ Error retrieving runs: {e}")
        traceback.print_exc()
        return None

def get_run_url(run_id, mlflow_tracking_uri=None):
    """Generate run URL"""

    client = get_mlflow_client()

    try:
        # Get the run to retrieve experiment_id
        run = client.get_run(run_id)
        experiment_id = run.info.experiment_id

        # Get tracking URI if not provided
        if mlflow_tracking_uri is None:
            mlflow_tracking_uri = client.tracking_uri

        # Build the URL
        # Format: {tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}
        url = f"{mlflow_tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"

        return url

    except Exception as e:
        print(f"Error opening run: {e}")
        return None


def delete_run (run_id: str):
    client = get_mlflow_client()
    try:
        client.delete_run(run_id)
        logger.info(f"Run {run_id} marked as deleted.")
    except Exception as e:
        logger.info(f"Error deleting run: {e}")

def list_models ():

    """Return a list of registered models."""

    client = get_mlflow_client()

    try:

        # Get all registered models with their versions
        registered_models = client.search_registered_models()

        model_details = []

        for rm in registered_models:
            model_detail = {}
            model_detail['model_name'] = rm.name
            model_detail['description'] = rm.description
            model_detail['tags'] = rm.tags
            model_detail['created_date'] = rm.creation_timestamp
            model_detail['last_modified_date'] = rm.last_updated_timestamp

            # Get all versions for this model
            versions = client.search_model_versions(f"name='{rm.name}'")
            version_details = []

            for version in versions:
                version_detail = {}
                version_detail['version'] = version.version
                version_detail['run_id'] = version.run_id
                version_detail['source'] = version.source
                version_detail['tags'] = version.tags
                version_detail['created_date'] = version.creation_timestamp
                version_detail['last_modified_date'] = version.last_updated_timestamp
                version_details.append(version_detail)

            model_detail['version_details'] = version_details

            model_details.append(model_detail)

        return model_details
    except Exception as e:
        logging.error(f"❌ Error retrieving models: {e}")
        traceback.print_exc()
        return None

def register_model(model_name, run_id, description = ""):
    """
    Register run with specified run_id as a model.
    Create model_name if model_name does not exist.

    Parameters
    ----------
    model_name : str
        The model name
    run_id : str
        The run_id of the run to be registered
    description: str
        (optional) The description of the model
    """
    client = get_mlflow_client()

    try:
        # Check if model exists, if not create it with metadata
        try:
            client.get_registered_model(model_name)
            logger.info(f"Model '{model_name}' already exists")
        except mlflow.exceptions.RestException:
            # Model doesn't exist, create it
            client.create_registered_model(
                name=model_name,
                description=description
            )
            logger.info(f"Created new registered model '{model_name}'")

        # Register a new model version
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Run {run_id} has been registered as {model_name} version {model_version.version}")
        return model_version

    except Exception as e:
        logger.error(f"❌ Error registering new version: {e}")
        traceback.print_exc()
        return None


def add_model_version_tags(model_name, version, tags):
    """
    Add tags to a specific model version.

    Args:
        model_name: Name of the registered model
        version: Version number (string or int)
        tags: Dictionary of key-value pairs to add as tags

    Returns:
        bool: True if successful, False otherwise
    """
    client = get_mlflow_client()

    try:
        for key, value in tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=str(version),
                key=key,
                value=str(value)
            )
        logger.info(f"Added {len(tags)} tags to {model_name} v{version}")
        return True

    except Exception as e:
        logger.error(f"❌ Error adding tags to {model_name} v{version}: {e}")
        traceback.print_exc()
        return False


def set_model_alias(model_name, version, alias):
    """
    Set an alias for a specific model version.

    Args:
        model_name: Name of the registered model
        version: Version number (string or int)
        alias: Alias name (e.g., 'champion', 'challenger', 'production')

    Returns:
        bool: True if successful, False otherwise
    """
    client = get_mlflow_client()

    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(version)
        )
        logger.info(f"Set alias '{alias}' for {model_name} v{version}")
        return True

    except Exception as e:
        logger.error(f"❌ Error setting alias '{alias}' for {model_name} v{version}: {e}")
        traceback.print_exc()
        return False


def remove_model_alias(model_name, alias):
    """
    Remove an alias from a model.

    Args:
        model_name: Name of the registered model
        alias: Alias name to remove

    Returns:
        bool: True if successful, False otherwise
    """
    client = get_mlflow_client()

    try:
        client.delete_registered_model_alias(
            name=model_name,
            alias=alias
        )
        logger.info(f"Removed alias '{alias}' from {model_name}")
        return True

    except Exception as e:
        logger.error(f"❌ Error removing alias '{alias}' from {model_name}: {e}")
        traceback.print_exc()
        return False


def list_model_versions (model_name):
    """
    List model versions

    Args:
        model_name: Name of the registered model

    Returns:
        list: Model versions with details: version, aliases, tags, source, creation_timestamp, last_modified_date
    """
    client = get_mlflow_client()
    # Get all model versions with their aliases
    versions = client.search_model_versions(f"name='{model_name}'")
    version_details = []
    for version in versions:
        # print(f"Version {version.version}: Aliases = {version.aliases}")
        version_detail = {}
        version_detail['version'] = version.version
        version_detail['aliases'] = version.aliases
        version_detail['tags'] = version.tags
        version_detail['source'] = version.source
        version_detail['creation_timestamp'] = version.creation_timestamp
        version_detail['last_modified_date'] = version.last_updated_timestamp
        version_details.append(version_detail)
    return version_details

