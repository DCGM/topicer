import os
import logging
from fastapi import Request

from topicer import factory as topicer_factory

from topicer_api.config import config as app_config


logger = logging.getLogger(__name__)


class LoadedTopicers(dict):
    pass


def load_topicers() -> LoadedTopicers:
    loaded_topicers = LoadedTopicers()

    config_files = [file for file in os.listdir(app_config.TOPICER_API_CONFIGS_DIR) if
                    file.endswith(app_config.TOPICER_API_CONFIGS_EXTENSION)]
    for config_file in config_files:
        config_name = os.path.splitext(config_file)[0]
        config_path = os.path.join(app_config.TOPICER_API_CONFIGS_DIR, config_file)

        try:
            loaded_topicers[config_name] = topicer_factory(config_path)
        except Exception as e:
            logger.warning(f"Failed to load topicer config '{config_name}' from '{config_path}': {e}")

    logger.info(f"Loaded topicer configurations: {list(loaded_topicers.keys())}")

    return loaded_topicers


def get_loaded_topicers(request: Request) -> LoadedTopicers:
    return request.app.state.loaded_topicers
