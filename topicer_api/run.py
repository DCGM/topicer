import uvicorn
import logging.config

from topicer_api.config import config


logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def main():
    logger.info(f"Running TopicerAPI")

    uvicorn.run("main:app",
                host=config.APP_HOST,
                port=config.APP_PORT,
                reload=config.APP_RELOAD)


if __name__ == "__main__":
    exit(main())
