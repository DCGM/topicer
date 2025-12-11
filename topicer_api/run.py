import uvicorn
import logging

logger = logging.getLogger(__name__)


def main():
    logger.info(f"Running TopicerAPI")

    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8666,
                reload=True)


if __name__ == "__main__":
    exit(main())
