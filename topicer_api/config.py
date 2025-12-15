import os

TRUE_VALUES = {"true", "1"}

class Config:
    def __init__(self):
        self.APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
        self.APP_PORT = int(os.getenv("APP_PORT", "8000"))
        self.APP_RELOAD = self._env_bool("APP_RELOAD", False)

        self.TOPICER_API_CONFIGS_DIR = os.getenv("TOPICER_API_CONFIGS_DIR", "./configs")
        self.TOPICER_API_CONFIGS_EXTENSION = os.getenv("TOPICER_API_CONFIGS_EXTENSION", ".yaml")

    @staticmethod
    def _env_bool(key: str, default: bool = False) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return val.strip().lower() in TRUE_VALUES

config = Config()
