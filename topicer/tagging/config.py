from functools import lru_cache
from pathlib import Path
import yaml
from .schemas import AppConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

@lru_cache
def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig.model_validate(data)