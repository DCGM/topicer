import yaml
import json
import logging

class Config:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            config: dict = yaml.safe_load(f)
        self.weaviate_cfg: dict = config.get('weaviate', {})
        self.openai_cfg: dict = config.get('openai', {})

        logging.debug("Loaded config: %s", json.dumps(config, indent=4))

        if (not self.weaviate_cfg) or (not self.openai_cfg):
            raise ValueError(
                "Invalid config file: missing 'weaviate' or 'openai' sections")

        if (not self.openai_cfg.get('model')):
            self.openai_cfg['model'] = 'gpt-4o-mini'