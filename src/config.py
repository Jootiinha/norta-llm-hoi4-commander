from pathlib import Path
from typing import Any

import yaml


PIPELINE_CONFIG_PATH = Path("configs/pipeline.yaml")


def load_pipeline_config(path: Path = PIPELINE_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Arquivo de configuracao nao encontrado: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise SystemExit(f"Configuracao invalida em {path}: esperado um mapa YAML.")

    return config


def get_config_section(section: str, path: Path = PIPELINE_CONFIG_PATH) -> dict[str, Any]:
    config = load_pipeline_config(path)
    if section not in config:
        raise SystemExit(f"Secao '{section}' nao encontrada em {path}.")

    value = config[section]
    if not isinstance(value, dict):
        raise SystemExit(f"Secao '{section}' invalida em {path}: esperado um mapa YAML.")

    return value


def require_config_value(config: dict[str, Any], key: str, section: str) -> Any:
    if key not in config or config[key] is None:
        raise SystemExit(f"Configure '{section}.{key}' em {PIPELINE_CONFIG_PATH}.")

    return config[key]


def require_config_key(config: dict[str, Any], key: str, section: str) -> Any:
    if key not in config:
        raise SystemExit(f"Configure '{section}.{key}' em {PIPELINE_CONFIG_PATH}.")

    return config[key]
