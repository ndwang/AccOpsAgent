"""Configuration loader for accelerator configurations."""

import os
from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from .schema import AcceleratorConfig


class ConfigLoadError(Exception):
    """Raised when configuration loading fails."""

    pass


def load_accelerator_config(config_path: Union[str, Path]) -> AcceleratorConfig:
    """Load and validate an accelerator configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        AcceleratorConfig: Validated configuration object

    Raises:
        ConfigLoadError: If the file cannot be loaded or validation fails
    """
    config_path = Path(config_path)

    # Check if file exists
    if not config_path.exists():
        raise ConfigLoadError(f"Configuration file not found: {config_path}")

    # Load YAML file
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Failed to parse YAML file: {e}")
    except IOError as e:
        raise ConfigLoadError(f"Failed to read configuration file: {e}")

    if config_data is None:
        raise ConfigLoadError("Configuration file is empty")

    # Resolve tao_init_file path relative to config file if it's relative
    if "tao_init_file" in config_data and config_data["tao_init_file"]:
        tao_path = Path(config_data["tao_init_file"])
        if not tao_path.is_absolute():
            # Make it relative to the config file location
            config_data["tao_init_file"] = str(
                (config_path.parent / tao_path).resolve()
            )

    # Validate using Pydantic
    try:
        config = AcceleratorConfig(**config_data)
    except ValidationError as e:
        raise ConfigLoadError(f"Configuration validation failed: {e}")

    return config


def validate_config_file(config_path: Union[str, Path]) -> tuple[bool, str]:
    """Validate a configuration file without loading it fully.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        tuple: (is_valid, error_message)
            - is_valid: True if config is valid, False otherwise
            - error_message: Empty string if valid, error details otherwise
    """
    try:
        load_accelerator_config(config_path)
        return True, ""
    except ConfigLoadError as e:
        return False, str(e)
