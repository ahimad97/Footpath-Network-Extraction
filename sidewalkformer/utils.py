"""Configuration helpers."""

import yaml
from addict import Dict


def load_config(path):
    """Load a YAML config as an attribute-accessible ``addict.Dict``."""
    with open(path) as file:
        return Dict(yaml.safe_load(file))


def cfg_get(config, key, default=None):
    """Return ``config[key]``, or ``default`` when the key is missing.

    ``addict.Dict`` returns an empty ``Dict`` for missing attributes instead of
    raising, so ``getattr(config, key, default)`` never falls back to
    ``default``. Dict-valued entries are treated as missing for the same reason.
    """
    if isinstance(config, dict):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    return default if isinstance(value, dict) else value
