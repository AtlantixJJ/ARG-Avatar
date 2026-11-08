"""Config loading and by-name class/function construction.

Extracted from xlib/utils.py (which re-exports everything here for backward
compatibility) so that the load-bearing config/construction machinery lives in
a dependency-light module.
"""
import os
import sys
import types
import re
import importlib
import inspect
from typing import Any, Optional, Tuple

from omegaconf import OmegaConf, DictConfig


def _load_config_with_bases(yaml_file: str, visited: Optional[set] = None) -> DictConfig:
    """Load a YAML config and recursively merge optional `base_configs`."""
    if visited is None:
        visited = set()

    config_path = os.path.abspath(os.path.expanduser(yaml_file))
    if config_path in visited:
        raise ValueError(f"Detected cyclic base_configs reference: {config_path}")
    visited.add(config_path)

    conf = OmegaConf.load(config_path)
    base_configs = conf.get("base_configs", [])
    if OmegaConf.is_config(base_configs):
        base_configs = OmegaConf.to_container(base_configs, resolve=True)
    if base_configs is None:
        base_configs = []
    if not isinstance(base_configs, list):
        raise ValueError(f"`base_configs` must be a list in {config_path}")

    merged_base_confs = []
    config_dir = os.path.dirname(config_path)
    for base_file in base_configs:
        if not isinstance(base_file, str):
            raise ValueError(f"Invalid base config path in {config_path}: {base_file}")
        base_path = base_file if os.path.isabs(base_file) else os.path.join(config_dir, base_file)
        merged_base_confs.append(_load_config_with_bases(base_path, visited))

    # Remove metadata key so downstream config objects stay clean.
    conf_dict = OmegaConf.to_container(conf, resolve=False)
    conf_dict.pop("base_configs", None)
    conf_no_base = OmegaConf.create(conf_dict)

    if merged_base_confs:
        return OmegaConf.merge(*merged_base_confs, conf_no_base)
    return conf_no_base


def load_config(yaml_file: str, cli_args: list = [], **kwargs) -> DictConfig:
    """Load a configuration from YAML, including optional recursive `base_configs`."""
    yaml_conf = _load_config_with_bases(yaml_file)
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(yaml_conf, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    return cfg


# Functionality to import modules/objects by name, and call functions by name
# ------------------------------------------------------------------------------------------

def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: Optional[str] = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: Optional[str] = None, **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))
