"""Small helpers: json save/load of results, settings loading and merging."""
import copy
import json
import os

import numpy as np


def _to_json(value):
    """Convert numpy arrays/scalars (also inside dicts and lists) to plain Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    return value


def save_data(path, data, metadata=None):
    """Save results as json: {"metadata": ..., "data": ...}; missing folders are created."""
    folder = os.path.dirname(str(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as file:
        json.dump({"metadata": _to_json(metadata or {}), "data": _to_json(data)}, file)
    print(f"saved {len(data)} entries to {path}")


def load_data(path):
    """Load a file written by save_data; lists in data come back as numpy arrays. Returns (data, metadata)."""
    with open(path) as file:
        content = json.load(file)
    data = {k: np.array(v) if isinstance(v, list) else v for k, v in content["data"].items()}
    return data, content["metadata"]


def deep_merge(base, override):
    # find out for what this is used 
    """Return a copy of base with the (nested) keys of override replacing those of base."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

SIMULATION_KEYS = ("protocol",)   # the motion; every other key of a case describes the model (xml, flex, material, mesh)


def _load_override(path, keep, what):
    """Override json; only keys of its own kind (model or simulation) are allowed."""
    with open(path) as f:
        override = json.load(f)
    wrong = [k for k in override if not keep(k)]
    if wrong:
        raise KeyError(f"{path} is a {what} override but contains {wrong}")
    return override


def load_case_settings(settings_file, case, override_xml=None, override_simulation=None):
    """Settings of one case from settings.json, optionally merged with override json files of the same
    structure as one case entry: override_xml for the model, e.g. {"model": {"young": 110e3}},
    override_simulation for the motion, e.g. {"protocol": {"speed": 0.002}}."""

    # try the open the settinsg file
    with open(settings_file) as f:
        settings = json.load(f)

    # the specified case is not in the setings file
    if case not in settings:
        raise KeyError(f"{case} not in {settings_file}; available: {list(settings)}")

    case_settings = settings[case]

    if override_xml:
        case_settings = deep_merge(case_settings, _load_override(override_xml, lambda k: k not in SIMULATION_KEYS, "model"))
    if override_simulation:
        case_settings = deep_merge(case_settings, _load_override(override_simulation, lambda k: k in SIMULATION_KEYS, "simulation"))
    return case_settings


def load_case_settings_model(settings_file, case, override_xml=None):
    """Model part of one case (xml, flex, tool, base, "model": young, poisson, damping, mesh_count, ...),
    optionally merged with an override json, e.g. {"model": {"young": 110e3, "mesh_count": [21, 21, 8]}}."""
    settings = load_case_settings(settings_file, case, override_xml=override_xml)
    return {k: v for k, v in settings.items() if k not in SIMULATION_KEYS}


def load_case_settings_simulation(settings_file, case, override_simulation=None):
    """Simulation part of one case ("protocol"), optionally merged with an override json,
    e.g. {"protocol": {"speed": 0.002}}."""
    settings = load_case_settings(settings_file, case, override_simulation=override_simulation)
    return {k: v for k, v in settings.items() if k in SIMULATION_KEYS}
