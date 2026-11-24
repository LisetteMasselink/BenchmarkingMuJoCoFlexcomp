import xml.etree.ElementTree as ET
import glob
import os
import re
import meshio
import trimesh
import numpy as np


def _ints(text):
    return [int(v) for v in text.split()]


def _grid_pins(pin, count):
    """Grid coordinates (ix, iy, iz) pinned by one <pin> of a grid flexcomp (ids run z-fastest)."""
    _, ny, nz = count
    coords = set()
    if pin.get("id"):
        coords |= {(i // (ny * nz), (i // nz) % ny, i % nz) for i in _ints(pin.get("id"))}
    if pin.get("gridrange"):
        lo, hi = np.array(_ints(pin.get("gridrange"))).reshape(2, 3)
        coords |= {(x, y, z) for x in range(lo[0], hi[0] + 1) for y in range(lo[1], hi[1] + 1)
                   for z in range(lo[2], hi[2] + 1)}
    if pin.get("grid") or pin.get("range"):
        raise ValueError("only <pin id> and <pin gridrange> are supported when changing the mesh density")
    return coords


def set_grid_density(xml_path, flex_name, new_count):
    """Xml text of xml_path with the grid flexcomp flex_name re-meshed to new_count = [nx, ny, nz] vertices.
    The size of the block is kept (spacing = size / (count - 1)) and the pins are rewritten as gridrange: every
    pinned block must be a whole face/edge/corner of the grid (e.g. the bottom layer), so it maps onto the new grid."""
    tree = ET.parse(xml_path)
    flexcomp = next((f for f in tree.iter("flexcomp") if f.get("name") == flex_name), None)
    if flexcomp is None:
        raise KeyError(f"flexcomp '{flex_name}' not found in {xml_path}")
    if flexcomp.get("type") != "grid":
        raise ValueError(f"flexcomp '{flex_name}' is of type '{flexcomp.get('type')}': mesh density can only be "
                         f"changed for a grid (for gmsh, make a new .msh and pin ids)")
    old_count = np.array(_ints(flexcomp.get("count")))
    new_count = np.array(new_count, dtype=int)
    if new_count.shape != (3,) or (new_count < 2).any():
        raise ValueError(f"mesh_count must be 3 numbers >= 2, got {new_count.tolist()}")
    size = np.array([float(v) for v in flexcomp.get("spacing").split()]) * (old_count - 1)
    flexcomp.set("count", " ".join(str(n) for n in new_count))
    flexcomp.set("spacing", " ".join(f"{s:.12g}" for s in size / (new_count - 1)))

    for pin in flexcomp.findall("pin"):
        coords = np.array(sorted(_grid_pins(pin, old_count)))
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        last_old, last_new = old_count - 1, new_count - 1
        # per axis: the full range, or only the first or only the last layer
        per_axis = [(l, h) in ((0, e), (0, 0), (e, e)) for l, h, e in zip(lo, hi, last_old)]
        if len(coords) != np.prod(hi - lo + 1) or not all(per_axis):
            raise ValueError(f"pins of '{flex_name}' are not a whole face/edge/corner of the grid, cannot remap them")
        remap = lambda c: np.where(c == last_old, last_new, 0)
        for key in ("id", "gridrange", "grid", "range"):
            pin.attrib.pop(key, None)
        pin.set("gridrange", " ".join(str(v) for v in np.r_[remap(lo), remap(hi)]))
    return ET.tostring(tree.getroot(), encoding="unicode")
