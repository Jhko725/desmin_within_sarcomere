import csv
from pathlib import Path

import h5py
import numpy as np


def load_localizations(directory="../source_data/fig_2", *, prefix):
    """Read desmin/actinin CSVs, selecting columns by their exact header names.

    Returns the same dictionary structure as sq1_final. Row order is retained.
    Missing headers, empty data and non-finite coordinates raise an error.
    """
    result = {}
    columns = ("x (um)", "y (um)", "z (um)")
    for protein in ("desmin", "actinin"):
        path = Path(directory) / f"{prefix}{protein}.csv"
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not set(columns).issubset(reader.fieldnames or ()):
                raise ValueError(f"{path} must contain columns {columns}.")
            values = np.asarray([[float(row[c]) for c in columns] for row in reader])
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"{path} must contain nonempty, finite XYZ coordinates.")
        result[protein] = values
    return result


def read_SIM_data(filepath: Path | str):
    f = h5py.File(filepath)
    data = f["Data"][:]
    pixel_sizes = f["Data"].attrs["pixel_sizes"]
    dtype = data.dtype
    if dtype.kind == "u":
        data = data / np.iinfo(dtype).max
    return data, pixel_sizes
