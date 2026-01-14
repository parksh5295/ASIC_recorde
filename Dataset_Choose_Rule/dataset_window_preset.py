"""
Presets for temporal window chunk sizes (row counts) per dataset.
Used when --cstemporal is specified to auto-select chunk_size.
"""

WINDOW_PRESETS = {
    # Row-based (4h recommended)
    'CICIDS2017': 100_000,
    'CICIDS': 100_000,
    'CICIoT2023': 600_000,
    'CICIoT': 600_000,
    'netML': 400,           # 120_000,      # Time-based ≈4h slice
    'DARPA98': 400,        # Time-based ≈4h slice
    'DARPA': 400,
    'NSL-KDD': 400,        # Pseudo-time
    'NSL_KDD': 400,
    'MiraiBotnet': 200,    # Episode-based, 100–200 rows
    'Kitsune': 140_000,    # Native window merged
}


def get_temporal_chunk_size(file_type: str):
    """Return preset chunk size for the dataset; None if not defined."""
    return WINDOW_PRESETS.get(file_type, WINDOW_PRESETS.get(file_type.replace('-', '_')))

