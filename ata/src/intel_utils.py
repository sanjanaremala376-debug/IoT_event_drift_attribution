"""Small shared helpers for the Intel Lab notebooks (kept deliberately tiny)."""
from pathlib import Path
import pandas as pd

RAW_COLUMNS = ["date", "time", "epoch", "moteid",
               "temperature", "humidity", "light", "voltage"]
MEASUREMENTS = ["temperature", "humidity", "light", "voltage"]

# Documented mote IDs (official Intel Lab page): 1..54
DOCUMENTED_MOTES = range(1, 55)

# Plausible physical ranges, used ONLY to flag values (raw values are never deleted)
VALID_RANGES = {
    "temperature": (0.0, 50.0),    # deg C, indoor lab
    "humidity":    (0.0, 100.0),   # % relative humidity
    "light":       (0.0, 2000.0),  # Lux (raw max in file is 1847.36)
    "voltage":     (2.0, 3.0),     # Volts (2xAA-style mote battery)
}


def load_raw(path: Path) -> pd.DataFrame:
    """Read data.txt without dropping or altering any line.

    Fields are separated by a SINGLE space, so a blank field (e.g. missing light)
    shows up as an empty string and keeps every other value in its correct column.
    Returns one row per raw line, in original file order, with numeric columns
    and a parsed `timestamp` column (NaT if unparseable).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    df = pd.read_csv(path, sep=" ", header=None, names=RAW_COLUMNS, dtype=str,
                     na_values=[""], keep_default_na=False)
    for c in RAW_COLUMNS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.drop(columns=["date", "time"])
    df["raw_line"] = range(len(df))  # original line number, so every row stays traceable
    return df
