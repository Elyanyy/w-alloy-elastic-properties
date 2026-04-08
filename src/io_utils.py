import json
import pandas as pd
from pathlib import Path


def save_records_csv(records, filepath):
    """
    Save a list of dictionaries to CSV.
    """
    df = pd.DataFrame(records)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


def save_json(obj, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def load_csv(filepath):
    return pd.read_csv(filepath)