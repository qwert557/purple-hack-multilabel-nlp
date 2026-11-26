import pandas as pd

def parse_list_int(s):
    if isinstance(s, str) and s.startswith("["):
        return [int(x) for x in s.strip("[]").split(",") if x.strip()]
    return []

def safe_str(x):
    return "" if pd.isna(x) else str(x)