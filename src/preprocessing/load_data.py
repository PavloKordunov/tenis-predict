import pandas as pd
import os

def load_raw_matches(path: str) -> pd.DataFrame:
    all_dfs = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)
