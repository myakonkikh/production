from typing import Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

def load_pipline(path: str) -> Optional[Pipeline]:
    try:
        return joblib.load(path)
    except Exception as e:
        print(f'Error loading model from {path}: {e}')
        return None
    

def predict(pipe: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    return pipe.predict(df)
