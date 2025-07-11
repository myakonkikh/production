import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

def make_pipline(
    model_params: dict,
    num_columns: list,
    cat_columns: list
    ) -> Pipeline:

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', dtype=int))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_columns),
            ('cat', cat_transformer, cat_columns)
        ]
    )

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(**model_params))
    ])

    return pipe


def save_pipline(pipe: Pipeline, pipe_path: str):
    joblib.dump(pipe, pipe_path)


def train_pipline(
    model_params: dict,
    X_train: np.array,
    y_train: np.array,
    num_columns: str,
    cat_columns: str,
    pipe_path: str
    ) -> Pipeline:

    pipe = make_pipline(model_params, num_columns, cat_columns)
    pipe.fit(X_train, y_train)
    save_pipline(pipe, pipe_path)

    return pipe
