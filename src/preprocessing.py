import pandas as pd
from sklearn.model_selection import train_test_split

def load_data() -> pd.DataFrame:
    return pd.read_csv('./data/raw/heart.csv')

def split_features_target(df: pd.DataFrame) -> tuple:
    y = df['target']
    X = df.drop(columns=['target'])
    return X, y

def split_data(
    df: pd.DataFrame,
    test_size: float=0.2,
    random_state: float=42
    ) -> tuple:
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def features_columns_names_by_type(df: pd.DataFrame) -> tuple:
    X, _ = split_features_target(df)
    cat_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang']
    num_columns = list(set(X.columns) - set(cat_columns))
    return num_columns, cat_columns

