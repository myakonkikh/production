import os

from loguru import logger

from preprocessing import load_data, split_data, features_columns_names_by_type
from train import train_pipline
from inference import load_pipline
from evaluate import evaluate_pipline

def main():
    logger.info('Starting the pipline')
    pipe_path = os.path.join('models', 'pipline.joblib')

    logger.info('Preprocessing data')
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    num_columns, cat_columns = features_columns_names_by_type(df)

    logger.info('Training model')
    pipe = train_pipline(
        model_params={'C': 1.},
        X_train=X_train,
        y_train=y_train,
        num_columns=num_columns,
        cat_columns=cat_columns,
        pipe_path=pipe_path
    )

    logger.info('Load pipline')
    pipe = load_pipline(pipe_path)

    logger.info('Evaluate model on test set')
    metric = evaluate_pipline(pipe, X_test, y_test)
    logger.info(f'Test metric {metric}')

    logger.info('Pipline finished')


if __name__ == "__main__":
    main()
