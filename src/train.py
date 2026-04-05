import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.model import build_ann, build_xgb


CAT_COLS = ['airline', 'source_city', 'destination_city',
            'departure_time', 'arrival_time', 'stops', 'class']

NUM_COLS = ['duration', 'duration_sq', 'days_left',
            'stops_num', 'dep_time_num', 'arr_time_num',
            'is_business', 'urgency_num']


def get_ann_preprocessor():
    return ColumnTransformer([
        ('ohe',    OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
        ('scaler', StandardScaler(), NUM_COLS)
    ])


def get_xgb_preprocessor():
    return ColumnTransformer([
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value',
                               unknown_value=-1), CAT_COLS),
        ('num', 'passthrough', NUM_COLS)
    ])


def train_xgb(X_train, y_train, X_val, y_val, params=None):
    preprocessor = get_xgb_preprocessor()

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t   = preprocessor.transform(X_val)

    model = build_xgb(params)
    model.fit(
        X_train_t, y_train,
        eval_set=[(X_train_t, y_train), (X_val_t, y_val)],
        verbose=False
    )

    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/xgb_preprocessor.pkl')
    model.save_model('models/xgb_model.json')

    print("XGBoost saved to models/")
    return model, preprocessor


def train_ann(X_train, y_train, X_val, y_val):
    preprocessor = get_ann_preprocessor()

    X_train_t  = preprocessor.fit_transform(X_train)
    X_val_t    = preprocessor.transform(X_val)
    input_dim  = X_train_t.shape[1]

    model = build_ann(input_dim)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-7, verbose=1)
    ]

    history = model.fit(
        X_train_t, y_train,
        validation_data=(X_val_t, y_val),
        epochs=200,
        batch_size=512,
        callbacks=callbacks,
        verbose=1
    )

    os.makedirs('models', exist_ok=True)
    model.save('models/ann_model.keras')
    joblib.dump(preprocessor, 'models/ann_preprocessor.pkl')

    print("ANN saved to models/")
    return model, preprocessor, history


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    splits = joblib.load('data/splits.pkl')
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    print("Training XGBoost...")
    train_xgb(X_train, y_train, X_val, y_val)

    print("\nTraining ANN...")
    train_ann(X_train, y_train, X_val, y_val)