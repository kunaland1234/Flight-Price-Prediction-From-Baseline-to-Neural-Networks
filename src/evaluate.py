import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import tensorflow as tf


def compute_metrics(y_true_log, y_pred_log):
    y_true_inr = np.expm1(y_true_log)
    y_pred_inr = np.expm1(y_pred_log)

    return {
        'R2':   round(r2_score(y_true_log, y_pred_log), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true_inr, y_pred_inr)), 2),
        'MAE':  round(mean_absolute_error(y_true_inr, y_pred_inr), 2)
    }


def evaluate_xgb(X, y_true_log):
    preprocessor = joblib.load('models/xgb_preprocessor.pkl')
    model = XGBRegressor()
    model.load_model('models/xgb_model.json')

    X_t      = preprocessor.transform(X)
    y_pred   = model.predict(X_t)
    metrics  = compute_metrics(y_true_log, y_pred)

    print("\n=== XGBoost ===")
    print(f"R2:   {metrics['R2']}")
    print(f"RMSE: Rs. {metrics['RMSE']:,.0f}")
    print(f"MAE:  Rs. {metrics['MAE']:,.0f}")
    return metrics


def evaluate_ann(X, y_true_log):
    preprocessor = joblib.load('models/ann_preprocessor.pkl')
    model = tf.keras.models.load_model('models/ann_model.keras')

    X_t     = preprocessor.transform(X)
    y_pred  = model.predict(X_t, verbose=0).flatten()
    metrics = compute_metrics(y_true_log, y_pred)

    print("\n=== ANN ===")
    print(f"R2:   {metrics['R2']}")
    print(f"RMSE: Rs. {metrics['RMSE']:,.0f}")
    print(f"MAE:  Rs. {metrics['MAE']:,.0f}")
    return metrics


def compare_models(X, y_true_log):
    print("Evaluating on provided split...\n")
    xgb_metrics = evaluate_xgb(X, y_true_log)
    ann_metrics = evaluate_ann(X, y_true_log)

    print("\n=== Summary ===")
    print(f"{'Metric':<8} {'XGBoost':>12} {'ANN':>12}")
    print("-" * 34)
    for key in ['R2', 'RMSE', 'MAE']:
        print(f"{key:<8} {xgb_metrics[key]:>12,.2f} {ann_metrics[key]:>12,.2f}")


if __name__ == '__main__':
    splits = joblib.load('data/splits.pkl')
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    print("=== Validation Set ===")
    compare_models(X_val, y_val)

    print("\n=== Test Set ===")
    compare_models(X_test, y_test)