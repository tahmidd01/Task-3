"""Evaluate frozen models and diagnose holdout residuals."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import SEED


def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {'mae': mean_absolute_error(actual, predicted), 'mse': mse,
            'rmse': np.sqrt(mse), 'r2': r2_score(actual, predicted)}


def evaluate_models(ranking, fitted, X_train, y_train, X_test, y_test):
    results = []
    for row in ranking.itertuples():
        model = fitted[row.model]
        train = regression_metrics(y_train, model.predict(X_train))
        test = regression_metrics(y_test, model.predict(X_test))
        adjusted = np.nan
        if row.ols and len(y_train) > row.feature_count + 1:
            adjusted = 1 - (1 - train['r2']) * (len(y_train) - 1) / (len(y_train) - row.feature_count - 1)
        results.append({'model': row.model, 'train_rmse': train['rmse'], 'train_r2': train['r2'],
                        'train_adjusted_r2_ols': adjusted, **{f'test_{k}': v for k, v in test.items()}})
    return ranking.merge(pd.DataFrame(results), on='model', validate='one_to_one')


def holdout_predictions(model, X_test, y_test):
    predicted = model.predict(X_test)
    result = X_test.copy().assign(Actual_Sales=y_test, Predicted_Sales=predicted)
    result['Prediction_Error'] = result['Actual_Sales'] - result['Predicted_Sales']
    result['Absolute_Error'] = result['Prediction_Error'].abs()
    return result.rename_axis('Source_Row_Zero_Based').reset_index()


def diagnose_predictions(model, X_test, y_test, predictions):
    result = permutation_importance(model, X_test, y_test, scoring='neg_root_mean_squared_error',
                                    n_repeats=40, random_state=SEED)
    importance = pd.DataFrame({'channel': X_test.columns, 'rmse_increase_mean': result.importances_mean,
                               'rmse_increase_std': result.importances_std})
    residual = predictions['Prediction_Error']
    correlation = spearmanr(predictions['Predicted_Sales'], residual)
    diagnostic = pd.DataFrame([{
        'mean_residual': residual.mean(), 'median_residual': residual.median(),
        'residual_vs_prediction_spearman': correlation.statistic,
        'residual_vs_prediction_p': correlation.pvalue,
        'largest_absolute_error': predictions['Absolute_Error'].max(),
        'largest_error_source_row': int(predictions.loc[predictions['Absolute_Error'].idxmax(), 'Source_Row_Zero_Based']),
    }])
    return importance, diagnostic
