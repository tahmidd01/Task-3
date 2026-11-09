"""Tune and rank estimators on a shared set of training-only folds."""

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src import CHANNELS, SEED, TARGET
from src.feature_engineering import BudgetFeatures

SCORING = {'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error',
           'mse': 'neg_mean_squared_error', 'r2': 'r2'}


@dataclass
class Candidate:
    name: str
    features: str
    estimator: object
    scale: bool = False
    grid: dict = field(default_factory=dict)
    ols: bool = False

    def pipeline(self):
        return Pipeline([
            ('features', BudgetFeatures(self.features)),
            ('scale', StandardScaler() if self.scale else 'passthrough'),
            ('model', self.estimator),
        ])


def candidates():
    linear = [
        ('Simple Linear Regression', 'TV only'),
        ('Multiple Linear Regression', 'Three channels'),
        ('Total spend OLS', 'Total only'),
        ('Total and mix OLS', 'Total and two shares'),
        ('TV and Radio OLS', 'TV and Radio'),
        ('TV and Radio interaction OLS', 'TV and Radio interaction'),
        ('Quadratic TV and Radio OLS', 'Quadratic TV and Radio'),
        ('Quadratic plus Newspaper OLS', 'Quadratic plus Newspaper'),
    ]
    regularized = [
        ('Ridge Regression', Ridge(), {'model__alpha': [.01, .1, 1, 10]}),
        ('Lasso Regression', Lasso(max_iter=10000), {'model__alpha': [.001, .01, .1, .5]}),
        ('ElasticNet Regression', ElasticNet(max_iter=10000),
         {'model__alpha': [.001, .01, .1], 'model__l1_ratio': [.25, .75]}),
    ]
    ensembles = [
        ('Random Forest', RandomForestRegressor(n_estimators=120, random_state=SEED),
         {'model__min_samples_leaf': [1, 3], 'model__max_features': [1.0]}),
        ('Extra Trees', ExtraTreesRegressor(n_estimators=120, random_state=SEED),
         {'model__min_samples_leaf': [1, 3], 'features__spec': ['Three channels', 'TV and Radio']}),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=160, learning_rate=.05, random_state=SEED),
         {'model__max_depth': [2, 3], 'features__spec': ['Three channels', 'TV and Radio']}),
    ]
    return [
        Candidate('Dummy Mean Baseline', 'Three channels', DummyRegressor()),
        *(Candidate(name, features, LinearRegression(), scale=True, ols=True) for name, features in linear),
        *(Candidate(name, 'Three channels', model, scale=True, grid=grid) for name, model, grid in regularized),
        *(Candidate(name, 'Three channels', model, grid=grid) for name, model, grid in ensembles),
    ]


def split_data(clean):
    return train_test_split(clean[CHANNELS], clean[TARGET], test_size=.2, random_state=SEED)


def train_models(X_train, y_train, repeats=5, model_specs=None):
    cv = list(RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=SEED).split(X_train))
    rows, fold_rows, fitted = [], [], {}
    with threadpool_limits(limits=1):
        for candidate in model_specs if model_specs is not None else candidates():
            logging.info('Cross-validating %s', candidate.name)
            search = GridSearchCV(candidate.pipeline(), candidate.grid, scoring=SCORING,
                                  cv=cv, refit='rmse', error_score='raise', return_train_score=True)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            fitted[candidate.name] = model
            transforms = model.named_steps['features']
            names = transforms.get_feature_names_out().tolist()
            row = {'model': candidate.name, 'feature_set': transforms.spec,
                   'selected_features': ', '.join(names), 'feature_count': len(names),
                   'parameters': json.dumps(search.best_params_, sort_keys=True), 'ols': candidate.ols}
            for metric in SCORING:
                sign = 1 if metric == 'r2' else -1
                values = search.cv_results_
                i = search.best_index_
                row[f'cv_{metric}_mean'] = sign * values[f'mean_test_{metric}'][i]
                row[f'cv_{metric}_std'] = values[f'std_test_{metric}'][i]
            row['cv_train_rmse_mean'] = -values['mean_train_rmse'][i]
            rows.append(row)
            for fold in range(len(cv)):
                scores = {metric: (1 if metric == 'r2' else -1) * values[f'split{fold}_test_{metric}'][i]
                          for metric in SCORING}
                fold_rows.append({'model': candidate.name, 'repeat': fold // 5 + 1,
                                  'fold': fold % 5 + 1, **scores})
    ranking = pd.DataFrame(rows).sort_values(['cv_rmse_mean', 'feature_count', 'cv_mae_mean'], kind='stable')
    ranking = ranking.reset_index(drop=True)
    ranking.insert(0, 'rank', np.arange(len(ranking)) + 1)
    ranking.insert(1, 'selected', ranking['rank'].eq(1))
    return ranking, pd.DataFrame(fold_rows), fitted
