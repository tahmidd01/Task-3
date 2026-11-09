"""Use candidate feature sets with explicit names and no redundant total."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src import CHANNELS


def budget_profile(frame):
    spend = frame[CHANNELS]
    total = spend.sum(axis=1)
    shares = spend.div(total.where(total.gt(0)), axis=0).fillna(0)
    return frame.assign(Total=total, **{f'{name}_share': shares[name] for name in CHANNELS})


def make_features(frame, spec):
    if spec == 'TV only':
        return frame[['TV']].copy()
    if spec == 'Total only':
        return budget_profile(frame)[['Total']]
    if spec == 'Total and two shares':
        return budget_profile(frame)[['Total', 'TV_share', 'Radio_share']]
    if spec == 'TV and Radio':
        return frame[['TV', 'Radio']].copy()
    if spec == 'Three channels':
        return frame[CHANNELS].copy()
    if spec == 'TV and Radio interaction':
        return frame[['TV', 'Radio']].assign(TV_x_Radio=frame['TV'] * frame['Radio'])
    if spec == 'Quadratic TV and Radio':
        return frame[['TV', 'Radio']].assign(
            TV_squared=frame['TV'] ** 2, Radio_squared=frame['Radio'] ** 2,
            TV_x_Radio=frame['TV'] * frame['Radio'],
        )
    if spec == 'Quadratic plus Newspaper':
        return make_features(frame, 'Quadratic TV and Radio').assign(Newspaper=frame['Newspaper'])
    raise ValueError(f'Unknown feature set: {spec}')


class BudgetFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, spec='Three channels'):
        self.spec = spec

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_numpy()
        self.output_features_ = make_features(X, self.spec).columns.to_numpy()
        return self

    def transform(self, X):
        check_is_fitted(self, 'output_features_')
        return make_features(X, self.spec)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'output_features_')
        return self.output_features_
