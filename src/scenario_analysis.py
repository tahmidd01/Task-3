"""Score proposed budgets with the persisted model and training support checks."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src import CHANNELS
from src.feature_engineering import budget_profile


def budget_reference(training, observed):
    lower, upper = training.min(), training.max()
    scale = (upper - lower).replace(0, 1)
    coordinates = (training - lower) / scale
    neighbors = NearestNeighbors(n_neighbors=2).fit(coordinates)
    nearest_other = neighbors.kneighbors(coordinates, return_distance=True)[0][:, 1]
    return {'minimum': lower, 'maximum': upper, 'median': training.median(),
            'average': observed.mean(), 'scale': scale, 'neighbors': neighbors,
            'distance_limit': float(np.quantile(nearest_other, .95))}


def predict_budgets(model, budgets, reference):
    inputs = budgets[CHANNELS].apply(pd.to_numeric, errors='raise').astype(float)
    if inputs.empty or not np.isfinite(inputs).to_numpy().all() or inputs.lt(0).to_numpy().any():
        raise ValueError('Provide at least one finite, nonnegative budget for each channel.')
    result = budget_profile(inputs)
    result['Predicted_Sales'] = model.predict(inputs)
    result['Total_vs_Observed_Average'] = result['Total'] - reference['average'].sum()
    outside = inputs.lt(reference['minimum']) | inputs.gt(reference['maximum'])
    result['Outside_Training_Channels'] = outside.apply(lambda row: ', '.join(row.index[row]), axis=1)
    result['Inside_Training_Range'] = ~outside.any(axis=1)
    normalized = (inputs - reference['minimum']) / reference['scale']
    distance = reference['neighbors'].kneighbors(normalized, n_neighbors=1, return_distance=True)[0][:, 0]
    result['Nearest_Training_Distance'] = distance
    result['Training_Mix_Supported'] = result['Inside_Training_Range'] & (distance <= reference['distance_limit'])
    return result


def sample_scenarios(model, reference):
    base = reference['median']
    proposals = {'Training median baseline': base}
    for channel in CHANNELS:
        for increase in (.10, .20):
            proposed = base.copy()
            proposed[channel] *= 1 + increase
            proposals[f'{channel} +{increase:.0%}'] = proposed
    proposals['All channels +10%'] = base * 1.1
    proposals['TV and Radio +10%'] = base * pd.Series({'TV': 1.1, 'Radio': 1.1, 'Newspaper': 1.0})
    for channel in ('TV', 'Radio'):
        proposal = base.copy()
        shift = .2 * proposal['Newspaper']
        proposal['Newspaper'] -= shift
        proposal[channel] += shift
        proposals[f'Move 20% of Newspaper to {channel}'] = proposal
    scenarios = predict_budgets(model, pd.DataFrame(proposals).T, reference)
    scenarios['Sales_Change'] = scenarios['Predicted_Sales'] - scenarios.iloc[0]['Predicted_Sales']
    scenarios['Budget_Change'] = scenarios['Total'] - scenarios.iloc[0]['Total']
    return scenarios.rename_axis('Scenario').reset_index()


def response_grid(model, reference, points=45):
    tv = np.linspace(reference['minimum']['TV'], reference['maximum']['TV'], points)
    radio = np.linspace(reference['minimum']['Radio'], reference['maximum']['Radio'], points)
    budgets = pd.DataFrame(product(tv, radio), columns=['TV', 'Radio'])
    budgets['Newspaper'] = reference['median']['Newspaper']
    return predict_budgets(model, budgets, reference)
