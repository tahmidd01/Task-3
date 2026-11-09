"""Produce auditable inspection, distribution and relationship tables."""

import pandas as pd
from scipy.stats import shapiro

from src import CHANNELS, TARGET, BUDGET_UNIT, SALES_UNIT
from src.feature_engineering import budget_profile


def describe_data(raw, clean):
    data = budget_profile(clean)
    numeric = data[CHANNELS + ['Total', TARGET]]
    descriptive = numeric.describe().T.join(numeric.skew().rename('skew')).join(
        numeric.kurt().rename('excess_kurtosis')
    )
    descriptive['shapiro_p'] = [shapiro(numeric[column]).pvalue for column in descriptive.index]
    profile = pd.DataFrame({
        'field': raw.columns, 'dtype': raw.dtypes.astype(str).to_numpy(),
        'unique': raw.nunique(dropna=True).to_numpy(), 'missing': raw.isna().sum().to_numpy(),
        'categorical': [not pd.api.types.is_numeric_dtype(dtype) for dtype in raw.dtypes],
    })
    dictionary = pd.DataFrame([
        ('TV', 'Predictor', 'Television advertising expenditure', BUDGET_UNIT),
        ('Radio', 'Predictor', 'Radio advertising expenditure', BUDGET_UNIT),
        ('Newspaper', 'Predictor', 'Newspaper advertising expenditure', BUDGET_UNIT),
        ('Sales', 'Target', 'Product sales across markets', SALES_UNIT),
        ('Total', 'Derived', 'Sum of the three channel budgets', BUDGET_UNIT),
        ('*_share', 'Derived', 'Channel budget divided by total, or zero for a zero budget', 'fraction'),
    ], columns=['field', 'role', 'description', 'unit'])
    if len(raw.columns) > len(CHANNELS) + 1:
        fields = [name for name in raw if name.lower().startswith('unnamed:') or name.lower() in {'id', 'index'}]
        for name in fields:
            dictionary.loc[len(dictionary)] = [name, 'Excluded', 'Sequential row identifier', 'ordinal']
    return data, {
        'raw_profile': profile, 'data_dictionary': dictionary,
        'raw_first_and_last': pd.concat([raw.head().assign(sample='First'), raw.tail().assign(sample='Last')]),
        'descriptive_statistics': descriptive.rename_axis('field').reset_index(),
        'correlations': numeric.corr().rename_axis('field').reset_index(),
        'engineered_budget_features': data,
    }
