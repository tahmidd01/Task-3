"""Describe adjusted channel associations with robust OLS inference."""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import CHANNELS


def analyze_channels(X_train, y_train):
    design = sm.add_constant(X_train, has_constant='add')
    model = sm.OLS(y_train, design).fit(cov_type='HC3')
    interval = model.conf_int()
    channels = pd.DataFrame({
        'coefficient': model.params, 'hc3_std_error': model.bse, 'p_value': model.pvalues,
        'ci_95_low': interval[0], 'ci_95_high': interval[1],
    })
    channels['standardized_coefficient'] = model.params * X_train.std() / y_train.std()
    channels['vif'] = pd.Series({name: variance_inflation_factor(design.to_numpy(), design.columns.get_loc(name))
                                 for name in CHANNELS})
    channels['sales_change_per_10_budget_units'] = channels['coefficient'].where(channels.index != 'const') * 10
    fit = pd.DataFrame([{'r2': model.rsquared, 'adjusted_r2': model.rsquared_adj,
                         'hc3_f_statistic': model.fvalue, 'hc3_f_p_value': model.f_pvalue,
                         'n': int(model.nobs)}])
    return channels.rename_axis('term').reset_index(), fit
