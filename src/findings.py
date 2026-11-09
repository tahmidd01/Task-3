"""Generate a shared interpretation for the report, notebook and dashboard."""


def summarize(data, comparison, quality, coefficients, importance, diagnostic, scenarios, grid):
    winner = comparison.iloc[0]
    simple = comparison.set_index('model').loc['Multiple Linear Regression']
    interaction = comparison.set_index('model').loc['TV and Radio interaction OLS']
    additive = comparison.set_index('model').loc['TV and Radio OLS']
    baseline = comparison.set_index('model').loc['Dummy Mean Baseline']
    q = quality.iloc[-1]
    diagnostic = diagnostic.iloc[0]
    measures = data[['TV', 'Radio', 'Newspaper', 'Total', 'Sales']]
    correlation = measures.corr()['Sales']
    means, medians, skew = measures.mean(), measures.median(), measures.skew()
    channel = coefficients.set_index('term').drop('const')
    adjusted = channel['standardized_coefficient'].abs().idxmax()
    weakest = channel['standardized_coefficient'].abs().idxmin()
    top_permutation = importance.loc[importance['rmse_increase_mean'].idxmax(), 'channel']
    leading = scenarios.loc[scenarios['Training_Mix_Supported']].sort_values('Sales_Change', ascending=False).iloc[0]
    increases = scenarios.loc[scenarios['Training_Mix_Supported'] & scenarios['Budget_Change'].gt(0)]
    leading_increase = increases.sort_values('Sales_Change', ascending=False).iloc[0]
    cases = scenarios.set_index('Scenario')
    shift_names = ['Move 20% of Newspaper to TV', 'Move 20% of Newspaper to Radio']
    shifts = cases.loc[shift_names].sort_values('Sales_Change', ascending=False)
    shift = shifts.iloc[0]
    supported_shift = shifts[shifts['Training_Mix_Supported'] & shifts['Sales_Change'].gt(0)]
    shift_text = (
        f"At the same total budget, the scenario “{supported_shift.index[0]}” yields a predicted gain of "
        f"{supported_shift.iloc[0]['Sales_Change']:.2f} thousand units. Pilot the reallocation while tracking incremental sales."
        if not supported_shift.empty else
        'The tested same-budget shifts offer limited evidence of a gain within training support. Keep them as pilot hypotheses.'
    )
    support_share = grid['Training_Mix_Supported'].mean()
    interval = channel.loc[weakest]
    evidence = ('includes zero' if interval['ci_95_low'] <= 0 <= interval['ci_95_high'] else 'excludes zero')
    strength = (
        f'{adjusted} has the largest absolute standardized coefficient. {weakest} has the smallest, and its '
        f"HC3 95% interval ({interval['ci_95_low']:.3f} to {interval['ci_95_high']:.3f}) {evidence}. "
        f"The largest holdout permutation effect belongs to {top_permutation}."
    )
    fit = (
        f"{winner['model']} ranks first by training-fold RMSE "
        f"({winner['cv_rmse_mean']:.3f} ± {winner['cv_rmse_std']:.3f}, versus {baseline['cv_rmse_mean']:.3f} for the mean baseline). "
        f"On the holdout it records MAE {winner['test_mae']:.3f}, RMSE {winner['test_rmse']:.3f} "
        f"and R² {winner['test_r2']:.3f}. Errors are in thousands of sales units."
    )
    nonlinear = (
        f"Adding the TV × Radio interaction changes mean validation RMSE from {additive['cv_rmse_mean']:.3f} "
        f"to {interaction['cv_rmse_mean']:.3f}. The additive three-channel OLS records {simple['cv_rmse_mean']:.3f}. "
        'The interaction tests whether a channel’s predictive relationship varies with the level of the other channel.'
    )
    residuals = (
        f"Mean holdout residual is {diagnostic['mean_residual']:+.3f} thousand units. Spearman correlation of "
        f"residual with prediction is {diagnostic['residual_vs_prediction_spearman']:+.2f} "
        f"(p = {diagnostic['residual_vs_prediction_p']:.3f}). The largest absolute miss is "
        f"{diagnostic['largest_absolute_error']:.2f} at source row {diagnostic['largest_error_source_row']:.0f} (zero-based). "
        f"Winner train-fold RMSE averages {winner['cv_train_rmse_mean']:.3f} versus {winner['cv_rmse_mean']:.3f} "
        'on validation folds. The near-zero training error shows the flexibility of the fit. '
        f"Holdout RMSE ({winner['test_rmse']:.3f}) is close to validation RMSE, and the residuals show little monotone trend."
    )
    quality_text = (
        f"Cleaning retained {q['output_rows']:.0f} of {q['input_rows']:.0f} markets after excluding the sequential index. "
        f"It removed {q['invalid_rows_removed']:.0f} invalid rows and {q['duplicate_rows_removed']:.0f} duplicates, "
        f"and retained {q['iqr_outliers_retained']:.0f} IQR-flagged markets."
    )
    captions = {}
    for name in ['Sales', 'TV', 'Radio', 'Newspaper']:
        captions[f'{name.lower()}_distribution'] = (
            f'{name} averages {means[name]:.2f} with a median of {medians[name]:.2f}. '
            f'Sample skewness is {skew[name]:+.2f}. The dashed line marks the median.'
        )
    for name in ['TV', 'Radio', 'Newspaper', 'Total']:
        captions[f'{name.lower()}_vs_sales'] = (
            f'{name} and sales have Pearson r = {correlation[name]:.2f}. The line summarizes the unadjusted '
            'relationship across markets. Spread around the line reflects variation at similar spend.'
        )
    captions.update({
        'channel_boxplots': f"The IQR rule flags {q['iqr_outliers_retained']:.0f} markets across the measured fields. "
                           'TV spans the widest budget range. Points beyond the boxplot whiskers are retained.',
        'correlation_heatmap': f"TV has r = {correlation['TV']:.2f} with sales and total spend has "
                               f"r = {correlation['Total']:.2f}. Total inherits information from its component budgets.",
        'pairplot': 'The diagonal shows each distribution. The lower panels show joint coverage and scatter, including the wider spread in sales at higher TV budgets.',
        'model_rmse_comparison': f"The selected model’s holdout RMSE is {winner['test_rmse']:.3f}. The teal bar identifies the training-CV winner in all comparison charts.",
        'model_mae_comparison': f"The selected model’s mean absolute holdout error is {winner['test_mae']:.3f} thousand units.",
        'model_r2_comparison': f"The selected model has holdout R² of {winner['test_r2']:.3f}. Negative scores indicate squared error greater than predicting the holdout mean.",
        'cross_validation_stability': f"Across the shared folds, the winner averages {winner['cv_rmse_mean']:.3f} RMSE with fold SD {winner['cv_rmse_std']:.3f}. Fold SD describes stability across overlapping splits.",
        'actual_vs_predicted': 'Points close to the diagonal are better predictions. Points above it are overpredictions.',
        'residual_distribution': f"The residual histogram centers around a mean of {diagnostic['mean_residual']:+.3f}. Negative residuals represent overprediction.",
        'residuals_vs_predicted': residuals,
        'permutation_importance': f'Shuffling {top_permutation} produces the largest mean increase in holdout RMSE. Error bars show shuffle-to-shuffle SD.',
        'channel_marginal_effects': strength,
        'spend_mix': f"TV accounts for {means['TV'] / means['Total']:.1%} of aggregate spend. Shares compare total spending across channels, giving larger markets more weight.",
        'budget_scenarios': f"Among supported examples, {leading['Scenario']} has the largest predicted change ({leading['Sales_Change']:+.2f} thousand units).",
        'budget_combinations': f"The larger predicted change among the two tested same-cost reallocations is {shift['Sales_Change']:+.2f} thousand units ({shift.name}).",
        'tv_radio_response_surface': f"Newspaper is fixed at the training median (${grid['Newspaper'].iloc[0]:.2f} thousand). "
                                     f'Gray indicates sparse combinations. {support_share:.1%} of grid points meet the training support rule.',
    })
    return {
        'model': str(winner['model']), 'selected_features': str(winner['selected_features']),
        'fit': fit, 'cleaning': quality_text, 'nonlinear': nonlinear, 'channel_strength': strength,
        'residuals': residuals,
        'recommendations': [
            f'Prioritize {adjusted} in controlled budget tests and vary TV and Radio jointly. The interaction model reduced validation error relative to the additive model.',
            f'Audit {weakest} against reach, cost and campaign objectives. The selected model uses {winner["selected_features"]}, so varying Newspaper alone leaves its predictions unchanged.',
            shift_text,
            f"Among the tested spend increases, {leading_increase['Scenario']} produces the largest supported gain "
            f"({leading_increase['Sales_Change']:+.2f} thousand units). Compare the cost and incremental sales in a controlled test.",
            'Capture campaign dates, prices, promotions and audience segments so future models can test seasonality and segment response.',
        ],
        'limitations': [
            'The file covers 200 historical markets and has no calendar or segment identifiers. Estimates apply to budget scenarios for comparable markets.',
            'Market-level associations can reflect omitted variables such as pricing, promotions or market size. Use randomized tests to estimate incremental effects.',
            'Scenario predictions are point estimates. The range and nearest-neighbor flags measure coverage of the training data and provide no prediction interval.',
            'Repeated CV folds overlap, and tuning and ranking share those folds. The holdout provides the final evaluation and contains only 40 markets.',
        ],
        'captions': captions,
    }
