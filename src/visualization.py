"""Build consistent, publication-ready figures from pipeline results."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

from src import BUDGET_UNIT, CHANNELS, SALES_UNIT, TARGET

INK, BLUE, TEAL, GOLD, RED = '#172b42', '#315fc0', '#0b827f', '#a66715', '#b84c56'
COLORS = dict(zip(CHANNELS + ['Total', TARGET], [BLUE, TEAL, GOLD, '#6856a5', BLUE]))


def set_theme():
    sns.set_theme(style='whitegrid', context='notebook', font='DejaVu Sans', rc={
        'axes.spines.right': False, 'axes.spines.top': False, 'axes.labelcolor': INK,
        'axes.titlecolor': INK, 'axes.titleweight': 'bold', 'axes.titlepad': 13,
        'grid.color': '#e1e9f2', 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'savefig.facecolor': '#ffffff',
    })


def save_chart(figure, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{name}.png'
    figure.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.18)
    plt.close(figure)
    return path


def eda_charts(data, folder):
    for column in [TARGET, *CHANNELS]:
        fig, ax = plt.subplots(figsize=(7.3, 3.7), layout='constrained')
        sns.histplot(data[column], bins=16, kde=True, color=COLORS[column], ax=ax)
        ax.axvline(data[column].median(), color=INK, linestyle='--', label='Median')
        unit = SALES_UNIT if column == TARGET else BUDGET_UNIT
        ax.set(title=f'{column} distribution', xlabel=f'{column} ({unit})', ylabel='Markets')
        ax.legend(frameon=False)
        save_chart(fig, folder, f'{column.lower()}_distribution')

    fig, ax = plt.subplots(figsize=(7.3, 3.7), layout='constrained')
    sns.boxplot(data=data[CHANNELS], orient='h', palette=COLORS, hue_order=CHANNELS, ax=ax)
    ax.set(title='TV has the widest spending range', xlabel=f'Channel spend ({BUDGET_UNIT})', ylabel='')
    save_chart(fig, folder, 'channel_boxplots')

    for channel in CHANNELS + ['Total']:
        fig, ax = plt.subplots(figsize=(7.3, 3.7), layout='constrained')
        sns.regplot(data=data, x=channel, y=TARGET, ci=None, ax=ax,
                    scatter_kws={'alpha': .57, 's': 23, 'color': COLORS[channel]},
                    line_kws={'color': INK, 'linewidth': 2})
        ax.set(title=f'{channel} advertising and sales', xlabel=f'{channel} spend ({BUDGET_UNIT})',
               ylabel=f'Sales ({SALES_UNIT})')
        save_chart(fig, folder, f'{channel.lower()}_vs_sales')

    fig, ax = plt.subplots(figsize=(6, 4.4), layout='constrained')
    sns.heatmap(data[CHANNELS + ['Total', TARGET]].corr(), annot=True, vmin=-1, vmax=1,
                center=0, cmap='vlag', fmt='.2f', square=True, ax=ax, cbar_kws={'label': 'Pearson r'})
    ax.set_title('Correlation between sales and budgets')
    ax.tick_params(axis='x', rotation=22)
    ax.tick_params(axis='y', rotation=0)
    save_chart(fig, folder, 'correlation_heatmap')

    grid = sns.pairplot(data[CHANNELS + [TARGET]], corner=True, diag_kind='hist', height=1.7,
                        plot_kws={'color': BLUE, 'alpha': .52, 's': 14}, diag_kws={'color': TEAL})
    grid.figure.subplots_adjust(top=.91)
    grid.figure.suptitle('Pairwise views of advertising spend and sales', fontweight='bold')
    for ax, name in zip(grid.axes[-1], CHANNELS + [TARGET]):
        ax.set_xlabel(f'{name}\n({SALES_UNIT if name == TARGET else BUDGET_UNIT})', fontsize=8)
    for ax, name in zip(grid.axes[:, 0], CHANNELS + [TARGET]):
        if name != 'TV':
            ax.set_ylabel(f'{name}\n({SALES_UNIT if name == TARGET else BUDGET_UNIT})', fontsize=8)
    save_chart(grid.figure, folder, 'pairplot')


def model_charts(comparison, folds, predictions, importance, folder):
    for metric, label in [('rmse', 'RMSE'), ('mae', 'MAE'), ('r2', 'R²')]:
        fig, ax = plt.subplots(figsize=(8.7, 5.8), layout='constrained')
        colors = [TEAL if selected else BLUE for selected in comparison['selected']]
        ax.barh(comparison['model'], comparison[f'test_{metric}'], color=colors)
        ax.invert_yaxis()
        unit = '' if metric == 'r2' else f' ({SALES_UNIT})'
        ax.set(title=f'Holdout {label} by model', xlabel=f'{label}{unit}', ylabel='')
        save_chart(fig, folder, f'model_{metric}_comparison')

    fig, ax = plt.subplots(figsize=(8.8, 6.1), layout='constrained')
    sns.boxplot(data=folds, y='model', x='rmse', order=comparison['model'], color='#93c9d3', ax=ax)
    ax.set(title='Validation error across 25 training folds', xlabel=f'Fold RMSE ({SALES_UNIT})', ylabel='')
    save_chart(fig, folder, 'cross_validation_stability')

    actual, predicted, error = (predictions[column] for column in ('Actual_Sales', 'Predicted_Sales', 'Prediction_Error'))
    fig, ax = plt.subplots(figsize=(6.7, 4.0), layout='constrained')
    ax.scatter(actual, predicted, color=BLUE, alpha=.7, edgecolor='white', linewidth=.4)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(limits, limits, '--', color=INK, label='Exact prediction')
    ax.set(title='Actual and predicted holdout sales', xlabel=f'Actual ({SALES_UNIT})',
           ylabel=f'Predicted ({SALES_UNIT})')
    ax.legend(frameon=False)
    save_chart(fig, folder, 'actual_vs_predicted')

    fig, ax = plt.subplots(figsize=(6.7, 3.8), layout='constrained')
    sns.histplot(error, bins=13, color=TEAL, ax=ax)
    ax.axvline(0, linestyle='--', color=INK)
    ax.set(title='Holdout residual distribution', xlabel=f'Actual minus predicted ({SALES_UNIT})', ylabel='Markets')
    save_chart(fig, folder, 'residual_distribution')

    fig, ax = plt.subplots(figsize=(6.7, 3.8), layout='constrained')
    ax.scatter(predicted, error, color=BLUE, alpha=.73)
    ax.axhline(0, linestyle='--', color=INK)
    worst = predictions.loc[predictions['Absolute_Error'].idxmax()]
    ax.annotate(f"Row {int(worst['Source_Row_Zero_Based'])}",
                (worst['Predicted_Sales'], worst['Prediction_Error']),
                xytext=(7, 8 if worst['Prediction_Error'] < 0 else -15), textcoords='offset points', fontsize=9)
    ax.set(title='Holdout residuals versus predicted sales', xlabel=f'Predicted sales ({SALES_UNIT})',
           ylabel=f'Residual ({SALES_UNIT})')
    save_chart(fig, folder, 'residuals_vs_predicted')

    fig, ax = plt.subplots(figsize=(6.7, 3.5), layout='constrained')
    ax.barh(importance['channel'], importance['rmse_increase_mean'], xerr=importance['rmse_increase_std'],
            color=[COLORS[name] for name in importance['channel']], capsize=3)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, linewidth=.8)
    ax.set(title='Holdout permutation importance', xlabel=f'Increase in RMSE ({SALES_UNIT})', ylabel='')
    save_chart(fig, folder, 'permutation_importance')


def business_charts(data, coefficients, scenarios, grid, folder):
    channels = coefficients.set_index('term').loc[CHANNELS]
    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout='constrained')
    estimate = channels['coefficient'] * 10
    interval = [estimate - channels['ci_95_low'] * 10, channels['ci_95_high'] * 10 - estimate]
    ax.errorbar(estimate, CHANNELS, xerr=interval, color=BLUE, marker='o', linestyle='', capsize=4)
    ax.axvline(0, linestyle='--', color=INK)
    ax.invert_yaxis()
    ax.set(title='Adjusted sales association for $10,000 more spend',
           xlabel=f'OLS sales difference ({SALES_UNIT}) with HC3 95% CI', ylabel='')
    save_chart(fig, folder, 'channel_marginal_effects')

    means = data[CHANNELS].mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout='constrained')
    ax.bar(CHANNELS, means, color=[COLORS[name] for name in CHANNELS])
    for i, value in enumerate(means):
        ax.text(i, value + means.max() * .025, f'{value / means.sum():.1%} of aggregate spend', ha='center', fontsize=9)
    ax.set(title='Average advertising spend and aggregate budget shares', ylabel=f'Average spend ({BUDGET_UNIT})',
           ylim=(0, means.max() * 1.20))
    save_chart(fig, folder, 'spend_mix')

    fig, ax = plt.subplots(figsize=(8.2, 5.1), layout='constrained')
    colors = [TEAL if supported else GOLD for supported in scenarios['Training_Mix_Supported']]
    ax.barh(scenarios['Scenario'], scenarios['Sales_Change'], color=colors)
    ax.axvline(0, color=INK, linewidth=.8)
    ax.invert_yaxis()
    ax.set(title='Predicted change from the median-budget baseline', xlabel=f'Sales change ({SALES_UNIT})', ylabel='')
    ax.legend(handles=[Patch(color=TEAL, label='Supported mix'), Patch(color=GOLD, label='Limited support')],
              loc='upper center', bbox_to_anchor=(.5, -.18), ncols=2, frameon=False)
    save_chart(fig, folder, 'budget_scenarios')

    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout='constrained')
    for name, group in scenarios.groupby('Training_Mix_Supported'):
        ax.scatter(group['Total'], group['Predicted_Sales'], color=TEAL if name else GOLD,
                   label='Supported mix' if name else 'Limited support', s=48, alpha=.84)
    baseline = scenarios.iloc[0]
    ax.axhline(baseline['Predicted_Sales'], color=INK, linestyle='--', linewidth=1)
    ax.set(title='Predicted sales and total spend by budget mix', xlabel=f'Total spend ({BUDGET_UNIT})',
           ylabel=f'Predicted sales ({SALES_UNIT})')
    ax.legend(frameon=False)
    save_chart(fig, folder, 'budget_combinations')

    pivot = grid.pivot(index='Radio', columns='TV', values='Predicted_Sales')
    supported = grid.pivot(index='Radio', columns='TV', values='Training_Mix_Supported')
    fig, ax = plt.subplots(figsize=(7.4, 4.6), layout='constrained')
    masked = np.ma.masked_where(~supported.to_numpy(), pivot.to_numpy())
    contour = ax.contourf(pivot.columns, pivot.index, masked, levels=15, cmap='viridis', corner_mask=False)
    ax.set_facecolor('#e6ebf2')
    ax.grid(False)
    fig.colorbar(contour, ax=ax, label=f'Predicted sales ({SALES_UNIT})')
    ax.set(title=f"TV and Radio response at Newspaper = ${grid['Newspaper'].iloc[0]:.2f}k",
           xlabel=f'TV spend ({BUDGET_UNIT})', ylabel=f'Radio spend ({BUDGET_UNIT})')
    save_chart(fig, folder, 'tv_radio_response_surface')
