"""Interactive dashboard for the saved Task 4 sales prediction model."""

import json

import joblib
import pandas as pd
import streamlit as st

from src import BUDGET_UNIT, CHANNELS, ROOT, SALES_UNIT
from src.scenario_analysis import budget_reference, predict_budgets

OUTPUT = ROOT / 'outputs'
MODEL = ROOT / 'models' / 'best_sales_model.joblib'
PROCESSED = ROOT / 'data' / 'processed' / 'advertising_cleaned.csv'
REFERENCE = OUTPUT / 'tables' / 'training_budget_reference.csv'
COMPARISON = OUTPUT / 'tables' / 'model_comparison.csv'
FINDINGS = OUTPUT / 'reports' / 'findings.json'
SCENARIOS = OUTPUT / 'predictions' / 'sample_budget_scenarios.csv'
PREDICTIONS = OUTPUT / 'predictions' / 'test_predictions.csv'


@st.cache_resource(show_spinner=False)
def load_model_and_reference(model_version, reference_version, data_version):
    data = pd.read_csv(PROCESSED)
    reference = budget_reference(pd.read_csv(REFERENCE), data[CHANNELS])
    return joblib.load(MODEL), data, reference


def figure(key, findings):
    st.image(str(OUTPUT / 'figures' / f'{key}.png'), width='stretch')
    st.caption(findings['captions'][key])


def simulator(model, reference, scenarios):
    st.header('Sales prediction simulator')
    st.write('Enter a proposed allocation, then compare the prediction with the training-median baseline.')
    with st.form('budget_scenario'):
        columns = st.columns(3)
        values = {
            channel: column.number_input(f'{channel} budget ({BUDGET_UNIT})', min_value=0.0,
                                         max_value=float(reference['maximum'][channel]) * 100,
                                         value=float(reference['median'][channel]), step=1.0, format='%.2f')
            for channel, column in zip(CHANNELS, columns)
        }
        submitted = st.form_submit_button('Predict sales', type='primary')
    if submitted or 'active_budget' not in st.session_state:
        st.session_state.active_budget = values
    proposal = predict_budgets(model, pd.DataFrame([st.session_state.active_budget]), reference)
    row = proposal.iloc[0]
    baseline = scenarios.iloc[0]['Predicted_Sales']
    result, total, difference = st.columns(3)
    result.metric('Predicted sales', f"{row['Predicted_Sales']:.2f}k units",
                  f"{row['Predicted_Sales'] - baseline:+.2f}k vs median budget")
    total.metric('Total spend', f"${row['Total']:.2f}k")
    difference.metric('Difference vs observed average', f"${row['Total_vs_Observed_Average']:+.2f}k")
    if not row['Inside_Training_Range']:
        st.warning(f"Outside the observed training range for {row['Outside_Training_Channels']}. Treat the result as extrapolation.")
    elif not row['Training_Mix_Supported']:
        st.warning('Individual budgets are in range, but few similar combinations appear in training.')
    else:
        st.success('The proposed mix is within the training range and near a recorded mix.')
    channel_table = pd.DataFrame({
        f'Proposed ({BUDGET_UNIT})': row[CHANNELS].astype(float),
        f'Observed mean ({BUDGET_UNIT})': reference['average'],
        'Share (%)': [100 * row[f'{channel}_share'] for channel in CHANNELS],
        f'Training minimum ({BUDGET_UNIT})': reference['minimum'],
        f'Training maximum ({BUDGET_UNIT})': reference['maximum'],
    })
    st.subheader('Budget distribution')
    st.dataframe(channel_table.round(2), width='stretch')
    st.bar_chart(channel_table.iloc[:, :2], horizontal=True, x_label=f'Budget ({BUDGET_UNIT})')
    proposal.insert(0, 'Scenario', 'Current proposal')
    proposal['Sales_Change'] = row['Predicted_Sales'] - baseline
    proposal['Budget_Change'] = row['Total'] - scenarios.iloc[0]['Total']
    combined = pd.concat([proposal, scenarios], ignore_index=True)
    st.subheader('Compare with prepared scenarios')
    st.dataframe(combined[['Scenario', *CHANNELS, 'Total', 'Predicted_Sales', 'Sales_Change',
                           'Training_Mix_Supported']].round(2), hide_index=True, width='stretch')
    st.download_button('Download scenario predictions', combined.to_csv(index=False).encode('utf-8'),
                       'budget_scenarios.csv', 'text/csv')
    st.caption('Predictions are conditional point estimates for comparable markets. Sales are measured in thousands of units.')


def main():
    st.set_page_config(page_title='Sales Prediction | Task 4', page_icon='📈', layout='wide')
    st.sidebar.markdown('### Task 4 · Advertising analytics')
    st.sidebar.caption('Kaggle Advertising CSV · Regression')
    page = st.sidebar.radio('Section', ['Overview', 'Advertising Analytics', 'Model Performance',
                                       'Sales Prediction Simulator', 'Business Recommendations'])
    st.title('Sales Prediction')
    st.caption('Market-level advertising analytics · TV, Radio and Newspaper')
    required = (MODEL, PROCESSED, REFERENCE, COMPARISON, FINDINGS, SCENARIOS, PREDICTIONS)
    if missing := [path.name for path in required if not path.is_file()]:
        st.error('Run `python main.py` to generate the dashboard artifacts. Missing ' + ', '.join(missing))
        st.stop()
    model, data, reference = load_model_and_reference(*(p.stat().st_mtime_ns for p in (MODEL, REFERENCE, PROCESSED)))
    findings = json.loads(FINDINGS.read_text(encoding='utf-8'))
    comparison = pd.read_csv(COMPARISON)
    winner = comparison.iloc[0]

    if page == 'Overview':
        st.header('Dataset and model overview')
        for column, label, value in zip(st.columns(4), ['Markets', 'Mean sales', 'Holdout RMSE', 'Holdout R²'],
                                        [len(data), f"{data['Sales'].mean():.2f}k units",
                                         f"{winner['test_rmse']:.3f}k units", f"{winner['test_r2']:.3f}"]):
            column.metric(label, value)
        st.write(findings['fit'])
        st.write(findings['cleaning'])
        left, right = st.columns(2)
        with left:
            figure('sales_distribution', findings)
        with right:
            figure('spend_mix', findings)
        st.subheader('Advertising spend summary')
        st.dataframe(data[CHANNELS].describe().T.round(2), width='stretch')
    elif page == 'Advertising Analytics':
        st.header('Advertising analytics')
        st.write(findings['channel_strength'])
        chart = st.selectbox('Relationship', ['TV', 'Radio', 'Newspaper', 'Total'])
        left, right = st.columns(2)
        with left:
            figure(f'{chart.lower()}_vs_sales', findings)
            figure('channel_boxplots', findings)
        with right:
            figure('correlation_heatmap', findings)
            figure('channel_marginal_effects', findings)
    elif page == 'Model Performance':
        st.header('Model performance')
        st.write(findings['fit'])
        st.write('Selected features', findings['selected_features'])
        st.dataframe(comparison[['model', 'cv_rmse_mean', 'cv_rmse_std', 'test_mae', 'test_rmse', 'test_r2']].round(3),
                     hide_index=True, width='stretch')
        left, right = st.columns(2)
        with left:
            figure('actual_vs_predicted', findings)
            figure('residuals_vs_predicted', findings)
        with right:
            figure('cross_validation_stability', findings)
        st.download_button('Download holdout predictions', PREDICTIONS.read_bytes(), PREDICTIONS.name, 'text/csv')
        st.caption(f'Holdout errors use {SALES_UNIT}. R² is a unitless measure of fit.')
    elif page == 'Sales Prediction Simulator':
        simulator(model, reference, pd.read_csv(SCENARIOS))
    else:
        st.header('Business recommendations')
        for i, recommendation in enumerate(findings['recommendations'], 1):
            st.markdown(f'{i}. {recommendation}')
        figure('tv_radio_response_surface', findings)
        with st.expander('Study limitations'):
            for limitation in findings['limitations']:
                st.write(limitation)


if __name__ == '__main__':
    main()
