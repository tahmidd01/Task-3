import hashlib
import json
from io import BytesIO
from zipfile import ZipFile

import joblib
import nbformat
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from src import CHANNELS, ROOT
from src.advertising_analysis import analyze_channels
from src.data_cleaning import clean_data
from src.data_loader import FILENAME, SOURCE, ensure_dataset, load_raw
from src.feature_engineering import budget_profile, make_features
from src.model_evaluation import regression_metrics
from src.model_training import candidates, split_data, train_models
from src.scenario_analysis import budget_reference, predict_budgets, sample_scenarios
from src.visualization import save_chart, set_theme

RAW = ROOT / 'data' / 'raw' / FILENAME
OUTPUT = ROOT / 'outputs'
MODEL = ROOT / 'models' / 'best_sales_model.joblib'


@pytest.fixture(scope='module')
def data():
    return clean_data(load_raw(RAW))[0]


@pytest.fixture(scope='module')
def saved_model():
    return joblib.load(MODEL)


def test_source_and_checksum():
    metadata = json.loads(RAW.with_name('dataset_metadata.json').read_text())
    assert metadata['source_url'] == SOURCE
    assert metadata['size_bytes'] == RAW.stat().st_size == 5166
    assert metadata['sha256'] == hashlib.sha256(RAW.read_bytes()).hexdigest()
    assert metadata['observations'] == 200
    assert ensure_dataset(RAW.parent) == RAW


def test_download_fallback_validates_and_preserves_bytes(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError('Simulated KaggleHub outage')

    import kagglehub

    archive = BytesIO()
    with ZipFile(archive, 'w') as handle:
        handle.writestr(FILENAME, RAW.read_bytes())
    monkeypatch.setattr(kagglehub, 'dataset_download', unavailable)
    monkeypatch.setattr('src.data_loader.urlopen', lambda *args, **kwargs: BytesIO(archive.getvalue()))
    assert ensure_dataset(tmp_path).read_bytes() == RAW.read_bytes()
    assert json.loads((tmp_path / 'dataset_metadata.json').read_text())['download_date']


def test_raw_schema_and_cleaning(data):
    raw = load_raw(RAW)
    assert raw.shape == (200, 5)
    assert data.shape == (200, 4)
    assert data.columns.tolist() == CHANNELS + ['Sales']
    assert data.ge(0).all().all() and data.notna().all().all()
    pd.testing.assert_frame_equal(data, raw[CHANNELS + ['Sales']].astype(float))


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match='Missing required columns'):
        load_raw(BytesIO(b'TV,Radio,Sales\n1,2,3\n'))


def test_normalization_invalid_rows_duplicates_and_missing_patterns():
    raw = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [' ', 2, 3, 4],
                        [2, 'oops', 3, 4], [1, 2, np.inf, 4], [1, 2, 3, -4]],
                       columns=[' tv ', 'RADIO', 'news-paper', 'SALES'])
    raw['Unnamed: 0'] = np.arange(1, len(raw) + 1)
    clean, report, patterns = clean_data(raw)
    assert len(clean) == 1
    assert report.iloc[-1][['missing', 'nonnumeric', 'infinite', 'negative']].tolist() == [1, 1, 1, 1]
    assert report.iloc[-1]['duplicate_rows_removed'] == 1
    assert patterns.set_index('missing_fields').loc['TV', 'rows'] == 1


def test_non_sequential_identifier_is_rejected(data):
    with pytest.raises(ValueError, match='not a sequential'):
        clean_data(data.assign(id=14))


def test_outliers_are_retained():
    raw = pd.DataFrame({'TV': range(8), 'Radio': [3] * 8, 'Newspaper': [0, 0, 0, 0, 0, 0, 0, 500], 'Sales': [4] * 8})
    clean, report, _ = clean_data(raw)
    assert len(clean) == 8
    assert report.set_index('scope').loc['Newspaper', 'iqr_outliers_retained'] == 1


def test_budget_features_including_zero_budget():
    frame = pd.DataFrame([[0, 0, 0], [10, 20, 10]], columns=CHANNELS)
    profile = budget_profile(frame)
    assert profile['Total'].tolist() == [0, 40]
    assert profile[['TV_share', 'Radio_share', 'Newspaper_share']].iloc[0].eq(0).all()
    assert profile.loc[1, 'Radio_share'] == .5
    quadratic = make_features(frame, 'Quadratic TV and Radio')
    assert quadratic.loc[1, 'TV_x_Radio'] == 200 and quadratic.loc[1, 'Radio_squared'] == 400
    assert 'Total' not in quadratic


def test_split_is_repeatable_disjoint_and_target_free(data):
    train, test, y_train, y_test = split_data(data)
    assert len(train) == len(y_train) == 160 and len(test) == len(y_test) == 40
    assert train.index.intersection(test.index).empty
    assert train.columns.tolist() == CHANNELS
    assert train.index.equals(y_train.index) and test.index.equals(y_test.index)
    pd.testing.assert_frame_equal(split_data(data)[0], train)


def test_training_and_scaling_use_training_only(data):
    train, test, y_train, _ = split_data(data)
    ranking, folds, models = train_models(train, y_train, repeats=1, model_specs=candidates()[:3])
    assert len(folds) == 15 and ranking['selected'].sum() == 1
    assert ranking.iloc[0]['model'] == 'Multiple Linear Regression'
    model = models['Multiple Linear Regression']
    np.testing.assert_allclose(model.named_steps['scale'].mean_, train.mean())
    assert len(model.predict(test)) == 40


def test_metrics():
    result = regression_metrics([2, 4, 6], [2, 3, 8])
    assert result == pytest.approx({'mae': 1, 'mse': 5 / 3, 'rmse': np.sqrt(5 / 3), 'r2': 3 / 8})


def test_channel_analysis(data):
    train, _, y_train, _ = split_data(data)
    coeff, fit = analyze_channels(train, y_train)
    assert coeff['term'].tolist() == ['const', *CHANNELS]
    assert (coeff['ci_95_low'] < coeff['ci_95_high']).all()
    assert coeff.dropna(subset=['vif'])['vif'].ge(1).all()
    assert fit.iloc[0]['n'] == len(train)


def test_saved_predictions_match_holdout(data, saved_model):
    _, test, _, y_test = split_data(data)
    exported = pd.read_csv(OUTPUT / 'predictions' / 'test_predictions.csv')
    np.testing.assert_allclose(exported['Actual_Sales'], y_test)
    np.testing.assert_allclose(exported['Predicted_Sales'], saved_model.predict(test))
    np.testing.assert_allclose(exported['Absolute_Error'], exported['Prediction_Error'].abs())


def test_scenarios_match_model_and_preserve_total_for_reallocation(data, saved_model):
    reference = budget_reference(split_data(data)[0], data[CHANNELS])
    scenarios = sample_scenarios(saved_model, reference).set_index('Scenario')
    np.testing.assert_allclose(scenarios['Predicted_Sales'], saved_model.predict(scenarios[CHANNELS]))
    assert scenarios.loc['TV +20%', 'TV'] == pytest.approx(reference['median']['TV'] * 1.2)
    assert scenarios.loc['Move 20% of Newspaper to Radio', 'Budget_Change'] == pytest.approx(0)
    zero = predict_budgets(saved_model, pd.DataFrame([[0., 0., 0.]], columns=CHANNELS), reference).iloc[0]
    assert zero['Total'] == zero['TV_share'] == 0
    assert not zero['Inside_Training_Range']


@pytest.mark.parametrize('bad_value', [-1, np.inf, np.nan, 'wrong'])
def test_invalid_budget_is_rejected(data, saved_model, bad_value):
    reference = budget_reference(split_data(data)[0], data[CHANNELS])
    with pytest.raises(ValueError):
        predict_budgets(saved_model, pd.DataFrame([[bad_value, 10, 10]], columns=CHANNELS), reference)


def test_figure_export(tmp_path):
    from matplotlib import pyplot as plt

    set_theme()
    figure, axes = plt.subplots()
    axes.plot([1, 2], [2, 4])
    path = save_chart(figure, tmp_path, 'test_chart')
    with Image.open(path) as image:
        assert image.format == 'PNG' and image.info['dpi'][0] == pytest.approx(300, abs=.1)
    charts = list((OUTPUT / 'figures').glob('*.png'))
    findings = json.loads((OUTPUT / 'reports' / 'findings.json').read_text(encoding='utf-8'))
    assert {chart.stem for chart in charts} == set(findings['captions'])
    assert len(charts) == 24


def test_pdf_sections_and_contents_page_numbers():
    pdf = PdfReader(OUTPUT / 'reports' / 'Sales_Prediction_Report.pdf')
    headings = ['Sales Prediction', 'Executive summary', 'Dataset and exploration', 'Feature and model selection',
                'Holdout model performance', 'Adjusted channel associations', 'Sales at proposed budgets',
                'Recommendations', 'Limitations and conclusion']
    assert len(pdf.pages) == len(headings)
    for number, (page, heading) in enumerate(zip(pdf.pages, headings), 1):
        text = page.extract_text()
        assert heading in text and f'{number:02d}' in text


def test_notebook_contains_executed_results():
    notebook = nbformat.read(ROOT / 'notebooks' / 'Sales_Prediction_Analysis.ipynb', as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in cells)
    assert all(output.output_type != 'error' for cell in cells for output in cell.outputs)
    assert any('image/png' in output.get('data', {}) for cell in cells for output in cell.outputs)


def test_streamlit_pages_and_scenario_submission():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception and app.metric[0].value == '200'
    for page in ['Advertising Analytics', 'Model Performance', 'Business Recommendations', 'Sales Prediction Simulator']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception and not app.error
    previous = app.metric[0].value
    app.number_input[0].set_value(1000.0)
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value != previous and 'extrapolation' in app.warning[0].value
    assert len(app.get('download_button')) == 1
