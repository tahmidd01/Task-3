"""Run the complete Sales Prediction pipeline from the project root."""

import argparse
import json
import logging
from pathlib import Path

import joblib

from src import ROOT
from src.advertising_analysis import analyze_channels
from src.data_cleaning import clean_data
from src.data_loader import ensure_dataset, load_raw
from src.exploratory_analysis import describe_data
from src.findings import summarize
from src.model_evaluation import diagnose_predictions, evaluate_models, holdout_predictions
from src.model_training import split_data, train_models
from src.report_generator import generate_report
from src.scenario_analysis import budget_reference, response_grid, sample_scenarios
from src.visualization import business_charts, eda_charts, model_charts, set_theme


def run_pipeline(root=ROOT):
    processed = root / 'data' / 'processed'
    outputs = root / 'outputs'
    figures, tables, predictions, reports, logs = (outputs / name for name in
                                                  ('figures', 'tables', 'predictions', 'reports', 'logs'))
    models = root / 'models'
    for folder in (processed, figures, tables, predictions, reports, logs, models):
        folder.mkdir(parents=True, exist_ok=True)

    logging.info('Verifying Kaggle source data and preparing the cleaned table')
    raw_path = ensure_dataset(root / 'data' / 'raw')
    raw = load_raw(raw_path)
    clean, quality, patterns = clean_data(raw)
    clean.to_csv(processed / 'advertising_cleaned.csv', index=False)
    quality.to_csv(processed / 'data_quality_report.csv', index=False)

    logging.info('Profiling the data and creating feature summaries')
    data, exports = describe_data(raw, clean)
    exports['missing_value_patterns'] = patterns
    X_train, X_test, y_train, y_test = split_data(clean)
    logging.info('Splitting %d training and %d holdout observations', len(X_train), len(X_test))
    exports['training_budget_reference'] = X_train

    ranking, folds, fitted = train_models(X_train, y_train)
    winner = ranking.iloc[0]['model']
    logging.info('Selected %s by training cross-validation; evaluating frozen models', winner)
    model_path = models / 'best_sales_model.joblib'
    joblib.dump(fitted[winner], model_path, compress=3)
    model = joblib.load(model_path)
    comparison = evaluate_models(ranking, fitted, X_train, y_train, X_test, y_test)
    test = holdout_predictions(model, X_test, y_test)
    importance, diagnostic = diagnose_predictions(model, X_test, y_test, test)

    logging.info('Estimating channel associations and predicting budget scenarios with the saved pipeline')
    coefficients, ols_fit = analyze_channels(X_train, y_train)
    reference = budget_reference(X_train, clean[X_train.columns])
    scenarios = sample_scenarios(model, reference)
    grid = response_grid(model, reference)
    exports.update(model_comparison=comparison, cross_validation_folds=folds, residual_diagnostics=diagnostic,
                   permutation_importance=importance, ols_coefficients=coefficients, ols_fit=ols_fit,
                   training_budget_ranges=X_train.agg(['min', 'median', 'max']).T.rename_axis('channel').reset_index())
    for name, table in exports.items():
        table.to_csv(tables / f'{name}.csv', index=False)
    for name, table in [('test_predictions', test), ('sample_budget_scenarios', scenarios), ('response_grid', grid)]:
        table.to_csv(predictions / f'{name}.csv', index=False)

    logging.info('Rendering EDA, model and business visualizations')
    set_theme()
    eda_charts(data, figures)
    model_charts(comparison, folds, test, importance, figures)
    business_charts(data, coefficients, scenarios, grid, figures)
    findings = summarize(data, comparison, quality, coefficients, importance, diagnostic, scenarios, grid)
    (reports / 'findings.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    logging.info('Building the final PDF report')
    metadata = json.loads(raw_path.with_name('dataset_metadata.json').read_text(encoding='utf-8'))
    generate_report(reports / 'Sales_Prediction_Report.pdf', figures, findings, metadata, exports, scenarios)
    logging.info('Complete. Outputs written to %s', outputs)
    return comparison.iloc[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Project root containing data/raw')
    root = parser.parse_args().root.resolve()
    log = root / 'outputs' / 'logs' / 'pipeline.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.FileHandler(log, encoding='utf-8'), logging.StreamHandler()])
    try:
        run_pipeline(root)
    except Exception:
        logging.exception('Sales prediction pipeline failed')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
