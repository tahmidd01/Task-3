# Sales Prediction Using Python

Task 4 estimates market-level product sales from TV, Radio and Newspaper advertising budgets. The project covers data cleaning, exploratory analysis, regression comparison, adjusted channel associations and a budget simulator.

## Run on Windows

From PowerShell, use Python 3.11 and the project virtual environment. The explicit executable path works without changing PowerShell's script execution policy.

```powershell
Set-Location 'D:\Task 3'
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' main.py
& '.\.venv\Scripts\python.exe' -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=240 '.\notebooks\Sales_Prediction_Analysis.ipynb'
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

Streamlit prints the dashboard URL. Its sidebar links to the dataset overview, channel analytics, model results, simulator and marketing recommendations. In the simulator, submit all three budgets and download a CSV that contains the proposal and the prepared alternatives. The notebook already contains executed results. To open it interactively after the pipeline runs, use `& '.\.venv\Scripts\python.exe' -m jupyterlab`.

## Data and provenance

The [Kaggle Advertising CSV](https://www.kaggle.com/datasets/bumba5341/advertisingcsv) was downloaded from Kaggle's official datasets API on November 9, 2026. The 5,166-byte `Advertising.csv` contains 200 observations and five fields. The [ISLR companion data](https://www.statlearning.com/s/Advertising.csv) match all numerical values after aligning header capitalization. [Figure 2.1 in the ISLR textbook](https://www.statlearning.com/s/ISLRSeventhPrinting.pdf) documents the units.

| Raw field | Role | Unit |
| --- | --- | --- |
| Unnamed: 0 | Sequential row identifier, excluded during cleaning | Row number |
| TV | Television advertising expenditure | Thousands of dollars |
| Radio | Radio advertising expenditure | Thousands of dollars |
| Newspaper | Newspaper advertising expenditure | Thousands of dollars |
| Sales | Regression target | Thousands of sales units |

The file contains no campaign dates, audience segments or product identifiers. It supports cross-sectional regression and proposed budget scenarios for comparable markets. Chronological and segment-level analyses require additional data.

`data/raw/dataset_metadata.json` records the original filename, source, retrieval date, size and SHA-256 checksum. The pipeline preserves and verifies the raw CSV. If the CSV is absent, it tries KaggleHub and then Kaggle's public archive endpoint. If Kaggle requests credentials, generate a token under **Settings → API** on Kaggle and call `kagglehub.login()` from an interactive Python session. Keep tokens outside the project. You can also manually extract the official `Advertising.csv` into `data/raw/`.

## Analysis and validation

- **Cleaning.** Normalize the column names, verify that the extra field is a sequential row index and validate nonnegative finite numeric values. The cleaning audit found no missing, invalid or duplicate observations. It retained all 200 markets, including two IQR-flagged Newspaper values.
- **Feature engineering.** Compare raw budgets, total spend, total with two spend shares, and selected TV and Radio interaction and quadratic terms. Define a zero-spend market's shares as zero. Keep total separate from its three raw components to avoid exact collinearity.
- **Model comparison.** Fit a mean dummy, simple and multiple linear regressions, OLS feature variants, Ridge, Lasso, ElasticNet, Random Forest, Extra Trees and Gradient Boosting. Use the same 25 folds (5 folds × 5 repeats) and small hyperparameter grids inside the 160-row training split. The seed is 42. Fit transforms within the pipeline, rank by mean validation RMSE and score the frozen candidates on the 40-row holdout.
- **Metrics.** MAE is the average absolute error. MSE averages squared errors. RMSE is its square root, on the sales scale. R² compares squared error with a mean prediction. Training adjusted R² applies to OLS fits and accounts for the number of terms. Fold SD measures variability across repeated splits.
- **Advertising associations.** Fit a separate training-only three-channel OLS model with HC3 robust standard errors and confidence intervals. Compare standardized slopes and VIF with the selected model's holdout permutation importance.
- **Scenario support.** Check each input against training minimum and maximum. Scale budget differences by channel range and flag proposals whose nearest observed mix is farther away than the 95th percentile of training nearest-neighbor distances.

## Results

Extra Trees selected TV and Radio. It achieved **0.582 mean CV RMSE** (fold SD 0.222), compared with 0.614 for the next-ranked quadratic OLS model. On the holdout, MAE was **0.381**, RMSE was **0.486** and R² was **0.993**. Error values are in thousands of sales units.

TV had the strongest standardized association and the largest increase in holdout RMSE when shuffled. Radio had the larger adjusted OLS association per dollar at the observed spending scale. The Newspaper slope's 95% interval included zero, and CV selected a model that omits that channel. Adding a TV × Radio interaction to the two-channel OLS reduced its validation RMSE from 1.649 to 0.943.

The training-median budget of **$197.45k** predicts **14.37k units**. A 20% increase in TV predicts **15.34k units**. Moving 20% of the median Newspaper budget to Radio predicts **15.37k units** at the same total cost. The model holds predicted sales constant when Newspaper alone changes because it uses TV and Radio. Audit Newspaper against reach and campaign objectives, and use controlled market tests to measure incremental sales before reallocating spending. The sample is small and observational, so unmeasured pricing, promotion and market differences may influence the results.

## Project files

| Path | Contents |
| --- | --- |
| `main.py` | Reproducible pipeline for data, models, charts, exports and PDF |
| `app.py` | Five-section Streamlit dashboard using the saved estimator |
| `src/` | Data loading, cleaning, exploration, features, model training and evaluation, association analysis, scenarios, visualization and reporting |
| `data/raw/` | Original Kaggle CSV and provenance metadata |
| `data/processed/` | Cleaned data and data-quality audit |
| `models/best_sales_model.joblib` | Selected scikit-learn pipeline |
| `notebooks/Sales_Prediction_Analysis.ipynb` | Executed analysis with results, charts and interpretations |
| `outputs/figures/` | 24 Matplotlib and Seaborn PNG charts at 300 DPI |
| `outputs/tables/` | Model comparison, fold scores, data profile, OLS results and diagnostics |
| `outputs/predictions/` | Holdout predictions, sample scenarios and the TV–Radio response grid |
| `outputs/reports/` | Submission PDF and machine-readable findings and figure captions |
| `outputs/logs/pipeline.log` | Execution log |
| `tests/test_analysis.py` | Dataset, modeling, export and dashboard regression checks |

The project uses Pandas, NumPy, SciPy, scikit-learn, Statsmodels, Joblib, Matplotlib, Seaborn, Jupyter, ReportLab, Streamlit and pytest.
