"""Typeset the analytical report with ReportLab."""

from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.utils import ImageReader

INK, BLUE, TEAL, MUTED, PALE = map(colors.HexColor, ['#172b42', '#315fc0', '#0b827f', '#496179', '#eef4fa'])
CONTENT = 491
REGULAR, BOLD = 'DejaVuSans', 'DejaVuSansBold'


def generate_report(path, figures, findings, metadata, tables, scenarios):
    fonts = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
    pdfmetrics.registerFont(TTFont(REGULAR, str(fonts / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont(BOLD, str(fonts / 'DejaVuSans-Bold.ttf')))
    styles = getSampleStyleSheet()
    body = ParagraphStyle('Body', parent=styles['BodyText'], fontName=REGULAR, fontSize=10,
                          leading=15, textColor=INK, spaceAfter=8)
    heading = ParagraphStyle('PageHeading', parent=body, fontName=BOLD, fontSize=25,
                             leading=31, spaceAfter=13, alignment=TA_LEFT)
    sub = ParagraphStyle('Subheading', parent=body, fontName=BOLD, fontSize=12,
                         leading=16, textColor=TEAL, spaceBefore=5, spaceAfter=6)
    small = ParagraphStyle('Caption', parent=body, fontSize=8.4, leading=12.3, textColor=MUTED, spaceAfter=9)
    kicker = ParagraphStyle('SectionLabel', parent=small, fontName=BOLD, textColor=TEAL,
                            fontSize=9.5, spaceAfter=5)
    story = []

    def paragraph(value, style=body):
        story.append(Paragraph(escape(str(value)), style))

    def section(label, title):
        paragraph(label.upper(), kicker)
        paragraph(title, heading)

    def image(name, width, height):
        file = figures / f'{name}.png'
        w, h = ImageReader(str(file)).getSize()
        scale = min(width / w, height / h)
        return Image(str(file), width=w * scale, height=h * scale, hAlign='LEFT')

    def figure(name, width=CONTENT, height=225):
        story.append(image(name, width, height))
        paragraph(findings['captions'][name], small)

    def figure_pair(first, second, height=155):
        width = (CONTENT - 16) / 2
        cells = [[image(name, width, height), Spacer(1, 4),
                  Paragraph(escape(findings['captions'][name]), small)] for name in (first, second)]
        chart_row = Table([[cells[0], '', cells[1]]], colWidths=[width, 16, width], hAlign='LEFT')
        chart_row.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                       ('LEFTPADDING', (0, 0), (-1, -1), 0),
                                       ('RIGHTPADDING', (0, 0), (-1, -1), 0)]))
        story.append(chart_row)

    def table(headers, rows, widths):
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), INK), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), BOLD), ('FONTNAME', (0, 1), (-1, -1), REGULAR),
            ('TEXTCOLOR', (0, 1), (-1, -1), INK), ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [PALE, colors.white]),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]
        flow = Table([headers, *rows], colWidths=widths, repeatRows=1, hAlign='LEFT')
        flow.setStyle(TableStyle(style))
        story.extend([flow, Spacer(1, 10)])

    def foot(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d5e2ef'))
        canvas.line(52, 48, A4[0] - 52, 48)
        canvas.setFillColor(MUTED)
        canvas.setFont(REGULAR, 8)
        canvas.drawString(52, 32, 'TASK 4  /  SALES PREDICTION USING PYTHON')
        canvas.drawRightString(A4[0] - 52, 32, f'{doc.page:02d}')
        canvas.restoreState()

    def cover(canvas, doc):
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(INK)
        canvas.rect(0, 240, width, height - 240, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor('#65d2c4'))
        canvas.rect(53, 657, 46, 4, fill=1, stroke=0)
        canvas.setFont(BOLD, 11)
        canvas.drawString(53, 679, 'TASK 4  /  DATA SCIENCE ASSIGNMENT')
        canvas.setFillColor(colors.white)
        canvas.setFont(BOLD, 50)
        canvas.drawString(50, 570, 'Sales Prediction')
        canvas.drawString(50, 502, 'Using Python')
        canvas.setFillColor(colors.HexColor('#c7d9ef'))
        canvas.setFont(REGULAR, 17)
        canvas.drawString(53, 441, 'Advertising analytics and budget scenarios')
        canvas.setStrokeColor(colors.HexColor('#57748f'))
        canvas.line(53, 338, width - 53, 338)
        canvas.setFont(BOLD, 10)
        canvas.drawString(53, 310, 'TV  /  RADIO  /  NEWSPAPER')
        canvas.setFillColor(TEAL)
        canvas.setFont(BOLD, 10)
        canvas.drawString(53, 187, 'DATA AND METHODS')
        canvas.setFillColor(INK)
        canvas.setFont(REGULAR, 13)
        canvas.drawString(53, 163, f"{metadata['observations']} markets  /  Regression and scenario analysis")
        canvas.drawString(53, 141, 'Source  /  Kaggle Advertising CSV')
        canvas.restoreState()
        foot(canvas, doc)

    story.extend([Spacer(1, 5), PageBreak()])
    section('Overview', 'Executive summary')
    paragraph('This study predicts market-level sales from spending on television, radio and newspaper advertising. '
              'It compares regression methods, evaluates adjusted channel associations and tests budget scenarios for comparable markets.')
    paragraph(findings['fit'])
    paragraph(findings['channel_strength'])
    paragraph('Introduction and problem statement', sub)
    paragraph('Marketing teams need to estimate how proposed channel allocations relate to sales before committing budget. '
              'The available data are cross-sectional. The prediction question is how much a comparable market might sell under a stated budget mix.')
    paragraph('Research objectives', sub)
    paragraph('Validate the original data, identify useful features, select a regression model using training folds, '
              'measure generalization on held-out markets and translate conditional predictions into testable marketing decisions.')
    paragraph('Contents', sub)
    table(['Section', 'Page'], [
        ('Executive summary, problem and objectives', '2'),
        ('Dataset, cleaning and exploration', '3'),
        ('Feature engineering, methods and model comparison', '4'),
        ('Holdout evaluation and diagnostics', '5'),
        ('Advertising effectiveness', '6'),
        ('Budget scenarios', '7'),
        ('Recommendations and response surface', '8'),
        ('Limitations, conclusion and references', '9'),
    ], [431, 60])

    story.append(PageBreak())
    section('01  /  Data', 'Dataset and exploration')
    paragraph(f"{metadata['filename']} contains {metadata['observations']} markets and {len(metadata['columns'])} raw columns. "
              'TV, Radio and Newspaper record spend in thousands of dollars. Sales records thousands of units. '
              'The source is the Kaggle Advertising CSV distribution. The ISLR textbook documents the units, '
              'and its companion CSV matches the Kaggle data after normalizing column capitalization.')
    paragraph('Data cleaning and preprocessing', sub)
    paragraph(findings['cleaning'])
    paragraph('Column names were normalized, then values checked for missing, nonnumeric, infinite and negative entries. '
              'The IQR flag identifies observations for review. Original records and a separate cleaning audit were preserved.')
    stats = tables['descriptive_statistics']
    table(['Field', 'Mean', 'Median', 'SD', 'Min', 'Max', 'Skew'], [
        [row.field, f'{row.mean:.2f}', f'{stats.iloc[i]["50%"]:.2f}', f'{row.std:.2f}', f'{row.min:.2f}',
         f'{row.max:.2f}', f'{row.skew:+.2f}'] for i, row in enumerate(stats.itertuples(index=False))
    ], [113, 63, 63, 63, 63, 63, 63])
    paragraph('Distributions and relationships', sub)
    figure_pair('sales_distribution', 'correlation_heatmap')
    sales = stats.set_index('field').loc['Sales']
    paragraph(f"The middle half of sales spans {sales['25%']:.2f} to {sales['75%']:.2f} thousand units. "
              f"The sales Shapiro-Wilk p-value is {sales['shapiro_p']:.3g}. Predictor distributions need not be normal for OLS. "
              'The descriptive CSV provides skewness, excess kurtosis, quartiles and normality checks for all numeric variables.')

    story.append(PageBreak())
    section('02  /  Methods', 'Feature and model selection')
    paragraph('The predictors were split randomly into 160 training markets and 40 holdout markets using seed 42. '
              'All candidates used identical 5-fold cross-validation repeated five times. Feature transforms and any scaling were inside each pipeline.')
    paragraph('Feature engineering and selection', sub)
    paragraph('Candidate sets compare raw channels, total spend, total with two channel shares, TV and Radio alone, '
              'and selected interaction and quadratic terms. Total was kept separate from its three components to avoid exact collinearity. '
              'Mean validation RMSE ranks the models. Compact parameter grids tune regularized and ensemble models.')
    comparison = tables['model_comparison']
    table(['Candidate', 'CV RMSE', 'CV SD', 'Test MAE', 'Test RMSE', 'Test R²'], [
        [row.model, f'{row.cv_rmse_mean:.3f}', f'{row.cv_rmse_std:.3f}', f'{row.test_mae:.3f}',
         f'{row.test_rmse:.3f}', f'{row.test_r2:.3f}'] for row in comparison.itertuples()
    ], [176, 61, 55, 65, 69, 65])
    paragraph(f"Selected inputs and derived terms include {findings['selected_features'].replace('_', ' ')}. "
              'The ranking uses training CV. Holdout results are shown for all frozen candidates for comparison.', small)
    paragraph(findings['nonlinear'])

    story.append(PageBreak())
    section('03  /  Validation', 'Holdout model performance')
    paragraph(findings['fit'])
    paragraph('MAE is the average absolute miss. MSE averages squared misses, and its square root gives RMSE in the original sales scale. '
              'R² describes squared-error improvement over predicting the evaluation-set mean. Adjusted R² is reported for training OLS fits and penalizes added terms.')
    figure_pair('actual_vs_predicted', 'residuals_vs_predicted')
    figure('cross_validation_stability', width=CONTENT, height=229)

    story.append(PageBreak())
    section('04  /  Advertising', 'Adjusted channel associations')
    paragraph('A separate OLS model uses the three original channel budgets and training markets. Each slope estimates the sales difference '
              'associated with one additional thousand dollars in a channel, holding the other budgets fixed. HC3 standard errors address unequal residual variance.')
    coeff = tables['ols_coefficients']
    table(['Term', 'Slope', 'HC3 SE', 'p-value', '95% interval', 'VIF'], [
        ['Intercept' if row.term == 'const' else row.term, f'{row.coefficient:.4f}', f'{row.hc3_std_error:.4f}',
         f'{row.p_value:.2g}', f'{row.ci_95_low:.4f} to {row.ci_95_high:.4f}',
         '—' if row.term == 'const' else f'{row.vif:.2f}'] for row in coeff.itertuples()
    ], [112, 70, 67, 63, 126, 53])
    ols = tables['ols_fit'].iloc[0]
    paragraph(f"Training OLS R² is {ols['r2']:.3f}, adjusted R² is {ols['adjusted_r2']:.3f}, and robust F is "
              f"{ols['hc3_f_statistic']:.1f} (p = {ols['hc3_f_p_value']:.2g}). The highest VIF is {coeff['vif'].max():.2f}, "
              'suggesting little variance inflation in this specification.')
    figure('channel_marginal_effects', height=213)
    figure('permutation_importance', height=158)

    story.append(PageBreak())
    section('05  /  Scenarios', 'Sales at proposed budgets')
    paragraph('The baseline uses the training median of each budget. The saved model scores the higher-spend and constant-budget alternatives. '
              'Spend is in thousands of dollars, and sales and changes are in thousands of units.')
    table(['Scenario', 'Total spend', 'Pred. sales', 'Sales change', 'Support'], [
        [row.Scenario, f'{row.Total:.2f}', f'{row.Predicted_Sales:.2f}', f'{row.Sales_Change:+.2f}',
         'Within' if row.Training_Mix_Supported else 'Limited'] for row in scenarios.itertuples()
    ], [196, 74, 73, 78, 70])
    paragraph('The simulator checks each channel’s training min and max. It also compares the proposal’s nearest training mix with the 95th '
              'percentile of training nearest-neighbor distances, after scaling channels by their training ranges.')
    figure('budget_scenarios', height=218)

    story.append(PageBreak())
    section('06  /  Marketing', 'Recommendations')
    for i, recommendation in enumerate(findings['recommendations'], start=1):
        paragraph(f'{i:02d}  {recommendation}')
    paragraph('TV and Radio response surface', sub)
    figure('tv_radio_response_surface', height=258)

    story.append(PageBreak())
    section('07  /  Interpretation', 'Limitations and conclusion')
    for limitation in findings['limitations']:
        paragraph(limitation)
    paragraph('Conclusion', sub)
    paragraph('The holdout results quantify how closely advertising budgets predict sales in this sample. '
              'The scenario simulator translates those relationships into comparable budget options and flags unfamiliar mixes. '
              'Controlled marketing tests can establish whether the strongest predicted changes translate into incremental sales.')
    paragraph('References', sub)
    references = [
        ('Bumba5341. Advertising.csv. Kaggle. Dataset source.', metadata['source_url']),
        ('James, Witten, Hastie and Tibshirani. An Introduction to Statistical Learning, 2013. Figure 2.1 and Chapter 3.',
         metadata['unit_reference']),
        ('Scikit-learn user guide. Cross-validation and model evaluation.',
         'https://scikit-learn.org/stable/modules/cross_validation.html'),
        ('Statsmodels documentation. Ordinary least squares and robust covariance.',
         'https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html'),
    ]
    for label, url in references:
        story.append(Paragraph(f'<link href="{escape(url)}" color="#315fc0">{escape(label)}</link>', body))
    paragraph(f"The raw CSV was retrieved on {metadata['download_date'] or 'an unverified date'}. "
              'File size and SHA-256 are recorded in data/raw/dataset_metadata.json.', small)

    document = SimpleDocTemplate(str(path), pagesize=A4, topMargin=18 * mm, bottomMargin=21 * mm,
                                 leftMargin=46, rightMargin=46, title='Sales Prediction Using Python',
                                 author='Task 4 | Advertising Analytics', pageCompression=1)
    document.build(story, onFirstPage=cover, onLaterPages=foot)
