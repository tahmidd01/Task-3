"""Validate numeric fields and record every row exclusion and IQR flag."""

import numpy as np
import pandas as pd

from src.data_loader import REQUIRED, canonical_columns


def clean_data(raw):
    normalized = raw.set_axis(canonical_columns(raw.columns), axis=1)
    extras = normalized.columns.difference(REQUIRED).tolist()
    if extras:
        if len(extras) != 1:
            raise ValueError(f'Unexpected columns: {extras}')
        identifier = extras[0]
        row_numbers = pd.to_numeric(normalized[identifier], errors='coerce').to_numpy()
        positions = np.arange(len(raw))
        index_like = identifier.casefold().startswith('unnamed:') or identifier.casefold() in {'index', 'id'}
        if not index_like or not (np.array_equal(row_numbers, positions) or
                                  np.array_equal(row_numbers, positions + 1)):
            raise ValueError(f'Unexpected field {identifier!r} is not a sequential row index.')

    data = normalized[REQUIRED].map(lambda value: np.nan if isinstance(value, str) and not value.strip() else value)
    numeric = data.apply(pd.to_numeric, errors='coerce').astype(float)
    missing = data.isna()
    nonnumeric = numeric.isna() & ~missing
    infinite = pd.DataFrame(np.isinf(numeric), index=data.index, columns=REQUIRED)
    negative = numeric.lt(0) & ~infinite
    invalid = (missing | nonnumeric | infinite | negative).any(axis=1)
    valid = numeric.loc[~invalid]
    duplicates = valid.duplicated()
    cleaned = valid.loc[~duplicates].copy()
    if cleaned.empty:
        raise ValueError('No valid advertising observations remain.')
    q1, q3 = cleaned.quantile(.25), cleaned.quantile(.75)
    iqr = q3 - q1
    outlier = cleaned.lt(q1 - 1.5 * iqr) | cleaned.gt(q3 + 1.5 * iqr)
    records = []
    for column in REQUIRED:
        records.append({
            'scope': column, 'missing': int(missing[column].sum()),
            'nonnumeric': int(nonnumeric[column].sum()), 'infinite': int(infinite[column].sum()),
            'negative': int(negative[column].sum()), 'iqr_outliers_retained': int(outlier[column].sum()),
            'lower_iqr_fence': float(q1[column] - 1.5 * iqr[column]),
            'upper_iqr_fence': float(q3[column] + 1.5 * iqr[column]),
        })
    records.append({
        'scope': 'All rows', 'missing': int(missing.any(axis=1).sum()),
        'nonnumeric': int(nonnumeric.any(axis=1).sum()), 'infinite': int(infinite.any(axis=1).sum()),
        'negative': int(negative.any(axis=1).sum()), 'iqr_outliers_retained': int(outlier.any(axis=1).sum()),
        'input_rows': len(raw), 'invalid_rows_removed': int(invalid.sum()),
        'duplicate_rows_removed': int(duplicates.sum()), 'output_rows': len(cleaned),
        'excluded_index_field': ', '.join(extras),
    })
    patterns = missing.apply(lambda row: ', '.join(row.index[row]) or 'Complete', axis=1)
    return cleaned, pd.DataFrame(records), patterns.value_counts().rename_axis('missing_fields').reset_index(name='rows')
