"""Acquire and validate the original Kaggle CSV with byte-level provenance."""

import hashlib
import json
import logging
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from src import CHANNELS, TARGET

SLUG = 'bumba5341/advertisingcsv'
SOURCE = f'https://www.kaggle.com/datasets/{SLUG}'
API = f'https://www.kaggle.com/api/v1/datasets/download/{SLUG}'
UNIT_SOURCE = 'https://www.statlearning.com/s/ISLRSeventhPrinting.pdf'
FILENAME = 'Advertising.csv'
REQUIRED = CHANNELS + [TARGET]


def canonical_columns(columns):
    known = {name.casefold(): name for name in REQUIRED}
    normalized = []
    for name in columns:
        cleaned = str(name).strip().lstrip('\ufeff').strip()
        key = re.sub(r'[\s_-]+', '', cleaned).casefold()
        normalized.append(known.get(key, cleaned))
    if len(normalized) != len(set(normalized)):
        raise ValueError('Ambiguous columns after normalization.')
    missing = set(REQUIRED) - set(normalized)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    return normalized


def load_raw(path):
    frame = pd.read_csv(path, encoding='utf-8-sig')
    canonical_columns(frame.columns)
    if frame.empty:
        raise ValueError('Advertising CSV is empty.')
    return frame


def write_metadata(raw_path, method, downloaded_on=None):
    frame = load_raw(raw_path)
    metadata = {
        'source_url': SOURCE, 'dataset_slug': SLUG, 'download_method': method,
        'download_date': downloaded_on, 'filename': raw_path.name,
        'size_bytes': raw_path.stat().st_size,
        'sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'observations': len(frame), 'columns': frame.columns.tolist(),
        'unit_reference': UNIT_SOURCE,
    }
    raw_path.with_name('dataset_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
    )


def download_csv(destination):
    try:
        import kagglehub

        with TemporaryDirectory(dir=destination.parent) as temporary:
            folder = Path(kagglehub.dataset_download(SLUG, output_dir=temporary))
            candidates = list(folder.rglob(FILENAME))
            if len(candidates) != 1:
                raise ValueError(f'Expected one {FILENAME} in the Kaggle download.')
            payload = candidates[0].read_bytes()
        load_raw(BytesIO(payload))
        return payload, 'KaggleHub dataset_download'
    except Exception as error:
        logging.warning('KaggleHub unavailable (%s). Trying the public Kaggle API.', type(error).__name__)

    try:
        with urlopen(API, timeout=45) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                payload = archive.read(FILENAME)
        load_raw(BytesIO(payload))
        return payload, 'Official Kaggle datasets download API'
    except Exception:
        raise RuntimeError(
            f'Could not download {SLUG}. Check Kaggle network access. If Kaggle requests '
            'authentication, generate a token in Kaggle Settings > API and run '
            'kagglehub.login() in an interactive Python session. You can also download '
            f'the archive from {SOURCE} and extract {FILENAME} into data/raw/.'
        ) from None


def ensure_dataset(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / FILENAME
    metadata_path = raw_dir / 'dataset_metadata.json'
    if path.exists():
        load_raw(path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if metadata['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Raw CSV checksum differs from its provenance record.')
        else:
            write_metadata(path, 'Existing local copy (download date unverified)')
        return path

    payload, method = download_csv(path)
    with path.open('xb') as stream:
        stream.write(payload)
    try:
        timestamp = datetime.now(ZoneInfo('Asia/Dhaka'))
    except ZoneInfoNotFoundError:
        timestamp = datetime.now().astimezone()
    write_metadata(path, method, timestamp.date().isoformat())
    return path
