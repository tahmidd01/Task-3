"""Reproducible advertising analytics and sales prediction."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['TV', 'Radio', 'Newspaper']
TARGET = 'Sales'
SEED = 42
BUDGET_UNIT = '$ thousands'
SALES_UNIT = 'thousand units'
