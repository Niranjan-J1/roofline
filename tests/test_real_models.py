"""Check formulas against real models, offline.

Each fixture holds a public config.json and a reference count worked out from the
model's weight-file headers. The adjustments made to that count are listed in the
fixture. The formulas skip biases and normalization weights, hence the tolerance.
"""

import json
from pathlib import Path

import pytest

from roofline.config import parse_config
from roofline.params import count_params

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))
TOLERANCE = 0.001


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_fixtures_exist():
    assert len(FIXTURES) >= 14


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_total_matches_real_model(path):
    data = load(path)
    total = count_params(parse_config(data["config"])).total
    assert total == pytest.approx(data["reference_params"], rel=TOLERANCE)
