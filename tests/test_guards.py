"""Guard tests: temperature validation and the chunking arithmetic. No model."""

import pytest
from conftest import make_engine

from jevmlx.engine import _rows_per_chunk, run_parallel_generation
from jevmlx.schema import StructuredSchema

SCHEMA = {"action": {"type": "enum", "choices": ["A", "B"], "description": "d"}}


class TestTemperatureValidation:
    """Y4: temperature must be a finite number > 0, checked before model use."""

    @pytest.mark.parametrize(
        "temperature", [0, -1, -0.5, float("nan"), float("inf"), float("-inf")]
    )
    def test_invalid_temperature_raises_before_model_use(self, temperature):
        # model/tokenizer are None on purpose: the guard must fire first.
        with pytest.raises(ValueError, match="temperature must be a finite number > 0"):
            run_parallel_generation(
                make_engine(None, None), "x", StructuredSchema(SCHEMA), temperature=temperature
            )

    def test_max_rows_zero_raises_before_model_use(self):
        with pytest.raises(ValueError, match="max_rows must be >= 1"):
            run_parallel_generation(
                make_engine(None, None), "x", StructuredSchema(SCHEMA), max_rows=0
            )


class TestRowsPerChunk:
    """Y5: the pure chunking arithmetic."""

    def test_auto_cap_from_budget(self):
        assert _rows_per_chunk(budget_bytes=10_000, bytes_per_row=300, max_rows=None) == 33

    def test_max_rows_caps_below_auto(self):
        assert _rows_per_chunk(budget_bytes=10_000, bytes_per_row=300, max_rows=10) == 10

    def test_max_rows_above_auto_does_not_lift_the_cap(self):
        # A large caller value must not defeat the budget: min(auto_cap, max_rows).
        assert _rows_per_chunk(budget_bytes=10_000, bytes_per_row=300, max_rows=1_000) == 33

    def test_budget_smaller_than_one_row_still_runs_one_row(self):
        assert _rows_per_chunk(budget_bytes=10, bytes_per_row=300, max_rows=None) == 1

    def test_zero_bytes_per_row_falls_back_to_one(self):
        assert _rows_per_chunk(budget_bytes=10_000, bytes_per_row=0, max_rows=None) == 1

    @pytest.mark.parametrize("max_rows", [0, -1])
    def test_max_rows_below_one_raises(self, max_rows):
        with pytest.raises(ValueError, match="max_rows must be >= 1"):
            _rows_per_chunk(budget_bytes=10_000, bytes_per_row=300, max_rows=max_rows)
