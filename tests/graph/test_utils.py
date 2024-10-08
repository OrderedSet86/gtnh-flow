from typing import Union

import pytest

from src.graph._utils import userAccurate


@pytest.mark.parametrize('number,expected_format', [
    (1.0, '1'),
    (1, '1'),
    (486, '486'),
    (1261, '1,261'),
    (9000, '9k'),
    (10_000, '10k'),
    (100_000, '100k'),
    (1_000_000, '1M'),
    (17_000_000, '17M'),
    (1_000_000_000, '1G'),
    (1_000_000_000_000, '1T'),
    # to floats
    (1_700_000, '1.7M'),
    (2_900_000_000, '2.9G'),
    # from floats
    (0.013, '0.013'),
    (0.0132, '0.0132'),
    (0.01323, '0.0132'),
    (0.0001, '0.0001'),
    (0.00013, '0.00013'),
    (0.000132, '0.000132'),
    (0.0001323, '0.000132'),
    (3.552, '3.552'),
    (3.5552, '3.555'),
    (3.4266666666667, '3.427'),
    # too long
    (12613, '12,613'),
    (1261358, '1,261,358'),
    # much over trillion
    (1_000_000_000_000_000, '1,000T'),
    (1_000_000_000_000_000_000, '1,000,000T'),
    # negatives, for completeness
    (-1, '-1'),
    (-1024, '-1,024'),
    (-3.78, '-3.78'),
    (-0.0001, '-0.0001'),
    (-0.0001323, '-0.000132'),
])
def test_user_accurate(number: Union[float, int], expected_format: str) -> None:
    assert userAccurate(number) == expected_format
