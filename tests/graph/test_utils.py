import pytest

from src.graph._utils import userAccurate


@pytest.mark.parametrize('number,expected_format', [
    (1.0, '1'),
    (1, '1'),
    (486, '486'),
    (9000, '9k'),
    (10_000, '10k'),
    (100_000, '100k'),
    (1_000_000, '1M'),
    (17_000_000, '17M'),
    (1_000_000_000, '1G'),
    (1_000_000_000_000, '1T'),
    # floats
    (1261, '1.261k'),
    (1_700_000, '1.7M'),
    (2_900_000_000, '2.9G'),
    # too long
    (12613, '12,613'),
    (1261358, '1,261,358'),
    # much over trillion
    (1_000_000_000_000_000, '1,000T'),
    (1_000_000_000_000_000_000, '1,000,000T'),
])
def test_user_accurate(number, expected_format):
    assert userAccurate(number) == expected_format
