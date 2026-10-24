import numpy as np
import pandas as pd

from features import FEATURE_COLS, HORIZON, build_features
from train import split_with_gap


def make_prices(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "close": close,
        "volume": rng.uniform(1e3, 5e3, n),
    })


def test_features_never_use_the_future():
    base = make_prices()
    changed = base.copy()
    changed.loc[61:, "close"] *= np.random.default_rng(1).uniform(0.5, 1.5, len(changed) - 61)

    a, b = build_features(base), build_features(changed)
    pd.testing.assert_frame_equal(a.loc[:60, FEATURE_COLS], b.loc[:60, FEATURE_COLS])
    # ...while the future volatility of day 60 SHOULD change (it is made of days 61-67)
    assert a.loc[60, "future_vol"] != b.loc[60, "future_vol"]


def test_future_vol_is_std_of_the_next_days_only():
    df = build_features(make_prices())
    t = 50
    expected = df["return_1d"].iloc[t + 1 : t + 1 + HORIZON].std()
    assert np.isclose(df.loc[t, "future_vol"], expected)


def test_last_rows_have_no_future_volatility():
    df = build_features(make_prices())
    assert df["future_vol"].tail(HORIZON).isna().all()
    assert df["future_vol"].iloc[-HORIZON - 1:-HORIZON].notna().all()


def test_split_leaves_a_gap_between_train_and_test():
    df = pd.DataFrame({"x": range(100)})
    train, test = split_with_gap(df, train_fraction=0.75, gap=HORIZON)
    assert test.index.min() - train.index.max() == HORIZON + 1
