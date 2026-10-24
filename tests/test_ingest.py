import pandas as pd
from ingest import DAY_MS, parse_klines

def fake_row(open_time_ms: int, price: float):
    return [open_time_ms, str(price), str(price), str(price), str(price), "10.0",
            open_time_ms + DAY_MS - 1, "1000.0", 5, "1", "1", "0"]


def test_parse_klines_converts_types_and_drops_today():
    today_ms = int(pd.Timestamp.now(tz="UTC").normalize().timestamp() * 1000)
    rows = [fake_row(today_ms - 2 * DAY_MS, 100.0),
            fake_row(today_ms - DAY_MS, 101.0),
            fake_row(today_ms, 102.0)]               #for today's candle that is not finished
    df = parse_klines(rows, "BTCUSDT")
    assert len(df) == 2
    assert df["close"].dtype == float
    assert df["date"].is_monotonic_increasing
