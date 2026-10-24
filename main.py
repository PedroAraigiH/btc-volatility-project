import argparse
from pathlib import Path
import pandas as pd
from features import FEATURE_COLS, build_features
from ingest import fetch_daily_candles, save_raw
from train import train_volatility_model


def run_pipeline(symbol: str = "BTCUSDT", start: str = "2020-01-01", refresh: bool = False):
    csv_path = Path("data") / f"raw_{symbol.lower()}.csv"

    if refresh or not csv_path.exists():
        raw = fetch_daily_candles(symbol, start)
        save_raw(raw, csv_path)
    else:
        print(f"Using cached data from {csv_path}")
        raw = pd.read_csv(csv_path, parse_dates=["date"])

    featured = build_features(raw)
    artifact = train_volatility_model(featured)

    latest = featured.dropna(subset=FEATURE_COLS).tail(1)
    probability = artifact["model"].predict_proba(latest[FEATURE_COLS])[0, 1]

    print(f"Last complete day: {latest['date'].iloc[0].date()}")
    print(f"Probability that the next {artifact['horizon']} days will be 'high volatility' "
          f"(daily std above {artifact['threshold']:.2%}): {probability:.1%}")
    return artifact, probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitcoin volatility pipeline")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--refresh", action="store_true", help="download fresh data")
    args = parser.parse_args()
    run_pipeline(args.symbol, args.start, args.refresh)