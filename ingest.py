"""
Data source: Binance public market-data API.
"""
import time
from pathlib import Path

import pandas as pd
import requests

KLINES_URL = "https://api.binance.com/api/v3/klines"
DAY_MS = 24 * 60 * 60 * 1000  
MAX_PER_CALL = 1000         


def get_with_retries(params: dict, retries: int = 3, pause: float = 2.0) -> list:
    """Call the API
       If it fails (network problem, rate limit), wait and try again"""
    
    headers = {"User-Agent": "BTCVolatilityPipeline/1.0 (educational project)"}
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(KLINES_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()        
            return response.json()
        except requests.RequestException as exc:
            if attempt == retries:
                raise RuntimeError(f"Binance request failed after {retries} attempts: {exc}") from exc
            time.sleep(pause * attempt)         


def parse_klines(rows: list, symbol: str) -> pd.DataFrame:
    """Turns the raw API answer into a clean table"""

    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=columns)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")      #milliseconds since 1970 to date
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)                         
    df["symbol"] = symbol
    df = df[["date", "symbol", "open", "high", "low", "close", "volume", "quote_volume"]]

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    df = df[df["date"] < today]
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_daily_candles(symbol: str = "BTCUSDT", start: str = "2020-01-01") -> pd.DataFrame:
    """Download all daily candles from the defined start until yesterday"""

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows = []
    print(f"Downloading daily candles for {symbol} since {start} ...")
    while True:
        batch = get_with_retries({"symbol": symbol, "interval": "1d", "startTime": start_ms, "limit": MAX_PER_CALL})
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + DAY_MS         
        if len(batch) < MAX_PER_CALL:    #we reached the present     
            break
        time.sleep(0.3)                         
    df = parse_klines(rows, symbol)
    print(f"Downloaded {len(df)} daily records ({df['date'].min().date()} -> {df['date'].max().date()}).")
    return df


def save_raw(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved raw data to {path}")


if __name__ == "__main__":
    data = fetch_daily_candles()
    save_raw(data, Path("data/raw_btcusdt.csv"))
    print(data.head())
