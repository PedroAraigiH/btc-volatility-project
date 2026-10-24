"""
Build the model inputs (or features) from daily prices.
"""
import numpy as np
import pandas as pd

HORIZON = 7   #we want to predict the volatility of the next 7 days

FEATURE_COLS = [
    "return_1d",         
    "abs_return_1d",    
    "volatility_7d",    
    "volatility_30d",   
    "volume_change_1d",  #(log scale)
] 


def build_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Add feature columns and 'future_vol' to the raw price table"""
    out = df.sort_values("date").reset_index(drop=True).copy()

    out["return_1d"] = out["close"].pct_change()
    out["abs_return_1d"] = out["return_1d"].abs()
    out["volatility_7d"] = out["return_1d"].rolling(7).std()
    out["volatility_30d"] = out["return_1d"].rolling(30).std()
    out["volume_change_1d"] = np.log(out["volume"]).diff()
    out["future_vol"] = out["return_1d"].rolling(horizon).std().shift(-horizon)

    return out.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    data = pd.read_csv("data/raw_btcusdt.csv", parse_dates=["date"])
    table = build_features(data)
    print(table[["date", "close"] + FEATURE_COLS + ["future_vol"]].tail(12))
