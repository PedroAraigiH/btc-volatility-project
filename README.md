# Bitcoin volatility forecasting pipeline

In this project I built a machine learning pipeline that downloads daily Bitcoin prices, builds features,
and predicts whether the next 7 days will be unusually volatile (top 25 % of the
volatility seen in the training period). It is framed as a risk management problem, to help
know in advance if the markets will be turbulent, which matters for risk limits and position sizing.
It does not try to predict price direction or trading profits.

## Quick start
1. `pip install -r requirements.txt`
2. `python main.py --refresh`      
3. `pytest`                         


## Pipeline
| Step | File | What it does |
|---|---|---|
| 1 | `ingest.py` | Downloads daily candles from the Binance public API (retries, finished days only) |
| 2 | `features.py` | Returns, 7/30-day volatility, volume change; 'future_vol' = volatility of the next 7 days |
| 3 | `train.py` | Chronological split with a gap, train-only threshold, baseline vs random forest (ROC-AUC) |
| 4 | `main.py` | Runs everything and prints the probability for the next 7 days |

## Avoiding data leakage
- Features on day *t* use only data up to day *t* (checked by unit tests).
- The label uses the returns of days *t+1 ... t+7* only, with no overlap with the features.
- The "high volatility" threshold is computed on the training period only.
- A 7-day gap separates train and test, because the last training labels look 7 days ahead.
- The last 7 days have no label (their future is unknown) and are excluded from training.

## Results
Data: Binance BTCUSDT daily candles, from 2020-01-01 to 2026-10-09 (2,474 days in total).
Split: first 1,827 days are for training, last 603 days are for testing (after the 7-day gap).
Only 5.3 % of the test days are "high volatility" (about 30 days).

| Model | ROC-AUC (hold out period) |
|---|---|
| Baseline: persistence (today's 7-day volatility) | 0.840 |
| Random forest | 0.703 |

Random forest feature importance are : 'volatility_30d' 0.416, 'volatility_7d' 0.361,
'abs_return_1d' 0.079, 'return_1d' 0.076, 'volume_change_1d' 0.068.

### Interpretation
- Volatility is highly persistent: today's 7-day volatility alone separates future wild weeks from calm weeks well (ROC-AUC 0.84).
- The random forest did not beat this simple baseline (0.70). The baseline was worth building.
- Possible explanations (not tested yet):
  1. Regime change: the threshold comes from the training period (2020 to early 2025, which includes very turbulent years), and the test period was calmer, so it contains few positive cases.
  2. Coarse scores: shallow trees output step-like probabilities with many ties, while the raw volatility value ranks days more finely.
  3. Little extra information: all features come from the same price series, so they add little beyond volatility itself.

## Limitations
- The test period contains only about 30 high volatility days, and consecutive days overlap, so the number of independent episodes is smaller.
- A single train/test split, one asset, price and volume features only.

## Next steps
- Walk-forward evaluation (several successive test periods) with 'TimeSeriesSplit'.
- A threshold relative to recent history, so labels are comparable across calm and turbulent periods.
- Compare with classical volatility models (GARCH, HAR) and a simple logistic regression.
- Test on other assets (ETH, equity indices).