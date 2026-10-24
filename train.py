"""Train a model that predicts whether the NEXT 7 days will be 'high volatility'."""

from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from features import FEATURE_COLS, HORIZON

MODEL_PATH = Path("models/volatility_model.joblib")


def split_with_gap(df, train_fraction=0.75, gap=HORIZON):

    split = int(len(df) * train_fraction)
    return df.iloc[:split], df.iloc[split + gap:]


def make_model():

    return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=42)


def train_volatility_model(featured):
    
    data = featured.dropna(subset=FEATURE_COLS + ["future_vol"]).reset_index(drop=True)
    train, test = split_with_gap(data)

    threshold = train["future_vol"].quantile(0.75)
    y_train = (train["future_vol"] > threshold).astype(int)
    y_test = (test["future_vol"] > threshold).astype(int)

    baseline_auc = roc_auc_score(y_test, test["volatility_7d"])

    model = make_model()
    model.fit(train[FEATURE_COLS], y_train)
    probabilities = model.predict_proba(test[FEATURE_COLS])[:, 1]
    model_auc = roc_auc_score(y_test, probabilities)

    print(f"Train days: {len(train)}, test days: {len(test)}")
    print(f"Share of high-volatility days in the test period: {y_test.mean():.1%}")
    print(f"ROC-AUC baseline : {baseline_auc:.3f}")
    print(f"ROC-AUC random forest: {model_auc:.3f}")
    print("Feature importance :")
    for name, importance in zip(FEATURE_COLS, model.feature_importances_):
        print(f"{name:18s} {importance:.3f}")

    final_threshold = data["future_vol"].quantile(0.75)
    final_model = make_model()
    final_model.fit(data[FEATURE_COLS], (data["future_vol"] > final_threshold).astype(int))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": final_model, "threshold": float(final_threshold), "horizon": HORIZON}
    joblib.dump(result, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return result
