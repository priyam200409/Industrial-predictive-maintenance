from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed"


def train_model(X, y):
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    return imputer, model


def main():
    df = pd.read_csv(DATA / "train_features.csv")
    split = pd.read_csv(REPORTS / "train_validation_split.csv")

    train_ids = split.loc[split["split"] == "train", "unit_id"]
    val_ids = split.loc[split["split"] == "validation", "unit_id"]

    train_df = df[df.unit_id.isin(train_ids)]
    val_df = df[df.unit_id.isin(val_ids)]

    features = [c for c in df.columns if c not in ["unit_id", "cycle", "RUL"]]

    X_train, y_train = train_df[features], train_df["RUL"]
    X_val, y_val = val_df[features], val_df["RUL"]

    results = []

    for cap in [None, 125, 150, 200]:
        target = y_train if cap is None else y_train.clip(upper=cap)

        imputer, model = train_model(X_train, target)
        pred = model.predict(imputer.transform(X_val))

        results.append({
            "RUL_cap": "uncapped" if cap is None else cap,
            "MAE": mean_absolute_error(y_val, pred),
            "RMSE": mean_squared_error(y_val, pred) ** 0.5,
            "mean_error": (pred - y_val).mean()
        })

    result = pd.DataFrame(results)
    result.to_csv(REPORTS / "day8_capped_rul_experiment.csv", index=False)

    print(result.to_string(index=False))
    print("\nBest:", result.loc[result.MAE.idxmin()].to_dict())


if __name__ == "__main__":
    main()