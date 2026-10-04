from pathlib import Path
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def main():
    df = pd.read_csv(DATA / "train_features.csv")
    split = pd.read_csv(REPORTS / "train_validation_split.csv")

    train_ids = split.loc[split["split"] == "train", "unit_id"]
    val_ids = split.loc[split["split"] == "validation", "unit_id"]

    train_df = df[df.unit_id.isin(train_ids)]
    val_df = df[df.unit_id.isin(val_ids)]

    features = [c for c in df.columns if c not in ["unit_id", "cycle", "RUL"]]

    X_train = train_df[features]
    y_train = train_df["RUL"].clip(upper=200)

    X_val = val_df[features]
    y_val = val_df["RUL"]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    result = pd.DataFrame({
        "unit_id": val_df["unit_id"].values,
        "cycle": val_df["cycle"].values,
        "RUL": y_val.values,
        "predicted_RUL": predictions
    })

    result["error"] = result["predicted_RUL"] - result["RUL"]
    result["absolute_error"] = result["error"].abs()

    MODELS.mkdir(exist_ok=True)

    joblib.dump(
        {"imputer": imputer, "model": model, "features": features},
        MODELS / "xgboost_capped_rul_200.joblib"
    )

    result.to_csv(
        REPORTS / "xgboost_capped_rul_predictions.csv",
        index=False
    )

    print("Model trained successfully.")
    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Features:", len(features))
    print("RUL cap: 200")
    print("Saved model: models/xgboost_capped_rul_200.joblib")
    print("Saved predictions: reports/xgboost_capped_rul_predictions.csv")


if __name__ == "__main__":
    main()