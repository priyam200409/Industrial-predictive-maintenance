import shap
import pandas as pd


def explain_prediction(model, X, top_n=5):
    imputer = model.named_steps["imputer"]
    regressor = model.named_steps["regressor"]

    X_transformed = imputer.transform(X)
    explainer = shap.TreeExplainer(regressor)
    values = explainer.shap_values(X_transformed)

    result = pd.DataFrame({
        "feature": X.columns,
        "shap_value": values[0]
    })

    result["impact"] = result["shap_value"].abs()
    return result.sort_values("impact", ascending=False).head(top_n)