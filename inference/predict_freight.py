import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice flagging model.
    """

    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Load saved StandardScaler.
    """

    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)

    return scaler


def predict_invoice_flag(input_data):
    """
    Predict whether new invoices should be flagged.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with invoice flag prediction.
    """

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    input_scaled = scaler.transform(input_df)

    input_df["Predicted_Flag"] = model.predict(input_scaled)

    return input_df


if __name__ == "__main__":

    # Example inference run (local testing)

    sample_data = {
        "invoice_quantity": [120, 500],
        "invoice_dollars": [2500, 18000],
        "Freight": [60, 450],
        "total_item_quantity": [118, 490],
        "total_item_dollars": [2495, 17500],
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)