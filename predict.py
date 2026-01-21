import argparse
import joblib
import pandas as pd

def predict(model_path, historic_data_path, future_data_path, out_file_path):
    model = joblib.load(model_path)
    future_df = pd.read_csv(future_data_path)
    # Match the features used in training
    features = future_df[["rainfall"]].fillna(0)

    predictions = model.predict(features)
    output_df = future_df[["time_period", "location"]].copy()
    output_df["sample_0"] = predictions
    output_df.to_csv(out_file_path, index=False)
    print(f"Predictions saved to {out_file_path}")

if __name__ == "__main__":
    import sys
    predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
