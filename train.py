def train(csv_fn, model_fn):
    df = pd.read_csv(csv_fn)
    features = ['rainfall', 'mean_temperature']
    X = df[features]
    Y = df['disease_cases']
    # Replace missing values with mean
    Y = Y.fillna(Y.mean())

    model = LinearRegression()
    model.fit(X, Y)
    joblib.dump(model, model_fn)
    print(f"Model saved to {model_fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a disease prediction model")
    parser.add_argument("train_data", help="Path to training data CSV file")
    parser.add_argument("model", help="Path to save the trained model")
    args = parser.parse_args()
    train(args.train_data, args.model)
