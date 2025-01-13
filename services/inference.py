import joblib

class ModelInference:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict_variation(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)
