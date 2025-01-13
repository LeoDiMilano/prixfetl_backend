from xgboost import XGBClassifier
import joblib

class ModelTraining:
    def __init__(self):
        self.model = XGBClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def save_model(self, path_to_file: str):
        joblib.dump(self.model, path_to_file)
