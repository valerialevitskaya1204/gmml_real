import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def cut_dataset(df, desired_sample_count=250000):
    class_proportions = df['Class'].value_counts(normalize=True)
    sample_counts = (class_proportions * desired_sample_count).round().astype(int)

    sampled_data = pd.concat([
        df[df['Class'] == cls].sample(n=count, random_state=42)
        for cls, count in sample_counts.items()
    ])

    final_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return final_data


class KDDAnomalyDetection:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df = cut_dataset(self.df)
        print(f"Dataset loaded with shape: {self.df.shape}")

    def preprocess_data(self):
        X = self.df.drop(columns=["Class"])
        categorical_columns = X.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_columns:
            X[col] = le.fit_transform(X[col])
        y = self.df["Class"]
        # y = y.apply(lambda x: 1 if x == "normal." else 0)

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def split_data(self):
        X, y = self.preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"Data split into training and test sets: {self.X_train.shape}, {self.X_test.shape}")

    def train_knn_classifier(self):
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.knn_classifier.fit(self.X_train, self.y_train)

    def evaluate_classifier(self):
        y_pred = self.knn_classifier.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        return report, confusion


class AnomalyDetection:
    def __init__(self, data_path):
        self.kdd_anomaly_detection = KDDAnomalyDetection(data_path)

    def run(self):
        self.kdd_anomaly_detection.load_data()
        self.kdd_anomaly_detection.split_data()
        self.kdd_anomaly_detection.train_knn_classifier()
        report, confusion = self.kdd_anomaly_detection.evaluate_classifier()
        print("\nKNN Classification Report (No Dimensionality Reduction):")
        print(report)
        print("Confusion Matrix:")
        print(confusion)


if __name__ == "__main__":
    data_path = "/Users/eneminova/Desktop/gmml_real/datasets/anomalies/creditcard.csv"
    anomaly_detection = AnomalyDetection(data_path)
    anomaly_detection.run()
