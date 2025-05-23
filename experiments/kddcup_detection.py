import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import umap.umap_ as umap


def cut_dataset(df, desired_sample_count=10000):
    class_proportions = df['label'].value_counts(normalize=True)
    sample_counts = (class_proportions * desired_sample_count).round().astype(int)

    sampled_data = pd.concat([
        df[df['label'] == cls].sample(n=count, random_state=42)
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
        X = self.df.drop(columns=["label"])
        categorical_columns = X.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_columns:
            X[col] = le.fit_transform(X[col])
        y = self.df["label"]
        y = y.apply(lambda x: 1 if x == "normal." else 0)

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def split_data(self):
        X, y = self.preprocess_data()
        print(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"Data split into training and test sets: {self.X_train.shape}, {self.X_test.shape}")

    def apply_dimensionality_reduction(self, method="PCA"):
        if method == "PCA":
            self.reduction_model = PCA(n_components=2)
            self.X_train_reduced = self.reduction_model.fit_transform(self.X_train)
            self.X_test_reduced = self.reduction_model.transform(self.X_test)
        elif method == "TSNE":
            tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
            self.X_train_reduced = tsne.fit_transform(self.X_train)
            self.X_test_reduced = tsne.fit_transform(self.X_test)  # Not ideal for test inference
        elif method == "UMAP":
            self.reduction_model = umap.UMAP(n_components=2)
            self.X_train_reduced = self.reduction_model.fit_transform(self.X_train)
            self.X_test_reduced = self.reduction_model.transform(self.X_test)
        elif method == "ISOMAP":
            self.reduction_model = Isomap(n_components=2)
            self.X_train_reduced = self.reduction_model.fit_transform(self.X_train)
            self.X_test_reduced = self.reduction_model.transform(self.X_test)
        else:
            raise ValueError("Method should be 'PCA', 'TSNE', 'UMAP', or 'ISOMAP'")
        print(f"{method} dimensionality reduction applied.")

    def train_knn_classifier(self):
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.knn_classifier.fit(self.X_train_reduced, self.y_train)

    def evaluate_classifier(self):
        y_pred = self.knn_classifier.predict(self.X_test_reduced)
        report = classification_report(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        return report, confusion

    def plot_embedding(self, title="Embedding"):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            self.X_train_reduced[:, 0],
            self.X_train_reduced[:, 1],
            c=self.y_train,
            cmap="coolwarm",
        )
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()

    def plot_embedding_3d(self, title="Embedding"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            self.X_train_reduced[:, 0],
            self.X_train_reduced[:, 1],
            self.X_train_reduced[:, 2],
            c=self.y_train,
            cmap="coolwarm",
            marker='o',
            s=50
        )
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        fig.colorbar(scatter)
        ax.view_init(elev=30, azim=45)
        plt.show()


class AnomalyDetection:
    def __init__(self, data_path):
        self.kdd_anomaly_detection = KDDAnomalyDetection(data_path)

    def run(self):
        self.kdd_anomaly_detection.load_data()
        self.kdd_anomaly_detection.split_data()

        for method in ["TSNE"]:
            print(f"\nEvaluating with {method}...")
            self.kdd_anomaly_detection.apply_dimensionality_reduction(method)
            self.kdd_anomaly_detection.train_knn_classifier()
            report, confusion = self.kdd_anomaly_detection.evaluate_classifier()
            print(report)
            print(confusion)
            if method == "TSNE":
                self.kdd_anomaly_detection.plot_embedding_3d(title=f"{method} Embedding")
            else:
                self.kdd_anomaly_detection.plot_embedding(title=f"{method} Embedding")


if __name__ == "__main__":
    data_path = "/Users/eneminova/Desktop/gmml_real/datasets/anomalies/kddcup.csv"
    anomaly_detection = AnomalyDetection(data_path)
    anomaly_detection.run()
