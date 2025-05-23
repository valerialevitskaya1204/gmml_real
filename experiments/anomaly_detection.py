import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import umap.umap_ as umap
import yaml
import sys


with open(r"/Users/eneminova/Desktop/gmml_real/conifgs/anomaly_config.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)


def cut_dataset(df, desired_sample_count=250000):
    class_proportions = df['Class'].value_counts(normalize=True)
    sample_counts = (class_proportions * desired_sample_count).round().astype(int)
    
    sampled_data = pd.concat([
        df[df['Class'] == cls].sample(n=count, random_state=42)
        for cls, count in sample_counts.items()
    ])
    
    final_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return final_data


class FraudDetection:
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
        self.df["scaled_amount"] = self.scaler.fit_transform(
            self.df["Amount"].values.reshape(-1, 1)
        )
        self.df["scaled_time"] = self.scaler.fit_transform(
            self.df["Time"].values.reshape(-1, 1)
        )
        self.df = self.df.drop(["Time", "Amount"], axis=1)

    def split_data(self):
        X = self.df.drop("Class", axis=1).values
        y = self.df["Class"].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(
            f"Data split into training and test sets: {self.X_train.shape}, {self.X_test.shape}"
        )

    def apply_dimensionality_reduction(self, method="PCA"):
        if method == "PCA":
            self.reduction_model = PCA(n_components=2)
        elif method == "Isomap":
            self.reduction_model = Isomap(n_neighbors=30, n_components=2, n_jobs=-1)
        elif method == "UMAP":
            self.reduction_model = umap.UMAP(n_components=2)
        else:
            raise ValueError("Method should be 'PCA', 'Isomap', or 'UMAP'")
        self.X_train_reduced = self.reduction_model.fit_transform(self.X_train)
        self.X_test_reduced = self.reduction_model.transform(self.X_test)
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
         # Create 3D plot for the reduced data
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Apply color map for better distinguishability
        scatter = ax.scatter(
            self.X_train_reduced[:, 0],
            self.X_train_reduced[:, 1],
            self.X_train_reduced[:, 2],
            c=self.y_train,
            cmap="coolwarm",  # Better colormap for distinction
            marker='o',
            s=50, 
            alpha=0.5# Larger dot size for better visibility
        )
        
        # Apply custom scaling to the plot
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        
        # Adding color bar to indicate the class (normal vs anomaly)
        fig.colorbar(scatter)

        # Rotate the plot by setting the azim and elev values
        ax.view_init(elev=30, azim=45)  # You can modify these angles for a different view
        
        plt.show()


class AnomalyDetection:
    def __init__(self, data_path):
        self.fraud_detection = FraudDetection(data_path)

    def run(self):
        self.fraud_detection.load_data()
        self.fraud_detection.preprocess_data()
        self.fraud_detection.split_data()

        for method in ["PCA", "UMAP"]:
            print(f"\nEvaluating with {method}...")
            self.fraud_detection.apply_dimensionality_reduction(method)
            self.fraud_detection.train_knn_classifier()
            report, confusion = self.fraud_detection.evaluate_classifier()
            print(report)
            print(confusion)
            self.fraud_detection.plot_embedding(title=f"{method} Embedding")


if __name__ == "__main__":
    data_path = params_list["DATA_PATH"][0]
    anomaly_detection = AnomalyDetection(data_path)
    anomaly_detection.run()
