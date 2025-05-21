import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

class Swiss:
    def __init__(self, task):
        self.task = task
        self.X = None
        self.t = None
        self.X_original = None
        self.t_original = None
        self.X_noisy = None
        self.t_noisy = None
        self.X_missing = None
        self.t_missing = None
        self.noise_scale = None
        self.missing_frac = None
        self.missing_mask = None 

    def create_swiss(self, n_samples=1000, noise=0.0, threshold=0.9):
        X, t = make_swiss_roll(n_samples=n_samples, noise=0.0, random_state=42)
        self.X_original = X
        self.t_original = t

        if self.task == 'damage':
            self.noise_scale = noise
            X_noisy = X + np.random.normal(scale=noise, size=X.shape)
            self.X = X_noisy
            self.t = t
            self.X_noisy = X_noisy

        elif self.task == 'miss':
            self.missing_frac = threshold
            missing_mask = np.random.rand(X.shape[0]) < threshold
            self.missing_mask = missing_mask
            X_missing, t_missing = X[missing_mask], t[missing_mask]
            self.X = X_missing
            self.t = t_missing
            self.X_missing = X_missing
            self.t_missing = t_missing

        elif self.task == 'original':
            self.X = X
            self.t = t

    def plot(self):
        fig = plt.figure(figsize=(18, 5))

        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(self.X_original[:, 0], self.X_original[:, 1], self.X_original[:, 2],
                    c=self.t_original, cmap=plt.cm.Spectral)
        ax1.set_title('Original Swiss Roll')

        ax2 = fig.add_subplot(132, projection='3d')
        if self.task == 'damage':
            ax2.scatter(self.X_noisy[:, 0], self.X_noisy[:, 1], self.X_noisy[:, 2],
                        c=self.t, cmap=plt.cm.Spectral)
            ax2.set_title(f'Swiss Roll with Noise (σ={self.noise_scale})')
        else:
            ax2.axis('off')
            ax2.set_title('Not Applicable')

        ax3 = fig.add_subplot(133, projection='3d')
        if self.task == 'miss':
            ax3.scatter(self.X_missing[:, 0], self.X_missing[:, 1], self.X_missing[:, 2],
                        c=self.t_missing, cmap=plt.cm.Spectral)
            ax3.set_title(f'Swiss Roll with {int((1 - self.missing_frac)*100)}% Missing')
        else:
            ax3.axis('off')
            ax3.set_title('Not Applicable')

        plt.tight_layout()
        plt.show()
