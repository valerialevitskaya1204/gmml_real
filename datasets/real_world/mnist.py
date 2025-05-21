import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state

class MNIST:
    def __init__(self, task):
        self.task = task
        self.X = None
        self.y = None
        self.X_original = None
        self.y_original = None
        self.X_noisy = None
        self.y_noisy = None
        self.X_missing = None
        self.y_missing = None
        self.noise_scale = None
        self.missing_frac = None
        self.missing_mask = None 

    def create_mnist(self, n_samples=1000, noise=0.0, threshold=0.9):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)

        rng = check_random_state(42)
        indices = rng.choice(X.shape[0], n_samples, replace=False)
        X = X[indices].astype(np.float64)
        y = y[indices]

        self.X_original = X
        self.y_original = y

        if self.task == 'damage':
            self.noise_scale = noise
            X_noisy = X + np.random.normal(scale=noise, size=X.shape)
            X_noisy = np.clip(X_noisy, 0, 255)
            self.X = X_noisy
            self.y = y
            self.X_noisy = X_noisy

        elif self.task == 'miss':
            self.missing_frac = threshold
            missing_mask = np.random.rand(X.shape[0]) < threshold
            self.missing_mask = missing_mask
            X_missing = X[missing_mask]
            y_missing = y[missing_mask]
            self.X = X_missing
            self.y = y_missing
            self.X_missing = X_missing
            self.y_missing = y_missing

        elif self.task == 'original':
            self.X = X
            self.y = y

    def _plot_images(self, ax, images, title):
            ax.clear()
            ax.set_title(title)
            ax.axis('off')
            if len(images) == 0:
                return
            for i in range(min(5, len(images))):
                img = images[i].reshape(28, 28)
                sub_ax = ax.inset_axes([i * 0.2, 0.1, 0.18, 0.8])
                sub_ax.imshow(img, cmap='gray')
                sub_ax.axis('off')

    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        self._plot_images(axes[0], self.X_original, 'Original MNIST')
        
        # Noisy
        if self.task == 'damage':
            self._plot_images(axes[1], self.X_noisy, f'Noisy MNIST (σ={self.noise_scale})')
        else:
            axes[1].axis('off')
            axes[1].set_title('Not Applicable')
        
        # Missing
        if self.task == 'miss':
            self._plot_images(axes[2], self.X_missing, f'MNIST with {int((1 - self.missing_frac)*100)}% Missing')
        else:
            axes[2].axis('off')
            axes[2].set_title('Not Applicable')
        
        plt.tight_layout()
        plt.show()