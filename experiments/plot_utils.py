import matplotlib.pyplot as plt
import numpy as np

def plot_swiss_results(X, embeddings, title):
    """Plot Swiss Roll embeddings from different methods"""
    plt.figure(figsize=(15, 10))
    
    # Plot original data
    ax = plt.subplot(231, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap=plt.cm.Spectral, s=10)
    plt.title('Original Swiss Roll')
    
    # Plot each method's embedding
    for i, (method_name, embedding) in enumerate(embeddings.items(), 1):
        ax = plt.subplot(2, 3, i+1)
        # Use the same coloring as original (first 1000 points if needed)
        colors = X[:len(embedding), 2] if len(embedding) < len(X) else X[:, 2]
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=plt.cm.Spectral, s=10)
        plt.title(f'{method_name} Embedding')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_mnist_results(X, embeddings, title, n_samples=1000):
    """Plot MNIST embeddings from different methods"""
    plt.figure(figsize=(15, 10))
    
    # Plot original data (first two dimensions)
    plt.subplot(231)
    plt.scatter(X[:n_samples, 0], X[:n_samples, 1], s=5)
    plt.title('Original MNIST (first 2 dims)')
    
    # Plot each method's embedding
    for i, (method_name, embedding) in enumerate(embeddings.items(), 1):
        plt.subplot(2, 3, i+1)
        plt.scatter(embedding[:n_samples, 0], embedding[:n_samples, 1], s=5)
        plt.title(f'{method_name} Embedding')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_reconstructions(original, reconstructed, title, is_mnist=False):
    """Plot PCA reconstructions"""
    if is_mnist:
        # For MNIST, show sample digits
        n_samples = min(5, reconstructed.shape[0])  # Ensure we don't exceed available samples
        plt.figure(figsize=(12, 5))
        plt.suptitle(title)
        
        # Use indices that exist in both original and reconstructed
        valid_indices = np.arange(min(original.shape[0], reconstructed.shape[0]))
        indices = np.random.choice(valid_indices, n_samples, replace=False)
        
        # Plot original digits
        plt.subplot(2, n_samples, 1)
        plt.title('Original Digits', y=1.1)
        for i, idx in enumerate(indices):
            plt.subplot(2, n_samples, i+1)
            plt.imshow(original[idx].reshape(28, 28), cmap='gray')
            plt.axis('off')
        
        # Plot reconstructed digits
        plt.subplot(2, n_samples, n_samples+1)
        plt.title('PCA Reconstructed', y=1.1)
        for i, idx in enumerate(indices):
            plt.subplot(2, n_samples, n_samples+i+1)
            plt.imshow(reconstructed[idx].reshape(28, 28), cmap='gray')
            plt.axis('off')
    else:
        # For Swiss Roll, show 3D plots
        fig = plt.figure(figsize=(12, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(original[:, 0], original[:, 1], original[:, 2], 
                   c=original[:, 2], cmap='nipy_spectral', s=10)
        ax1.set_title('Original Data')
        ax1.view_init(elev=20, azim=70)
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], 
                   c=reconstructed[:, 2], cmap='nipy_spectral', s=10)
        ax2.set_title('PCA Reconstruction')
        ax2.view_init(elev=20, azim=70)
    
    plt.tight_layout()
    plt.show()