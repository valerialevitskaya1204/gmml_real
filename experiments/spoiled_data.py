from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, TSNE, trustworthiness
from sklearn.metrics import mean_squared_error
import umap.umap_ as umap
from datasets.synthetic.swiss_roll import Swiss
from datasets.real_world.mnist import MNIST


if __name__ == "__main__":
    # Swiss Roll datasets
    original_data = Swiss('original')
    original_data.create_swiss(n_samples=1000)
    noisy_data = Swiss('damage')
    noisy_data.create_swiss(n_samples=1000, noise=0.5)
    missing_data = Swiss('miss')
    missing_data.create_swiss(n_samples=1000, threshold=0.1)
    
    swiss_datasets = {
        'original': original_data,
        'noisy': noisy_data,
        'missing': missing_data
    }

    # MNIST datasets
    mnist_original = MNIST('original')
    mnist_original.create_mnist(n_samples=1000)
    mnist_noisy = MNIST('damage')
    mnist_noisy.create_mnist(n_samples=1000, noise=25.5)
    mnist_missing = MNIST('miss')
    mnist_missing.create_mnist(n_samples=1000, threshold=0.1)
    
    mnist_datasets = {
        'original': mnist_original,
        'noisy': mnist_noisy,
        'missing': mnist_missing
    }

    methods = {
        'PCA': PCA(n_components=2, random_state=42),
        'MDS': MDS(n_components=2, random_state=42, normalized_stress='auto'),
        'Isomap': Isomap(n_components=2),
        't-SNE': TSNE(n_components=2, random_state=42),
        'UMAP': umap.UMAP(random_state=42)
    }

    # Evaluate Swiss Roll
    print("Swiss Roll Results:")
    swiss_results = {}
    for data_name, data in swiss_datasets.items():
        X = data.X
        swiss_results[data_name] = {}
        for method_name, method in methods.items():
            embedding = method.fit_transform(X)
            tw = trustworthiness(X, embedding, n_neighbors=10)
            swiss_results[data_name][method_name] = {'trustworthiness': tw}
            if method_name == 'PCA':
                reconstructed = method.inverse_transform(embedding)
                if data_name == 'original':
                    mse = mean_squared_error(data.X_original, reconstructed)
                elif data_name == 'noisy':
                    mse = mean_squared_error(data.X_original, reconstructed)
                elif data_name == 'missing':
                    mse = mean_squared_error(data.X_original[data.missing_mask], reconstructed)
                swiss_results[data_name][method_name]['reconstruction_error'] = mse

    # Print Swiss results
    for data_name in swiss_datasets:
        print(f"\n--- {data_name.upper()} ---")
        for method in methods:
            res = swiss_results[data_name][method]
            output = f"{method}: Trustworthiness = {res['trustworthiness']:.3f}"
            if 'reconstruction_error' in res:
                output += f", Reconstruction Error = {res['reconstruction_error']:.3f}"
            print(output)

    # Evaluate MNIST
    print("\nMNIST Results:")
    mnist_results = {}
    for data_name, data in mnist_datasets.items():
        X = data.X
        mnist_results[data_name] = {}
        for method_name, method in methods.items():
            embedding = method.fit_transform(X)
            tw = trustworthiness(X, embedding, n_neighbors=10)
            mnist_results[data_name][method_name] = {'trustworthiness': tw}
            if method_name == 'PCA':
                reconstructed = method.inverse_transform(embedding)
                if data_name == 'original':
                    mse = mean_squared_error(data.X_original, reconstructed)
                elif data_name == 'noisy':
                    mse = mean_squared_error(data.X_original, reconstructed)
                elif data_name == 'missing':
                    mse = mean_squared_error(data.X_original[data.missing_mask], reconstructed)
                mnist_results[data_name][method_name]['reconstruction_error'] = mse
    #print results
    for data_name in mnist_datasets:
        print(f"\n--- {data_name.upper()} ---")
        for method in methods:
            res = mnist_results[data_name][method]
            output = f"{method}: Trustworthiness = {res['trustworthiness']:.3f}"
            if 'reconstruction_error' in res:
                output += f", Reconstruction Error = {res['reconstruction_error']:.3f}"
            print(output)


    original_data.plot()
    noisy_data.plot()
    missing_data.plot()
    mnist_original.plot()
    mnist_noisy.plot()
    mnist_missing.plot()