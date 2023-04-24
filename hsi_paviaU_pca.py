import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from sklearn.decomposition import PCA


def read_HSI_PaviaU():
    x = loadmat('data/raw/pavia_u/PaviaU.mat')['paviaU']
    y = loadmat('data/raw/pavia_u/PaviaU_gt.mat')['paviaU_gt']
    print(f"X shape: {x.shape}\ny shape: {y.shape}")
    return x, y


def extract_pixels(x, y):
    q = x.reshape(-1, x.shape[2])
    df = pd.DataFrame(data=q)
    df = pd.concat([df, pd.DataFrame(data=y.ravel())], axis=1)
    df.columns = [f'band{i}' for i in range(1, 1 + x.shape[2])] + ['class']
    # df.to_csv('../data/processed/dataset.csv')
    return df


if __name__ == '__main__':
    # Read mat files:
    x, y = read_HSI_PaviaU()

    # Visualize:
    fig = plt.figure()
    for i in range(1, 1 + 6):
        fig.add_subplot(2, 3, i)
        q = np.random.randint(x.shape[2])
        plt.imshow(x[:, :, q], cmap='jet')
        plt.axis('off')
        plt.title(f'band - {q}')
    plt.show()

    # Visualize ground truth:
    plt.imshow(y, cmap='jet')
    plt.title('Ground truth')
    plt.show()

    # Convert to dataframe:
    df = extract_pixels(x, y)

    # Dimension reduction:
    pca = PCA(n_components=3)
    dt = pca.fit_transform(df.iloc[:, :-1].values)
    print(dt)
    q = pd.concat([pd.DataFrame(data=dt), pd.DataFrame(data=y.ravel())], axis=1)
    q.columns = [f'PC-{i}' for i in range(1, 4)] + ['class']
