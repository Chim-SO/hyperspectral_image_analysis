import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def read_HSI_IndianP():
    x = loadmat('data/raw/indian_pines/Indian_pines_corrected.mat')['indian_pines_corrected']
    y = loadmat('data/raw/indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
    print(f"x shape: {x.shape}\ny shape: {y.shape}")
    return x, y


def extract_pixels(x, y):
    q = x.reshape(-1, x.shape[2])
    df = pd.DataFrame(data=q)
    df = pd.concat([df, pd.DataFrame(data=y.ravel())], axis=1)
    df.columns = [f'band{i}' for i in range(1, 1 + x.shape[2])] + ['class']
    return df


if __name__ == '__main__':
    x, y = read_HSI_IndianP()

    # Display ground truth:
    plt.imshow(y, cmap='nipy_spectral')
    plt.axis('off')
    plt.title('Ground truth')
    plt.show()

    # Display random bands:
    fig = plt.figure()

    for i in range(1, 1 + 6):
        fig.add_subplot(2, 3, i)
        q = np.random.randint(x.shape[2])
        plt.imshow(x[:, :, q], cmap='nipy_spectral')
        plt.axis('off')
        plt.title(f'Band - {q}')
    plt.suptitle('Random bands visualization')
    plt.show()

    # Convert to dataframe:
    df = extract_pixels(x, y)

    # Dimensionality reduction:
    pca = PCA(n_components=40)
    dt = pca.fit_transform(df.iloc[:, :-1].values)
    q = pd.concat([pd.DataFrame(data=dt), pd.DataFrame(data=y.ravel())], axis=1)
    q.columns = [f'PC-{i}' for i in range(1, 41)] + ['class']

    # Visualization PCA:
    ev = pca.explained_variance_ratio_
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(ev))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    # Visualization PCA bands:
    fig = plt.figure()
    for i in range(1, 1 + 8):
        fig.add_subplot(2, 4, i)
        plt.imshow(q.loc[:, f'PC-{i}'].values.reshape(145, 145), cmap='nipy_spectral')
        plt.axis('off')
        plt.title(f'Band - {i}')
    plt.show()

    # SVM:
    x = q[q['class'] != 0]
    X = x.iloc[:, :-1].values
    y = x.loc[:, 'class'].values
    names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees',
             'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
             'Soybean-clean', 'Wheat', 'Woods', 'Buildings Grass Trees Drives', 'Stone Steel Towers']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11, stratify=y)
    svm = SVC(C=100, kernel='rbf', cache_size=10 * 1024)
    svm.fit(X_train, y_train)
    ypred = svm.predict(X_test)

    # Confusion matrix:
    data = confusion_matrix(y_test, ypred)
    df_cm = pd.DataFrame(data, columns=np.unique(names), index=np.unique(names))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(20, 18))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, cmap="Reds", annot=True, annot_kws={"size": 20}, fmt='d')
    plt.show()

    print(classification_report(y_test, ypred, target_names=names))

    l = []
    for i in range(q.shape[0]):
        if q.iloc[i, -1] == 0:
            l.append(0)
        else:
            l.append(svm.predict(q.iloc[i, :-1].values.reshape(1, -1)))

    clmap = np.array(l).reshape(145, 145).astype('float')
    plt.figure(figsize=(10, 8))
    plt.imshow(clmap, cmap='nipy_spectral')
    plt.colorbar()
    plt.axis('off')
    plt.show()
