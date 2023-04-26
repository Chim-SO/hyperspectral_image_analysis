import matplotlib.pyplot as plt

import numpy as np
import earthpy.plot as ep
import seaborn as sns
import earthpy.spatial as es

from scipy.io import loadmat

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from tqdm import tqdm

if __name__ == '__main__':
    # data:
    data = loadmat('data/raw/pavia_u/PaviaU.mat')['paviaU']
    gt = loadmat('data/raw/pavia_u/PaviaU_gt.mat')['paviaU_gt']
    df = pd.DataFrame(data.reshape(data.shape[0] * data.shape[1], -1))
    df.columns = [f'band{i}' for i in range(1, df.shape[-1] + 1)]
    df['class'] = gt.ravel()
    print(df.shape)

    # RGB image:
    mdata = np.moveaxis(data, -1, 0)
    ep.plot_rgb(mdata, (60, 30, 27), figsize=(12, 10))
    plt.show()

    # Ground truth:
    fig = plt.figure(figsize=(12, 10))
    plt.imshow(gt, cmap='nipy_spectral')
    plt.colorbar()
    plt.axis('off')
    plt.show()

    # Train data:
    X = df[df['class'] != 0].iloc[:, :-1].values
    y = tf.keras.utils.to_categorical(df[df['class'] != 0].iloc[:, -1].values,
                                      num_classes=np.unique(gt).shape[0],
                                      dtype='float32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)
    print(f"Train Data: {X_train.shape}\nTest Data: {X_test.shape}")

    # Model:
    model = Sequential(name='Pavia_University')

    model.add(Input(shape=X_train[0].shape, name='Input_Layer'))
    model.add(BatchNormalization())
    model.add(Dense(units=128, activation='relu', name='Layer1'))
    model.add(Dense(units=128, activation='relu', name='Layer2'))
    model.add(Dense(units=128, activation='relu', name='Layer3'))
    model.add(Dense(units=128, activation='relu', name='Layer4'))

    model.add(Dropout(rate=0.2, name='Dropout1', ))

    model.add(Dense(units=64, activation='relu', name='Layer5'))
    model.add(Dense(units=64, activation='relu', name='Layer6'))
    model.add(Dense(units=64, activation='relu', name='Layer7'))
    model.add(Dense(units=64, activation='relu', name='Layer8'))

    model.add(Dropout(rate=0.2, name='Dropout2'))

    model.add(Dense(units=32, activation='relu', name='Layer9'))
    model.add(Dense(units=32, activation='relu', name='Layer10'))
    model.add(Dense(units=32, activation='relu', name='Layer11'))
    model.add(Dense(units=32, activation='relu', name='Layer12'))

    model.add(Dense(units=y_train.shape[1], activation='softmax', name='Output_Layer'))
    model.summary()

    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit
    history = model.fit(x=X_train,
                        y=y_train,
                        validation_data=(X_test, y_test),
                        epochs=100)

    # predictions
    pred = np.argmax(model.predict(X_test), axis=1)

    # Classification Report
    print(classification_report(pred, np.argmax(y_test, 1),
                                target_names=['0. don\'t know', '1. Asphalt', '2. Meadows', '3. Gravel', '4. Trees',
                                              '5. Painted metal sheets', '6. Bare Soil', '7. Bitumen',
                                              '8. Self-Blocking Bricks', '9. Shadows']))