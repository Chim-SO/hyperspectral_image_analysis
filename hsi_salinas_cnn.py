import matplotlib.pyplot as plt

import numpy as np
import earthpy.plot as ep
import seaborn as sns
import earthpy.spatial as es

from keras import Sequential
from keras.utils import to_categorical

from scipy.io import loadmat

import pandas as pd

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.models import Model


from tqdm import tqdm
from numpy.random import seed
from time import time

if __name__ == '__main__':
    # Data
    data = loadmat('data/raw/salinas/Salinas_corrected.mat')['salinas_corrected']
    gt = loadmat('data/raw/salinas/Salinas_gt.mat')['salinas_gt']

    print(f'Data Shape: {data.shape[:-1]}\nNumber of Bands: {data.shape[-1]}')
    df = pd.DataFrame(data.reshape(data.shape[0] * data.shape[1], -1))
    df.columns = [f'band{i}' for i in range(1, df.shape[-1] + 1)]
    df['class'] = gt.ravel()

    # input
    t_df = df[df['class'] != 0]
    ind = ['band' + str(i) for i in range(1, t_df.shape[-1])]
    X = t_df.loc[:, ind]
    y = to_categorical(t_df.loc[:, 'class'])
    X_scaled = minmax_scale(X, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                        stratify=y,
                                                        test_size=0.30)
    print(f"X_train: {X_train.shape}\ny_train: {y_train.shape}\nX_test: {X_test.shape}\ny_test: {y_test.shape}")

    # model
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1), name='Layer1'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', name='Layer2'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', name='Layer3'))

    model.add(MaxPooling1D(pool_size=2, name='MaxPooling_Layer1'))
    model.add(Dropout(0.4, name='Dropout1'))

    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', name='Layer4'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', name='Layer5'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', name='Layer6'))

    model.add(MaxPooling1D(pool_size=2, name='MaxPooling_Layer2'))
    model.add(Dropout(0.4, name='Dropout2'))

    model.add(Flatten(name='Flatten'))

    model.add(Dense(25, activation='relu', name='DenseLayer'))

    model.add(Dense(y_train.shape[1], activation='softmax', name='OutputLayer'))

    model.summary()

    # Training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X_train,
                     y_train,
                     epochs=100,
                     batch_size=256,
                     validation_data=(X_test, y_test))

    # predictions
    pred = np.argmax(model.predict(X_test), axis=1)

    # Display conf:
    plt.figure(figsize=(12, 12))

    classes = ['1.Brocoli_green_weeds_1', '2.Brocoli_green_weeds_2',
               '3.Fallow',
               '4.Fallow_rough_plow',
               '5.Fallow_smooth',
               '6.Stubble',
               '7.Celery',
               '8.Grapes_untrained',
               '9.Soil_vinyard_develop',
               '10.Corn_senesced_green_weeds',
               '11.Lettuce_romaine_4wk',
               '12.Lettuce_romaine_5wk',
               '13.Lettuce_romaine_6wk',
               '14.Lettuce_romaine_7wk',
               '15.Vinyard_untrained',
               '16.Vinyard_vertical_trellis']

    mat = confusion_matrix(np.add(pred, 1), np.add(np.argmax(y_test, 1), 1))
    df_cm = pd.DataFrame(mat, index=classes, columns=classes)
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.show()