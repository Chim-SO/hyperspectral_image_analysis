import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale

if __name__ == '__main__':
    # data:
    data = loadmat('data/raw/pavia_u/PaviaU.mat')['paviaU']
    gt = loadmat('data/raw/pavia_u/PaviaU_gt.mat')['paviaU_gt']
    df = pd.DataFrame(data.reshape(data.shape[0] * data.shape[1], -1))
    df.columns = [f'band{i}' for i in range(1, df.shape[-1] + 1)]
    df['class'] = gt.ravel()
    print(df.shape)

    # Train data:
    X = df.iloc[:, :-1].values
    X_scaled = minmax_scale(X)
    print(f"Train Data: {X.shape}")

    # Model:
    encoding_dim = 20
    n_col = 103
    input_dim = Input(shape=(n_col,), name='InputLayer')

    # Encoder Layers
    encoded1 = Dense(100, activation='relu', name='EncodeLayer1')(input_dim)
    encoded2 = Dense(90, activation='relu', name='EncodeLayer2')(encoded1)
    encoded3 = Dense(85, activation='relu', name='EncodeLayer3')(encoded2)
    encoded4 = Dense(80, activation='relu', name='EncodeLayer4')(encoded3)
    encoded5 = Dense(75, activation='relu', name='EncodeLayer5')(encoded4)
    encoded6 = Dense(70, activation='relu', name='EncodeLayer6')(encoded5)

    # Coded Part
    encoded7 = Dense(encoding_dim, activation='linear', name='CodeLayer')(encoded6)

    # Decoder Layers
    decoded1 = Dense(70, activation='relu', name='DecodeLayer1')(encoded7)
    decoded2 = Dense(75, activation='relu', name='DecodeLayer2')(decoded1)
    decoded3 = Dense(80, activation='relu', name='DecodeLayer3')(decoded2)
    decoded4 = Dense(85, activation='relu', name='DecodeLayer4')(decoded3)
    decoded5 = Dense(90, activation='relu', name='DecodeLayer5')(decoded4)
    decoded6 = Dense(100, activation='relu', name='DecodeLayer6')(decoded5)
    decoded7 = Dense(n_col, activation='sigmoid', name='OutputLayer')(decoded6)

    # Combine Encoder and Deocder layers
    autoencoder = Model(inputs=input_dim, outputs=decoded7)
    autoencoder.summary()

    # Compile the Model
    autoencoder.compile(optimizer='adam',
                        loss='mse',
                        metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()]
                        )

    # Fit
    hist = autoencoder.fit(X_scaled,
                           X_scaled,
                           epochs=100,
                           batch_size=256,
                           shuffle=True)

    # Seperate encoder part from the auto encoder model
    encoder = Model(inputs=input_dim, outputs=encoded7)

    # Summary
    encoder.summary()

    # Get the data with the reduced dimesion

    encoded_X = pd.DataFrame(encoder.predict(X_scaled))

    # Add Column Names
    encoded_X = encoded_X.add_prefix('feature_')

    # Add the class labels for classification purposes
    encoded_X['class'] = df.loc[:, ['class']]

    print(encoded_X.head())
