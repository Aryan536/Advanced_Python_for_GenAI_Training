import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train / 255.0)[..., None]          # add channel axis -> (60000, 28, 28, 1)
x_test = (x_test / 255.0)[..., None]

cnn = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=3, activation="relu"),   # 32 filters learn edges/strokes
    layers.MaxPooling2D(),                                 # downsample 2x
    layers.Conv2D(64, kernel_size=3, activation="relu"),   # 64 filters learn shapes
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
cnn.summary()
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=2)
print(f"\nCNN test accuracy: {cnn.evaluate(x_test, y_test, verbose=0)[1]:.4f}")
# Tip: in Colab use Runtime > Change runtime type > T4 GPU to train CNNs much faster
