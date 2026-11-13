import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

# Real dataset: 70,000 handwritten digits, 28x28 pixels
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0          # scale pixels to 0-1
print("Train:", x_train.shape, "Test:", x_test.shape)

model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),                         # 28x28 -> 784 vector
    layers.Dense(128, activation="relu"),     # hidden layer 1
    layers.Dropout(0.2),                      # regularisation: randomly switch off 20% of neurons
    layers.Dense(64, activation="relu"),      # hidden layer 2
    layers.Dense(10, activation="softmax"),   # 10 classes -> probabilities
])
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=10, batch_size=128,
                    validation_split=0.1, callbacks=[early_stop], verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="validation")
ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="validation")
ax[1].set_title("Accuracy"); ax[1].legend()
plt.show()

# Predict 5 test images
probs = model.predict(x_test[:5], verbose=0)
fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
for ax, img, pr, true in zip(axes, x_test[:5], probs, y_test[:5]):
    ax.imshow(img, cmap="gray_r"); ax.axis("off")
    ax.set_title(f"pred {pr.argmax()} ({pr.max():.0%})\ntrue {true}", fontsize=9)
plt.show()
