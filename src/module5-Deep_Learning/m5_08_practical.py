import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# PRACTICAL: review sentiment classifier on 50,000 real IMDB movie reviews
tf.random.set_seed(42)
VOCAB, MAXLEN = 10_000, 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB)
print("Reviews:", len(x_train), "train /", len(x_test), "test")

# Every review is already a list of word ids; pad/cut them to the same length
x_train = keras.utils.pad_sequences(x_train, maxlen=MAXLEN)
x_test = keras.utils.pad_sequences(x_test, maxlen=MAXLEN)

model = keras.Sequential([
    keras.Input(shape=(MAXLEN,)),
    layers.Embedding(VOCAB, 32),                 # each word -> 32-D learned vector
    layers.GlobalAveragePooling1D(),             # average word vectors -> review vector
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),       # P(positive)
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=256, validation_split=0.2, verbose=2)
print(f"\nTest accuracy: {model.evaluate(x_test, y_test, verbose=0)[1]:.4f}")

# Score our own reviews: convert words to the same ids the dataset uses
word_index = keras.datasets.imdb.get_word_index()
def encode(text):
    ids = [1]                                    # 1 = start token
    for word in text.lower().replace(",", " ").replace(".", " ").split():
        idx = word_index.get(word)
        ids.append(idx + 3 if idx is not None and idx + 3 < VOCAB else 2)   # 2 = unknown word
    return keras.utils.pad_sequences([ids], maxlen=MAXLEN)

for review in ["An excellent and wonderful story, brilliant acting",
               "Boring, waste of time. The worst plot and terrible acting",
               "It was okay, some good moments but too long"]:
    p = float(model.predict(encode(review), verbose=0)[0, 0])
    print(f"{p:.2f} {'POSITIVE' if p > 0.5 else 'NEGATIVE'} : {review}")
