# STEP 9: Deep learning model (Keras neural network)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

n_features = X_train_p.shape[1]
nn = keras.Sequential([
    keras.Input(shape=(n_features,)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])
nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
           loss="binary_crossentropy",
           metrics=[keras.metrics.AUC(name="auc")])

# Returns are the minority class -> give them more weight
neg, pos = np.bincount(y_train)
class_weight = {0: 1.0, 1: neg / pos}

early_stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                           restore_best_weights=True)
history = nn.fit(X_train_p, y_train.values, validation_split=0.2, epochs=60, batch_size=64,
                 class_weight=class_weight, callbacks=[early_stop], verbose=0)
print("Epochs trained:", len(history.history["loss"]))

probas["Neural Network"] = nn.predict(X_test_p, verbose=0).ravel()
nn_scores = score_model("Neural Network", y_test, probas["Neural Network"])
nn_scores.pop("model")
results_df.loc["Neural Network"] = pd.Series(nn_scores).round(3)
print(results_df)

plt.figure(figsize=(8, 4))
plt.plot(history.history["auc"], label="train AUC")
plt.plot(history.history["val_auc"], label="validation AUC")
plt.xlabel("Epoch"); plt.title("Neural network training"); plt.legend()
plt.show()
