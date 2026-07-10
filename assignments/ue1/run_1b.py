import os, sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow {tf.__version__}")

# KMNIST laden
(ds_train_raw, ds_test_raw), info = tfds.load(
    "kmnist", split=["train", "test"], as_supervised=True, with_info=True)
print(f"Train: {info.splits['train'].num_examples}, Test: {info.splits['test'].num_examples}")

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (-1,))
    return image, label

ds_train = ds_train_raw.map(preprocess).shuffle(10000).batch(128).prefetch(1)
ds_test  = ds_test_raw.map(preprocess).batch(128).prefetch(1)

def build_plain(units=256):
    return keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(units, activation="relu"),
        layers.Dense(units, activation="relu"),
        layers.Dense(units // 2, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

def build_regularized(units=256, dropout=0.3):
    return keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(units, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(units, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(units // 2, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout / 2),
        layers.Dense(10, activation="softmax"),
    ])

def build_with_skip(units=256, dropout=0.3):
    inp = keras.Input(shape=(784,))
    x = layers.Dense(units, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    shortcut = x
    x = layers.Dense(units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units // 2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, x)

def train_model(model, name, epochs=15):
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    hist = model.fit(ds_train, validation_data=ds_test, epochs=epochs, callbacks=[cb], verbose=1)
    print(f"\n{name}: Best Val-Acc = {max(hist.history['val_accuracy']):.4f}\n")
    return hist

print("\n=== Manuelle Architekturen ===")
train_model(build_plain(),       "A: Plain MLP")
train_model(build_regularized(), "B: BatchNorm+Dropout")
train_model(build_with_skip(),   "C: Skip-Connection")

print("\n=== Random Search (30 Konfigurationen) ===")

def generate_search_space(n=30, seed=42):
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        n_layers   = int(rng.integers(3, 6))
        base_units = int(rng.choice([128, 256, 512]))
        units = [max(int(base_units * (0.5 ** i)), 32) for i in range(n_layers)]
        configs.append({
            "learning_rate": float(rng.uniform(1e-4, 1e-2)),
            "activation":    str(rng.choice(["relu", "elu", "tanh"])),
            "units":         units,
            "dropout":       float(rng.uniform(0.1, 0.5)),
            "optimizer":     str(rng.choice(["adam", "sgd", "rmsprop"])),
            "batch_norm":    bool(rng.choice([True, False])),
        })
    return configs

def build_model_from_config(config):
    inp = keras.Input(shape=(784,))
    x = inp
    for u in config["units"]:
        x = layers.Dense(u, activation=config["activation"])(x)
        if config["batch_norm"]:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(config["dropout"])(x)
    x = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, x)

results = []
for i, cfg in enumerate(generate_search_space(n=30)):
    model = build_model_from_config(cfg)
    opt_map = {
        "adam":    keras.optimizers.Adam(cfg["learning_rate"]),
        "sgd":     keras.optimizers.SGD(cfg["learning_rate"], momentum=0.9),
        "rmsprop": keras.optimizers.RMSprop(cfg["learning_rate"]),
    }
    model.compile(optimizer=opt_map[cfg["optimizer"]],
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    hist = model.fit(ds_train, validation_data=ds_test, epochs=15, callbacks=cbs, verbose=0)
    val_acc = max(hist.history["val_accuracy"])
    results.append({"config": cfg, "val_accuracy": val_acc, "epochs": len(hist.history["val_accuracy"])})
    print(f"[{i+1:2d}/30] Val-Acc={val_acc:.4f}  opt={cfg['optimizer']:8s}  bn={str(cfg['batch_norm']):5s}  lr={cfg['learning_rate']:.5f}  act={cfg['activation']}")

results.sort(key=lambda r: r["val_accuracy"], reverse=True)
print("\nTop-5 Konfigurationen:")
for j, r in enumerate(results[:5]):
    c = r["config"]
    print(f"  [{j+1}] Val-Acc={r['val_accuracy']:.4f}  opt={c['optimizer']}  bn={c['batch_norm']}  lr={c['learning_rate']:.5f}  act={c['activation']}  units={c['units']}")

print("\nFertig!")
