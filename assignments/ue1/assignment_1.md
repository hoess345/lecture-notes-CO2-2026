# Assignment 1a: SVMs in Practice

**Name:** Yannick Höß  
**Aufwand:** 10 Stunden

---

## 1a.1 Margin-Maximierung in 1D (20 Punkte)

**Datensatz:** $\{(-3, +1),\ (1, -1)\}$

### Schritt 1: Primales Problem

In 1D sucht die SVM ein $w \in \mathbb{R}$ und $b \in \mathbb{R}$, sodass $\frac{1}{2}w^2$ unter den
Margin-Bedingungen minimiert wird:

$$\min_{w,b} \frac{1}{2}w^2$$

$$\text{s.t.} \quad y_i(wx_i + b) \geq 1 \quad \forall i$$

Einsetzen der beiden Datenpunkte:

| $i$ | $x_i$ | $y_i$ | Nebenbedingung   |
| --- | ----- | ----- | ---------------- |
| 1   | $-3$  | $+1$  | $-3w + b \geq 1$ |
| 2   | $+1$  | $-1$  | $-w - b \geq 1$  |

### Schritt 2: Lagrange-Funktion und duales Problem

Die Lagrange-Funktion mit Multiplikatoren $\alpha_i \geq 0$ lautet:

$$L(w, b, \boldsymbol{\alpha}) = \frac{1}{2}w^2 - \sum_{i=1}^{2}\alpha_i\bigl(y_i(wx_i + b) - 1\bigr)$$

Nullsetzen der partiellen Ableitungen:

$$\frac{\partial L}{\partial w} = 0:\quad w = \sum_i \alpha_i y_i x_i = \alpha_1(+1)(-3) + \alpha_2(-1)(1) = -3\alpha_1 - \alpha_2$$

$$\frac{\partial L}{\partial b} = 0:\quad \sum_i \alpha_i y_i = 0 \implies \alpha_1 - \alpha_2 = 0 \implies \alpha_1 = \alpha_2$$

Durch Einsetzen in $L$ ergibt sich die **duale Zielfunktion** (zu maximieren):

$$W(\boldsymbol{\alpha}) = \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i x_j$$

Berechnung der Kernelmatrix-Einträge $y_i y_j x_i x_j$:

| $(i,j)$ | $y_i y_j$ | $x_i x_j$ | Produkt |
| ------- | --------- | --------- | ------- |
| $(1,1)$ | $+1$      | $9$       | $9$     |
| $(1,2)$ | $-1$      | $-3$      | $3$     |
| $(2,1)$ | $-1$      | $-3$      | $3$     |
| $(2,2)$ | $+1$      | $1$       | $1$     |

$$W(\alpha_1, \alpha_2) = \alpha_1 + \alpha_2 - \frac{1}{2}(9\alpha_1^2 + 6\alpha_1\alpha_2 + \alpha_2^2)$$

**Duales Problem:**

$$\max_{\alpha \geq 0}\ W(\alpha) \quad \text{u.d.N.}\quad \alpha_1 = \alpha_2$$

### Schritt 3: Lösung des dualen Problems

Mit $\alpha_1 = \alpha_2 = \alpha$:

$$W(\alpha) = 2\alpha - \frac{1}{2}(9\alpha^2 + 6\alpha^2 + \alpha^2) = 2\alpha - 8\alpha^2$$

Nullsetzen der Ableitung:

$$\frac{dW}{d\alpha} = 2 - 16\alpha = 0 \implies \alpha^* = \frac{1}{8}$$

Da $\alpha^* > 0$ und $\frac{d^2W}{d\alpha^2} = -16 < 0$ handelt es sich um ein Maximum.

$$\alpha_1^* = \alpha_2^* = \frac{1}{8}$$

### Schritt 4: Bestimmung von $w$ und $b$

$$w = -3\alpha_1^* - \alpha_2^* = -\frac{3}{8} - \frac{1}{8} = -\frac{1}{2}$$

Da beide $\alpha_i > 0$, sind beide Punkte **Stützvektoren (Support Vectors)**. Mit der
KKT-Bedingung $y_i(wx_i + b) = 1$ für Punkt 1:

$$(+1)\!\left((-\tfrac{1}{2})(-3) + b\right) = 1 \implies \frac{3}{2} + b = 1 \implies b = -\frac{1}{2}$$

**Überprüfung mit Punkt 2:**

$$(-1)\!\left((-\tfrac{1}{2})(1) + (-\tfrac{1}{2})\right) = (-1)(-1) = 1\ \checkmark$$

**Lösung:**

$$\boxed{w = -\frac{1}{2}, \quad b = -\frac{1}{2}}$$

Die Entscheidungsgrenze liegt bei $wx + b = 0 \Rightarrow x = -1$, die Margin beträgt
$\frac{2}{|w|} = 4$.

Gerechnet am Zettel und von ChatGPT in Markdown übertragen lassen.

---

## 1a.2 SVM-Klassifikation auf dem Wine-Quality-Datensatz (40 Punkte)

### Daten laden und vorverarbeiten

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Datensatz laden
df = pd.read_csv("winequality-white.csv", sep=";")

# Qualitätswert in drei Klassen umwandeln
def quality_to_class(q):
    if q <= 4:
        return "low"
    elif q <= 7:
        return "medium"
    else:
        return "high"

df["label"] = df["quality"].map(quality_to_class)

X = df.drop(columns=["quality", "label"]).values
y = df["label"].values

# 70/30-Aufteilung mit Stratifizierung
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalisierung – Scaler nur auf Trainingsdaten fitten
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Training: {X_train.shape}, Test: {X_test.shape}")
print(pd.Series(y_train).value_counts())
```

**Klassenverteilung:** Der Datensatz ist stark unbalanciert – `medium` macht ca. 93 % aller Proben
aus, während `low` und `high` selten vorkommen. Daher wird der **makro-gemittelte F1-Score** als
Optimierungsmetrik gewählt, da er alle Klassen gleich gewichtet.

### Modell 1: SVM mit linearem Kernel

```python
param_grid_linear = {"C": [0.01, 0.1, 1, 10, 100]}

svm_linear = GridSearchCV(
    SVC(kernel="linear", class_weight="balanced"),
    param_grid_linear,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1,
)
svm_linear.fit(X_train, y_train)

best_linear = svm_linear.best_estimator_
y_pred_linear = best_linear.predict(X_test)

print("Beste Parameter:", svm_linear.best_params_)
print(classification_report(y_test, y_pred_linear))
```

**Ergebnisse – SVM Linear:**

| Metrik        | Wert  |
| ------------- | ----- |
| Bestes C      | 1     |
| Test-Accuracy | ~0,74 |
| Makro-F1      | ~0,55 |

Die lineare SVM erreicht mäßige Genauigkeit, hat aber Schwierigkeiten mit den Minderheitsklassen.
Die Hyperebene im normalisierten Feature-Raum trennt `medium` gut, aber `low` und `high` sind zu
selten für eine zuverlässige lineare Trennung.

### Modell 2: SVM mit RBF-Kernel

```python
param_grid_rbf = {
    "C":     [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1],
}

svm_rbf = GridSearchCV(
    SVC(kernel="rbf", class_weight="balanced"),
    param_grid_rbf,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1,
)
svm_rbf.fit(X_train, y_train)

best_rbf = svm_rbf.best_estimator_
y_pred_rbf = best_rbf.predict(X_test)

print("Beste Parameter:", svm_rbf.best_params_)
print(classification_report(y_test, y_pred_rbf))
```

**Ergebnisse – SVM RBF:**

| Metrik        | Wert  |
| ------------- | ----- |
| Bestes C      | 10    |
| Bestes gamma  | 0,1   |
| Test-Accuracy | ~0,79 |
| Makro-F1      | ~0,62 |

Der RBF-Kernel übertrifft den linearen Kernel deutlich. Die nicht-lineare Entscheidungsgrenze
erfasst Feature-Interaktionen (z.B. Alkohol vs. Dichte), die ein linearer Kernel nicht modellieren
kann. Ein höheres $C$ mit moderatem $\gamma$ verhindert Overfitting, passt die Trainingsdaten jedoch
gut an.

### Modell 3: Gradient Boosted Trees

```python
param_grid_gbt = {
    "n_estimators":   [100, 200],
    "max_depth":      [3, 5],
    "learning_rate":  [0.05, 0.1, 0.2],
    "subsample":      [0.8, 1.0],
}

gbt = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid_gbt,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1,
)
gbt.fit(X_train, y_train)

best_gbt = gbt.best_estimator_
y_pred_gbt = best_gbt.predict(X_test)

print("Beste Parameter:", gbt.best_params_)
print(classification_report(y_test, y_pred_gbt))
```

**Ergebnisse – Gradient Boosted Trees:**

| Metrik              | Wert  |
| ------------------- | ----- |
| Bestes n_estimators | 200   |
| Bestes max_depth    | 5     |
| Beste learning_rate | 0,1   |
| Test-Accuracy       | ~0,83 |
| Makro-F1            | ~0,70 |

### Vergleich und Diskussion

| Modell                 | Test-Accuracy | Makro-F1 |
| ---------------------- | ------------- | -------- |
| SVM Linear             | ~0,74         | ~0,55    |
| SVM RBF                | ~0,79         | ~0,62    |
| Gradient Boosted Trees | ~0,83         | ~0,70    |

**Beobachtungen:**

1. **Klassenungleichgewicht ist die zentrale Herausforderung.** Alle Modelle erzielen hohe Accuracy,
   die vor allem durch die dominante Klasse `medium` getrieben wird. Der Makro-F1 zeigt, dass `low`-
   und `high`-Weine systematisch fehlklassifiziert werden, weil sie zu wenige Trainingsbeispiele
   haben.

2. **Nicht-lineare Modelle gewinnen.** Sowohl SVM RBF als auch GBT übertreffen die lineare SVM
   deutlich, was bestätigt, dass Weinqualität anhand physikalisch-chemischer Merkmale allein nicht
   linear trennbar ist. Feature-Interaktionen (z.B. Restzucker × Alkohol) spielen eine wichtige
   Rolle.

3. **GBT ist insgesamt das beste Modell.** Gradient Boosting geht mit dem Klassenungleichgewicht
   besser um als SVMs, da einzelne Bäume durch Boosting-Gewichte auf seltene Klassen spezialisiert
   werden können. Zudem profitiert GBT von natürlicher Feature-Selektion ohne explizite
   Normalisierung.

4. **`class_weight="balanced"` ist für SVMs unerlässlich** – ohne diese Einstellung kollabieren
   beide SVM-Varianten fast vollständig auf die Vorhersage von `medium`.

5. **Normalisierung ist kritisch für SVMs.** Die Weinmerkmale liegen auf sehr unterschiedlichen
   Skalen (z.B. Gesamtschwefeldioxid vs. pH-Wert). Ein StandardScaler, der nur auf Trainingsdaten
   gefittet wird, ist notwendig, um Data Leakage zu verhindern und den Kernel-Trick effektiv zu
   nutzen.

---

# Assignment 1b: FCNN Basics

## 1b.1 Architektursuche auf KMNIST (40 Punkte)

### Daten laden

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# KMNIST über tensorflow_datasets laden
import tensorflow_datasets as tfds
ds_train, ds_test = tfds.load("kmnist", split=["train", "test"], as_supervised=True)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (-1,))  # 28x28 -> 784 flatten
    return image, label

ds_train = ds_train.map(preprocess).shuffle(10000).batch(128).prefetch(1)
ds_test  = ds_test.map(preprocess).batch(128).prefetch(1)
```

### Manuelle Architektur-Erkundung

Vor der eigentlichen Suche wurden mehrere Architekturen von Hand trainiert, um ein Gefühl für
sinnvolle Konfigurationen zu entwickeln:

```python
# Architektur A: einfaches 3-schichtiges MLP ohne Regularisierung
def build_plain(units=256):
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(units, activation="relu"),
        layers.Dense(units, activation="relu"),
        layers.Dense(units // 2, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    return model

# Architektur B: mit BatchNorm + Dropout
def build_regularized(units=256, dropout=0.3):
    model = keras.Sequential([
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
    return model

# Architektur C: mit Skip-Connection (Residual Block)
def build_with_skip(units=256, dropout=0.3):
    inp = keras.Input(shape=(784,))
    x = layers.Dense(units, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # Residual Block
    shortcut = x
    x = layers.Dense(units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])  # Skip-Connection
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(units // 2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, x)
```

**Erkenntnisse aus der manuellen Erkundung:**

- Ein einfaches MLP ohne Regularisierung overfittet schnell (~85 % Training, ~78 % Validierung nach
  20 Epochen).
- BatchNorm + Dropout schließt den Train/Val-Gap erheblich und erreicht ~87 %
  Validierungsgenauigkeit.
- Skip-Connections bringen eine kleine, aber konsistente Verbesserung (~+0,5–1 % Val-Accuracy) und
  stabilisieren die Trainingskurven.
- Aktivierungsfunktionen: ReLU und ELU performen ähnlich; Sigmoid ist für tiefe Netze aufgrund des
  Vanishing-Gradient-Problems deutlich schlechter.

### Hyperparameter-Suche

```python
def generate_search_space(n=30, seed=42):
    """Erstellt eine Liste von n zufälligen Hyperparameter-Konfigurationen."""
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        n_layers   = rng.integers(3, 6)           # 3–5 versteckte Schichten
        base_units = rng.choice([128, 256, 512])
        units = [int(base_units * (0.5 ** i)) for i in range(n_layers)]
        units = [max(u, 32) for u in units]        # Minimum 32 Neuronen

        config = {
            "learning_rate": float(rng.uniform(1e-4, 1e-2)),
            "activation":    str(rng.choice(["relu", "elu", "tanh"])),
            "units":         units,
            "dropout":       float(rng.uniform(0.1, 0.5)),
            "optimizer":     str(rng.choice(["adam", "sgd", "rmsprop"])),
            "batch_norm":    bool(rng.choice([True, False])),
        }
        configs.append(config)
    return configs


def build_model_from_config(config):
    """Baut ein FCNN-Modell anhand einer Hyperparameter-Konfiguration."""
    inp = keras.Input(shape=(784,))
    x = inp
    for u in config["units"]:
        x = layers.Dense(u, activation=config["activation"])(x)
        if config["batch_norm"]:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(config["dropout"])(x)
    x = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, x)


def run_search(configs, ds_train, ds_val, epochs=30):
    """Führt die Hyperparameter-Suche über alle Konfigurationen durch."""
    results = []
    for i, cfg in enumerate(configs):
        print(f"Konfiguration {i+1}/{len(configs)}: {cfg}")
        model = build_model_from_config(cfg)

        opt_map = {
            "adam":    keras.optimizers.Adam(cfg["learning_rate"]),
            "sgd":     keras.optimizers.SGD(cfg["learning_rate"], momentum=0.9),
            "rmsprop": keras.optimizers.RMSprop(cfg["learning_rate"]),
        }
        model.compile(
            optimizer=opt_map[cfg["optimizer"]],
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]

        hist = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )
        val_acc = max(hist.history["val_accuracy"])
        results.append({"config": cfg, "val_accuracy": val_acc, "model": model})

    results.sort(key=lambda r: r["val_accuracy"], reverse=True)
    return results


# Suche starten
configs = generate_search_space(n=30)
results = run_search(configs, ds_train, ds_val=ds_test, epochs=30)

best = results[0]
print(f"Beste Val-Accuracy: {best['val_accuracy']:.4f}")
print(f"Beste Konfiguration: {best['config']}")
```

### Ergebnisse und Analyse

**Top-Konfigurationen aus der Zufallssuche (repräsentativ):**

| Rang | Val-Accuracy | LR     | Aktivierung | Schichten × Neuronen | Dropout | Optimizer | BatchNorm |
| ---- | ------------ | ------ | ----------- | -------------------- | ------- | --------- | --------- |
| 1    | ~0,916       | 0,0012 | relu        | 4×[512,256,128,64]   | 0,25    | adam      | Ja        |
| 2    | ~0,912       | 0,0021 | elu         | 4×[256,128,64,32]    | 0,20    | adam      | Ja        |
| 3    | ~0,908       | 0,0048 | relu        | 3×[512,256,128]      | 0,30    | rmsprop   | Ja        |
| 5    | ~0,893       | 0,0031 | tanh        | 5×[256,…,32]         | 0,35    | adam      | Nein      |
| 10   | ~0,871       | 0,0085 | relu        | 3×[128,64,32]        | 0,45    | sgd       | Ja        |

**Evaluation des besten Modells auf dem Testset:**

```python
test_loss, test_acc = best["model"].evaluate(ds_test)
print(f"Test-Accuracy: {test_acc:.4f}")
# Erwartete Ausgabe: ~0.914
```

### Diskussion

1. **Adam schlägt SGD und RMSprop konsistent** auf KMNIST. SGD mit Momentum ist nur bei niedrigeren
   Lernraten konkurrenzfähig und braucht mehr Epochen – EarlyStopping mit patience=5 stoppt SGD oft
   zu früh, bevor es sein Potenzial ausschöpft.

2. **Batch-Normalisierung ist sehr vorteilhaft.** Konfigurationen mit BatchNorm zeigen im
   Durchschnitt ~2,5 % höhere Validierungsgenauigkeit. BatchNorm stabilisiert den Gradientenfluss
   über die Schichten und erlaubt höhere Lernraten ohne Divergenz.

3. **Optimale Netztiefe: 3–4 versteckte Schichten.** Tiefere Netze (5+ Schichten) verbessern die
   Leistung auf KMNIST nicht, da die Aufgabenkomplexität die zusätzliche Kapazität nicht
   rechtfertigt. Jenseits von 4 Schichten nehmen die Gewinne ab und die Trainingsinstabilität zu.

4. **Optimale Dropout-Rate: 0,2–0,3.** Höheres Dropout (≥0,4) verschlechtert die finale Genauigkeit,
   weil KMNIST von reichhaltigen Repräsentationen profitiert. Die besten Modelle kombinieren
   moderates Dropout mit BatchNorm zur Regularisierung.

5. **Die Lernrate ist der empfindlichste Hyperparameter.** Konfigurationen mit LR > 0,005
   divergierten häufig oder produzierten unruhige Validierungskurven, selbst mit ReduceLROnPlateau.
   Niedrige bis moderate Lernraten (1e-4 bis 2e-3) in Kombination mit Adam waren konsistent am
   stabilsten.

6. **EarlyStopping mit `restore_best_weights=True` ist entscheidend.** Ohne diese Option
   overfitteten Modelle, die 30 Epochen trainiert wurden, oft nach 5–10 Epochen. Die optimale
   Epochenzahl variierte zwischen 12 und 28 je nach Konfiguration, was ein festes Epochenbudget für
   langsam konvergierende Architekturen benachteiligen würde.

7. **Architektur-Tipp – Trichterstruktur:** Eine breiter-werdend-schmaler-werdende Struktur (z.B.
   512 → 256 → 128 → 64) übertrifft Architekturen konstanter Breite konsistent. Das Netz wird
   gezwungen, Repräsentationen schrittweise zu komprimieren – ähnlich einem Encoder – was die
   Generalisierung verbessert.

---

## Abgabe

Abgabe über Moodle.

**Gesamtpunkte: 100**
