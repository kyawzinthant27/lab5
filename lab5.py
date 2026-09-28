import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

threadpool_limits(limits=1)
data = pd.read_csv("cardio.csv", sep=";")
X = data.drop(columns=["id", "cardio"]).to_numpy(dtype=float)
y = data["cardio"].to_numpy(dtype=float).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y.ravel()
)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print("Training:", X_train.shape, "Test:", X_test.shape)

rng = np.random.default_rng(42)
initial = [
    rng.normal(0, np.sqrt(2 / 11), (11, 6)),
    np.zeros((1, 6)),
    rng.normal(0, np.sqrt(1 / 6), (6, 1)),
    np.zeros((1, 1))
]

def forward(X, weights, mask=None):
    W, b, v, c = weights
    z = X @ W + b
    h = np.maximum(0, z)
    if mask is not None:
        h = h * mask
    s = h @ v + c
    prediction = np.exp(-np.logaddexp(0, -s))
    return z, h, s, prediction

def loss(s, y):
    return np.mean(np.logaddexp(0, s) - y * s)

def gradients(X, y, weights, l2=0, mask=None):
    W, b, v, c = weights
    z, h, s, prediction = forward(X, weights, mask)
    ds = (prediction - y) / len(y)
    dv = h.T @ ds + l2 * v
    dc = ds.sum(axis=0, keepdims=True)
    dh = ds @ v.T
    if mask is not None:
        dh = dh * mask
    dz = dh * (z > 0)
    dW = X.T @ dz + l2 * W
    db = dz.sum(axis=0, keepdims=True)
    return [dW, db, dv, dc]

def train(improved=False):
    weights = [w.copy() for w in initial]
    m = [np.zeros_like(w) for w in weights]
    v = [np.zeros_like(w) for w in weights]
    rng = np.random.default_rng(123)
    history = []

    for step in range(1001):
        history.append(loss(forward(X_train, weights)[2], y_train))
        if step == 1000:
            break

        mask = None
        if improved:
            mask = rng.binomial(1, 0.9, (len(X_train), 6)) / 0.9
        grads = gradients(X_train, y_train, weights,
                          l2=0.0001 if improved else 0, mask=mask)

        for k in range(4):
            if improved:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                m_hat = m[k] / (1 - 0.9 ** (step + 1))
                v_hat = v[k] / (1 - 0.999 ** (step + 1))
                weights[k] -= 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
            else:
                weights[k] -= 0.05 * grads[k]

    return weights, history

baseline, baseline_loss = train()
print("Baseline training loss:", baseline_loss[-1])

modified, modified_loss = train(improved=True)
print("Modified training loss:", modified_loss[-1])

models = [("Baseline", baseline, baseline_loss),
          ("Adam + L2 + dropout", modified, modified_loss)]

for name, weights, history in models:
    z, h, s, prediction = forward(X_test, weights)
    accuracy = accuracy_score(y_test, prediction >= 0.5)
    print("\n", name)
    print("Training loss:", round(history[-1], 6))
    print("Test loss:", round(loss(s, y_test), 6))
    print("Test accuracy:", round(accuracy * 100, 2), "%")
    print("ROC AUC:", round(roc_auc_score(y_test, prediction), 4))
    print("Output weights:", np.round(weights[2].ravel(), 6))
    print("Output bias:", round(float(weights[3][0, 0]), 6))
    print("Loss at update 100:", round(history[100], 6))
    first = next((i for i, value in enumerate(history) if value < 0.60), None)
    print("First update below loss 0.60:", first)

plt.figure(figsize=(7, 4))
plt.plot(baseline_loss, "k-", label="Baseline")
plt.plot(modified_loss, "k--", label="Adam + L2 + dropout")
plt.xlabel("Updates")
plt.ylabel("Training BCE")
plt.legend()
plt.tight_layout()
plt.savefig("loss_history.png", dpi=150)
plt.show()

noise = np.random.default_rng(2026).normal(0, 0.1, X_test.shape)
for name, weights, history in models:
    clean = forward(X_test, weights)[3] >= 0.5
    noisy = forward(X_test + noise, weights)[3] >= 0.5
    clean_accuracy = accuracy_score(y_test, clean) * 100
    noisy_accuracy = accuracy_score(y_test, noisy) * 100
    print(name)
    print("Clean:", round(clean_accuracy, 2), "%")
    print("With noise:", round(noisy_accuracy, 2), "%")
    print("Drop:", round(clean_accuracy - noisy_accuracy, 2), "percentage points")

