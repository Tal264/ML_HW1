
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Find dataset
files = sorted(Path(".").rglob("105l001p16_spikes.mat"))
if not files:
    raise FileNotFoundError("Could not find 105l001p16_spikes.mat")

# Load trial data in batches
with h5py.File(files[0], "r") as f:
    stim = f["stim"][:].flatten()
    spikes = f[f["spikes"][0, 0]]

    n_trials, n_time, n_neurons = spikes.shape
    responses = np.zeros((n_trials, n_neurons))

    for start in range(0, n_trials, 100):
        end = min(start + 100, n_trials)
        responses[start:end] = spikes[start:end, :, :].sum(axis=1)

labels = np.sort(np.unique(stim))

# Estimate noise using within-stimulus trial variability.
# This is a simple noise proxy, not a definitive measure of noise.
variability = np.zeros(n_neurons)

for neuron in range(n_neurons):
    variances = [
        np.var(responses[stim == label, neuron], ddof=1)
        for label in labels
    ]
    variability[neuron] = np.mean(variances)

# Choose two most variable neurons, excluding neuron 5
candidates = [i for i in range(n_neurons) if i != 5]
noisy_neurons = sorted(
    candidates, key=lambda i: variability[i], reverse=True
)[:2]

print("Original neuron: 5")
print("Selected noisy neurons:", noisy_neurons)
print("Variability scores:", variability[noisy_neurons])

# Split trials within each stimulus, so every label is represented
rng = np.random.default_rng(42)
train_mask = np.zeros(n_trials, dtype=bool)

for label in labels:
    indices = np.where(stim == label)[0]
    rng.shuffle(indices)
    n_train = int(0.7 * len(indices))
    train_mask[indices[:n_train]] = True

test_mask = ~train_mask

# Models
def gaussian(x, B, A, mu, sigma):
    return B + A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def fit_predict(model_name, x_train, y_train, x_test):
    if model_name == "Linear":
        coefficients = np.polyfit(x_train, y_train, 1)
        return np.polyval(coefficients, x_test)

    if model_name == "Cubic":
        coefficients = np.polyfit(x_train, y_train, 3)
        return np.polyval(coefficients, x_test)

    if model_name == "Gaussian":
        p0 = [
            max(0, np.min(y_train)),
            max(0.1, np.max(y_train) - np.min(y_train)),
            x_train[np.argmax(y_train)],
            1.0
        ]
        params, _ = curve_fit(
            gaussian,
            x_train, y_train,
            p0=p0,
            bounds=(
                [0, 0, x_train.min(), 0.1],
                [np.inf, np.inf, x_train.max(), 20]
            ),
            maxfev=20000
        )
        return gaussian(x_test, *params)

# Fit on training mean responses; evaluate on held-out individual trials
model_names = ["Linear", "Cubic", "Gaussian"]
results = {}

for neuron in noisy_neurons:
    x_train = []
    y_train = []

    for label in labels:
        mask = (stim == label) & train_mask
        x_train.append(label)
        y_train.append(responses[mask, neuron].mean())

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train)

    # Evaluate each model on individual held-out trials
    x_test = stim[test_mask].astype(float)
    y_test = responses[test_mask, neuron]

    results[neuron] = {}

    for model_name in model_names:
        predictions = fit_predict(
            model_name, x_train, y_train, x_test
        )
        mse = np.mean((y_test - predictions) ** 2)
        results[neuron][model_name] = mse
        print(
            f"Neuron {neuron} | {model_name} | "
            f"Test MSE: {mse:.3f}"
        )

# Plot held-out test MSE
fig, axes = plt.subplots(
    1, 2, figsize=(12, 5), sharey=False
)

for ax, neuron in zip(axes, noisy_neurons):
    values = [results[neuron][m] for m in model_names]
    ax.bar(model_names, values)
    ax.set_title(f"Neuron {neuron}")
    ax.set_ylabel("Test MSE")
    ax.tick_params(axis="x", rotation=20)

plt.suptitle("Generalization on Two Noisier Neurons")
plt.tight_layout()
plt.savefig("part5_test_mse.png", dpi=300)
plt.show()
