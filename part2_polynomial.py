
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Find the dataset file
files = sorted(Path(".").rglob("105l001p16_spikes.mat"))
if not files:
    raise FileNotFoundError("Could not find 105l001p16_spikes.mat")

# Load stimulus labels and V1 spike data
with h5py.File(files[0], "r") as f:
    stim = f["stim"][:].flatten()
    spikes = f[f["spikes"][0, 0]]

    n_trials, n_time, n_neurons = spikes.shape
    responses = np.zeros((n_trials, n_neurons))

    # Process in batches to avoid loading the full dataset into memory
    batch_size = 100
    for start in range(0, n_trials, batch_size):
        end = min(start + batch_size, n_trials)
        batch = spikes[start:end, :, :]
        responses[start:end, :] = batch.sum(axis=1)

# Calculate the mean response for each stimulus label
labels = np.sort(np.unique(stim))
mean_responses = np.array([
    responses[stim == label].mean(axis=0)
    for label in labels
])

# Use the same neuron as in Part 1
neuron_index = 5
y = mean_responses[:, neuron_index]

# Stimulus labels are used as x-values for now
x = labels.astype(float)

# Fit polynomial models and calculate MSE
degrees = [1, 3, 4, 7]
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="black", s=50, label="Measured responses")

x_smooth = np.linspace(x.min(), x.max(), 300)

for degree in degrees:
    coefficients = np.polyfit(x, y, degree)
    predictions = np.polyval(coefficients, x)
    mse = np.mean((y - predictions) ** 2)

    print(f"\nPolynomial degree: {degree}")
    print("Coefficients:", coefficients)
    print(f"MSE: {mse:.4f}")

    plt.plot(
        x_smooth,
        np.polyval(coefficients, x_smooth),
        label=f"Degree {degree} (MSE={mse:.2f})"
    )

plt.xlabel("Stimulus label")
plt.ylabel("Mean spike count per trial")
plt.title(f"Polynomial Fits for Neuron {neuron_index}")
plt.xticks(labels)
plt.legend()
plt.tight_layout()
plt.savefig("part2_polynomial.png", dpi=300)
plt.show()
