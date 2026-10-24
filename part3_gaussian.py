
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Find the dataset file
files = sorted(Path(".").rglob("105l001p16_spikes.mat"))
if not files:
    raise FileNotFoundError("Could not find 105l001p16_spikes.mat")

# Load the stimulus labels and V1 spike data
with h5py.File(files[0], "r") as f:
    stim = f["stim"][:].flatten()
    spikes = f[f["spikes"][0, 0]]

    n_trials, n_time, n_neurons = spikes.shape
    responses = np.zeros((n_trials, n_neurons))

    # Process in batches to reduce memory usage
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

# Use the same neuron as in Parts 1 and 2
neuron_index = 5
x = labels.astype(float)
y = mean_responses[:, neuron_index]

# Define the Gaussian tuning curve
def gaussian(theta, B, A, mu, sigma):
    return B + A * np.exp(-((theta - mu) ** 2) / (2 * sigma ** 2))

# Initial parameter estimates
baseline_guess = np.min(y)
amplitude_guess = np.max(y) - baseline_guess
preferred_guess = x[np.argmax(y)]
width_guess = 1.0

initial_guess = [
    baseline_guess,
    amplitude_guess,
    preferred_guess,
    width_guess
]

# Fit the Gaussian curve
params, _ = curve_fit(
    gaussian,
    x,
    y,
    p0=initial_guess,
    bounds=(
        [0, 0, x.min(), 0.1],
        [np.inf, np.inf, x.max(), 20]
    ),
    maxfev=20000
)

B, A, mu, sigma = params

# Calculate predictions and mean squared error
predictions = gaussian(x, *params)
mse = np.mean((y - predictions) ** 2)

print("Selected neuron index:", neuron_index)
print(f"Baseline (B): {B:.3f}")
print(f"Amplitude (A): {A:.3f}")
print(f"Preferred stimulus label (mu): {mu:.3f}")
print(f"Tuning width (sigma): {sigma:.3f}")
print(f"Mean squared error (MSE): {mse:.3f}")

# Plot measured responses and fitted Gaussian
x_smooth = np.linspace(x.min(), x.max(), 300)

plt.figure(figsize=(9, 6))
plt.scatter(x, y, color="black", s=50, label="Measured responses")
plt.plot(
    x_smooth,
    gaussian(x_smooth, *params),
    label=f"Gaussian fit (MSE={mse:.2f})"
)

plt.xlabel("Stimulus label")
plt.ylabel("Mean spike count per trial")
plt.title(f"Gaussian Tuning Curve — Neuron {neuron_index}")
plt.xticks(labels)
plt.legend()
plt.tight_layout()
plt.savefig("part3_gaussian.png", dpi=300)
plt.show()
