
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

    # Process in batches to reduce memory usage
    batch_size = 100
    for start in range(0, n_trials, batch_size):
        end = min(start + batch_size, n_trials)
        batch = spikes[start:end, :, :]
        responses[start:end, :] = batch.sum(axis=1)

# Mean response for each stimulus label
labels = np.sort(np.unique(stim))
mean_responses = np.array([
    responses[stim == label].mean(axis=0)
    for label in labels
])

# Same neuron used in Parts 1–3
neuron_index = 5
y = mean_responses[:, neuron_index]

# IMPORTANT:
# This assumes labels 1–8 correspond in order to orientations
# 0, 22.5, ..., 157.5 degrees. Verify this mapping in the
# assignment or dataset documentation before interpreting angles.
theta_deg = (labels - 1) * 22.5
theta_rad = np.deg2rad(theta_deg)

# Von Mises tuning curve for orientation (period = 180 degrees)
def von_mises(theta, B, A, mu, kappa):
    return B + A * np.exp(kappa * (np.cos(2 * (theta - mu)) - 1))

# Initial parameter estimates
baseline_guess = max(0, np.min(y))
amplitude_guess = max(0.1, np.max(y) - baseline_guess)
preferred_guess = theta_rad[np.argmax(y)]

initial_guess = [
    baseline_guess,
    amplitude_guess,
    preferred_guess,
    1.0
]

# Fit the Von Mises curve
params, _ = curve_fit(
    von_mises,
    theta_rad,
    y,
    p0=initial_guess,
    bounds=(
        [0, 0, 0, 0],
        [np.inf, np.inf, np.pi, 100]
    ),
    maxfev=30000
)

B, A, mu, kappa = params
predictions = von_mises(theta_rad, *params)
mse_von_mises = np.mean((y - predictions) ** 2)

# Fit a Gaussian to the same orientation data for comparison
def gaussian(theta, B, A, mu, sigma):
    return B + A * np.exp(-((theta - mu) ** 2) / (2 * sigma ** 2))

gaussian_params, _ = curve_fit(
    gaussian,
    theta_rad,
    y,
    p0=[
        max(0, np.min(y)),
        max(0.1, np.max(y) - np.min(y)),
        preferred_guess,
        np.deg2rad(30)
    ],
    bounds=(
        [0, 0, 0, np.deg2rad(1)],
        [np.inf, np.inf, np.pi, np.pi]
    ),
    maxfev=30000
)

gaussian_predictions = gaussian(theta_rad, *gaussian_params)
mse_gaussian = np.mean((y - gaussian_predictions) ** 2)

# Report results
print("Selected neuron index:", neuron_index)
print(f"Von Mises baseline (B): {B:.3f}")
print(f"Von Mises amplitude (A): {A:.3f}")
print(f"Preferred orientation (mu): {np.rad2deg(mu):.2f} degrees")
print(f"Concentration (kappa): {kappa:.3f}")
print(f"Von Mises MSE: {mse_von_mises:.3f}")
print(f"Gaussian MSE: {mse_gaussian:.3f}")

# Plot the fitted curves across the full 180-degree orientation range
theta_plot_deg = np.linspace(0, 180, 500)
theta_plot_rad = np.deg2rad(theta_plot_deg)

plt.figure(figsize=(10, 6))
plt.scatter(theta_deg, y, color="black", s=50, label="Measured responses")
plt.plot(
    theta_plot_deg,
    von_mises(theta_plot_rad, *params),
    label=f"Von Mises (MSE={mse_von_mises:.2f})"
)
plt.plot(
    theta_plot_deg,
    gaussian(theta_plot_rad, *gaussian_params),
    "--",
    label=f"Gaussian (MSE={mse_gaussian:.2f})"
)

plt.xlabel("Orientation (degrees)")
plt.ylabel("Mean spike count per trial")
plt.title(f"Von Mises vs. Gaussian — Neuron {neuron_index}")
plt.xlim(0, 180)
plt.xticks(np.arange(0, 181, 22.5))
plt.legend()
plt.tight_layout()
plt.savefig("part4_von_mises.png", dpi=300)
plt.show()
