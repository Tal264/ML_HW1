from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Find the dataset file
files = sorted(Path(".").rglob("105l001p16_spikes.mat"))
if not files:
    raise FileNotFoundError("Could not find 105l001p16_spikes.mat")

with h5py.File(files[0], "r") as f:
    stim = f["stim"][:].flatten()
    spikes = f[f["spikes"][0, 0]]

    n_trials, n_time, n_neurons = spikes.shape

    # Total spike count per trial and neuron.
    # h5py Dataset objects must be sliced into NumPy arrays before summing.
    # Process trials in batches to avoid loading the full dataset into memory.
    responses = np.zeros((n_trials, n_neurons), dtype=np.float64)
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

print("Number of V1 neurons:", n_neurons)
print("Number of trials:", n_trials)

print("\nTrials per stimulus:")
for label in labels:
    print(f"Label {int(label)}: {np.sum(stim == label)}")

print("\nMean responses shape:", mean_responses.shape)

# Plot the first six neurons
plt.figure(figsize=(10, 6))
for i in range(min(6, n_neurons)):
    plt.plot(labels, mean_responses[:, i], "o-", label=f"Neuron {i}")

plt.xlabel("Stimulus label")
plt.ylabel("Mean spike count per trial")
plt.title("V1 Orientation Tuning Curves")
plt.xticks(labels)
plt.legend()
plt.tight_layout()
plt.savefig("part1_tuning.png", dpi=300)
plt.show()

# Selected neuron used in all subsequent parts
neuron_index = 5
neuron_data = mean_responses[:, neuron_index]

print("\nSelected neuron index:", neuron_index)
print("Selected neuron's mean responses:", neuron_data)
