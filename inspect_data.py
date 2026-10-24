from pathlib import Path
import h5py
import numpy as np

mat_files = sorted(Path(".").rglob("*.mat"))
file_path = mat_files[0]

with h5py.File(file_path, "r") as data:
    spikes = data["spikes"]

    for index, ref in enumerate(spikes[:].flatten()):
        arr = data[ref]

        # Read all values for the first trial only
        trial = arr[0, :, :]

        print(f"\nArray {index + 1}")
        print("Trial shape:", trial.shape)
        print("Minimum:", trial.min())
        print("Maximum:", trial.max())
        print("Number of non-zero values:", np.count_nonzero(trial))
        print("Total number of values:", trial.size)

        # Count non-zero values along the second dimension
        counts_per_row = np.count_nonzero(trial, axis=1)
        active_rows = np.flatnonzero(counts_per_row)

        print("Number of rows with non-zero values:", len(active_rows))
        print("First 10 rows with non-zero values:", active_rows[:10])
