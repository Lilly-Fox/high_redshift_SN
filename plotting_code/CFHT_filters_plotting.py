import numpy as np
import matplotlib.pyplot as plt
import glob
import os

folder = r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\CFHT_megacam_filters_Cooke2013"

def read_file(file_path):
    col1, col2 = [], []
    with open(file_path, 'r') as file:
        for line in file:
            stripped = line.strip()
            parts = stripped.replace(',', ' ').split()
            col1.append(float(parts[0]))
            col2.append(float(parts[1]))
    return col1, col2


all_files = sorted(glob.glob(os.path.join(folder, "*.txt")))
data = []

for f in all_files:
    x, y = read_file(f)
    data.append((os.path.basename(f), x, y))

plt.figure()
for name, x, y in data:
    plt.plot(x, y, label=name)

plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\initial_CFHT_filters.png")
plt.show()
