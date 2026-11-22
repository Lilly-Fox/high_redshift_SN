import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# # Folder containing .dat files
# folder = r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\templates"

# # Find all .dat files
# dat_files = glob.glob(os.path.join(folder, "*.dat"))

# for file in dat_files:
#     data = np.loadtxt(file)
#     x = data[:, 0]
#     y = data[:, 1]

#     plt.plot(x, y, label=os.path.basename(file))

# plt.xlabel("LABEL")
# plt.ylabel("LABEL")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# #plt.savefig(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\galaxy_templates.png")
# plt.show()


#Composite Spectra .txt
# make the wavelength range between ~1300-1500A normalised

filepath = r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\LBG_templates\LBG_templates"
dat_files = glob.glob(os.path.join(filepath, "*.txt"))

for file in dat_files:
    data = np.loadtxt(file)
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, label=os.path.basename(file))


    mask = (x >= 1300) & (x <= 1500)

    # Min–max normalization
    f_min = y[mask].min()
    f_max = y[mask].max()
    flux_norm[mask] = (y[mask] - f_min) / (f_max - f_min)

    #plt.plot(x, flux_norm)

plt.xlabel("Wavelength (Angstroms)")
plt.ylabel("Flux")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\composite_spectra.png")
plt.show()