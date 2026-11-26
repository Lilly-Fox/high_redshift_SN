"""I'm assuming that the spectra are relatively flat after 2000A"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


LBG_DIR = Path(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\LBG_templates\LBG_templates")
OUTPUT_DIR = Path(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\LBG_extension_output")
OUTPUT_DIR.mkdir(exist_ok=True)

EXTEND_TO = 10000.0
WAVELENGTH_STEP = 1.0
NOISE_WINDOW = 100.0 # Wavelength range used for noise
RANDOM_SEED = 42 


def find_spectrum_files(directory):
    return sorted(list(directory.glob("*.dat")) + list(directory.glob("*.txt")))


def load_spectrum(filename):
    data = np.loadtxt(filename)
    wavelength = data[:, 0]
    flux = data[:, 1]
    return wavelength, flux


def extend_spectrum(wavelength, flux, extend_to=EXTEND_TO, wavelength_step=WAVELENGTH_STEP, noise_window=NOISE_WINDOW, rng=None):
    """Extend the spectra using the median level plus random noise."""
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    tail = wavelength >= wavelength[-1] - noise_window
    tail_flux = flux[tail]
    if tail_flux.size < 2:
        tail_flux = flux[-min(10, flux.size):]

    baseline = np.median(tail_flux)
    deviations = tail_flux - baseline
    mad = np.median(np.abs(deviations)) #median of absolute of deviations
    noise_sigma = 1.4826 * mad

    extended_wavelength = np.arange(wavelength[-1], extend_to, wavelength_step)
    random_noise = rng.normal(0.0, noise_sigma, extended_wavelength.size)
    extended_flux = baseline + random_noise

    # # Keep the extrapolated continuum non-negative.
    # extended_flux = np.clip(extended_flux, 0.0, None)

    full_wavelength = np.concatenate((wavelength, extended_wavelength))
    full_flux = np.concatenate((flux, extended_flux))

    return full_wavelength, full_flux, baseline, noise_sigma


def plot_extension(name, original_wavelength, original_flux, extended_wavelength, extended_flux, baseline, noise_sigma):
    fig, ax = plt.subplots()
    ax.plot(original_wavelength, original_flux,color="black", linewidth=1, label="Original LBG spectrum")
    ax.plot(extended_wavelength, extended_flux,color="blue", linewidth=0.5, label="Extrapolated spectrum")
    ax.axvline(original_wavelength[-1], color="tab:red",linestyle="--", label="End of original spectrum")
    ax.set_xlabel("Rest-frame wavelength (A)")
    ax.set_ylabel("Flux density")
    ax.set_xlim(original_wavelength[0], extended_wavelength[-1])
    ax.legend()
    fig.tight_layout()
    output_file = OUTPUT_DIR / f"{name}_extended.png"
    fig.savefig(output_file)
    plt.show()
    plt.close(fig)


def extend_lbg_spectra(spectra):
    rng = np.random.default_rng(RANDOM_SEED)

    for name, spectrum in spectra.items():
        wavelength = spectrum["wavelength"]
        flux = spectrum["flux"]

        extended_wavelength, extended_flux = extend_spectrum(
            wavelength, flux, rng=rng
        )

        spectrum["wavelength"] = extended_wavelength
        spectrum["flux"] = extended_flux

        print(
            f"Extended {name}: "
            f"{wavelength[-1]:.0f} to "
            f"{extended_wavelength[-1]:.0f} Angstrom"
        )

    return spectra

def main():
    files = find_spectrum_files(LBG_DIR)
    rng = np.random.default_rng(RANDOM_SEED)
    for filename in files:
        wavelength, flux = load_spectrum(filename)
        full_wavelength, full_flux, baseline, noise_sigma = extend_spectrum(wavelength, flux, rng=rng)
        plot_extension(filename.stem,wavelength,flux,full_wavelength,full_flux,baseline,noise_sigma)


main()
