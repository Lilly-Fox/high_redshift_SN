
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.integrate import trapezoid

GALAXY_DIR = Path(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\templates")
LBG_DIR = Path(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\LBG_templates\LBG_templates")
FILTER_DIR = Path(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\CFHT_megacam_filters_Cooke2013")
OUTPUT_DIR = Path(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\color-color_output")
OUTPUT_DIR.mkdir(exist_ok=True)

LBG_Z_MIN = 3.0
LBG_Z_MAX = 5.5 #normal is 3.5, mid is 5.0, high is 7.0, extra high is 12
Z_STEP = 0.05

REDSHIFTS = np.arange(LBG_Z_MIN, LBG_Z_MAX, Z_STEP)
NORM_MIN = 1300.0 #Angstrom
NORM_MAX = 1500.0 #Angstrom
LYMAN_LIMIT = 912.0
LYMAN_ALPHA = 1216.0
LBG_REFERENCE_REDSHIFT = 3.0
C = 2.99792458E18  # Speed of light in Angstrom/s (https://www.convertunits.com/from/speed+of+light+%5bvacuum%5d/to/angstrom/second)
FOREST_Z_MIN = 1.0
FOREST_Z_MAX = 6.0

FILTER_NAMES = ["u", "g", "r", "i", "z"]

MIN_FILTER_COVERAGE = 0.99 #99% must be covered
EXTEND_TO = 10000.0
WAVELENGTH_STEP = 1.0
NOISE_WINDOW = 100.0
RANDOM_SEED = 42


def load_dat_file(filename):
    data = np.loadtxt(filename)
    wavelength = data[:, 0]
    values = data[:, 1]
    return wavelength, values


def load_spectra(directory):
    spectra = {}
    files = sorted(list(directory.glob("*.dat")) + list(directory.glob("*.txt")))
    for filename in files:
        wavelength, flux = load_dat_file(filename)
        spectra[filename.stem] = {"wavelength": wavelength,"flux": flux}
    return spectra


def load_filters(directory):
    filters = {}
    files = sorted(list(directory.glob("*.dat")) + list(directory.glob("*.txt")))
    for file in files:
        filter_name = file.stem.replace("CFHTLS_", "")
        wavelength, throughput = load_dat_file(file)
        filters[filter_name] = {"wavelength": wavelength, "throughput": throughput}
    return filters


def normalise_spectrum(wavelength, flux):
    """Normalise using mean flux density between 1300 and 1500 Angstrom."""
    mask = ((wavelength >= NORM_MIN) &(wavelength <= NORM_MAX))
    mean_flux = trapezoid(flux[mask], wavelength[mask]) / (wavelength[mask][-1] - wavelength[mask][0])
    return flux / mean_flux


def normalise_all_spectra(spectra):
    for name, spectrum in spectra.items():
        spectrum["flux"] = normalise_spectrum(spectrum["wavelength"],spectrum["flux"])
    return spectra


def apply_lyman_limit(wavelength, flux):
    flux = flux.copy()
    flux[wavelength < LYMAN_LIMIT] = 0.0
    return flux


def lya_forest(redshift):
    """The flux decrement is about zero (i.e., nearly no effect) at z ~ 1 
    and slowly increases in attenuating the flux up to z ~ 6, where it is nearly 100% attenuated
    Transmission is 1 at z <= 1 and zero at z >= 6."""
    if redshift <= FOREST_Z_MIN:
        return 1.0
    if redshift >= FOREST_Z_MAX:
        return 0.0
    x = ((redshift - FOREST_Z_MIN) /(FOREST_Z_MAX - FOREST_Z_MIN))
    return 1.0 - x**2


def apply_lyman_forest(wavelength, flux, redshift):
    """Between 912 and 1216 Angstrom"""
    flux = flux.copy()
    transmission = lya_forest(redshift)
    mask = ((wavelength >= LYMAN_LIMIT) & (wavelength < LYMAN_ALPHA))
    flux[mask] = flux[mask] * transmission
    return flux


def correct_lbg_forest(wavelength, flux, redshift):
    """Correct the existing forest attenuation, already assumed to include the forest attenuation at z~3."""
    flux = flux.copy()
    original_transmission = (LBG_REFERENCE_REDSHIFT)
    new_transmission = lya_forest(redshift)
    mask = ((wavelength >= LYMAN_LIMIT) & (wavelength < LYMAN_ALPHA))
    if original_transmission != 0: #To avoid the division by z
        flux[mask] = flux[mask] * (new_transmission / original_transmission)
    else:
        flux[mask] = 0.0
    return flux


def apply_high_redshift_absorption(wavelength, flux, redshift):
    """For z >= 6, set all flux below Lyman-alpha to zero."""
    flux = flux.copy()
    if redshift >= 6.0:
        flux[wavelength < LYMAN_ALPHA] = 0.0
    return flux


def prepare_spectrum(wavelength,flux,redshift,is_lbg=False):
    """Transform rest-frame spectruminto the observed frame."""
    flux = apply_lyman_limit(wavelength, flux)
    if is_lbg:
        flux = correct_lbg_forest(wavelength, flux, redshift)
    else:
        flux = apply_lyman_forest(wavelength, flux, redshift)

    flux = apply_high_redshift_absorption(wavelength, flux, redshift)
    observed_wavelength = wavelength * (1.0 + redshift)
    observed_flux = flux / (1.0 + redshift)
    return observed_wavelength, observed_flux


def calculate_filter_flux(spectrum_wavelength,spectrum_flux,filter_wavelength,filter_throughput):
    """Calculate the mean f_nu (flux density per frequency) through a filter """
    total_response = trapezoid(filter_throughput / filter_wavelength,filter_wavelength)
    covered = ((filter_wavelength >= spectrum_wavelength[0]) & (filter_wavelength <= spectrum_wavelength[-1]))
    covered_wavelength = filter_wavelength[covered]
    throughput = filter_throughput[covered]
    flux = np.interp(covered_wavelength,spectrum_wavelength,spectrum_flux)
    covered_response = trapezoid(throughput / covered_wavelength,covered_wavelength)
    coverage_fraction = covered_response / total_response
    if coverage_fraction < MIN_FILTER_COVERAGE:
        return np.nan

    numerator = trapezoid(flux * throughput * covered_wavelength, covered_wavelength)
    mean_fnu = numerator / (C * covered_response)
    if not np.isfinite(mean_fnu) or mean_fnu <= 0:
        return np.nan
    
    return mean_fnu


def calculate_ab_magnitude(mean_fnu):
    return -2.5 * np.log10(mean_fnu)


def calculate_magnitudes(wavelength, flux, filters):
    magnitudes = {}
    for name in FILTER_NAMES:
        filt = filters[name]
        mean_fnu = calculate_filter_flux(wavelength,flux,filt["wavelength"],filt["throughput"])
        magnitudes[name] = calculate_ab_magnitude(mean_fnu)
    return magnitudes


def calculate_colours(magnitudes):
    colours = {}
    colours["u-g"] = (magnitudes["u"] - magnitudes["g"])
    colours["g-r"] = (magnitudes["g"] - magnitudes["r"])
    colours["g-i"] = (magnitudes["g"] - magnitudes["i"])
    colours["r-i"] = (magnitudes["r"] - magnitudes["i"])
    return colours

def calculate_evolution_tracks(spectra,filters,redshifts,is_lbg=False):
    """Moves through the arange of redshifts for each galaxy spectrum."""
    tracks = {}

    for name, spectrum in spectra.items():
        wavelength = spectrum["wavelength"]
        flux = spectrum["flux"]
        track = {"redshift": [], "u-g": [], "g-r": [], "g-i": [], "r-i": []}

        for redshift in redshifts:
            obs_wave, obs_flux = prepare_spectrum(wavelength, flux, redshift, is_lbg = is_lbg)
            magnitudes = calculate_magnitudes(obs_wave,obs_flux,filters)
            colours = calculate_colours(magnitudes)
            track["redshift"].append(redshift)
            for colour_name, colour_value in colours.items():
                track[colour_name].append(colour_value)

        for key in track:
            track[key] = np.asarray(track[key])

        tracks[name] = track
    return tracks



def plot_rest_frame_spectra(galaxy_spectra, lbg_spectra):
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, len(galaxy_spectra) + len(lbg_spectra)))

    i = 0
    for name, spectrum in galaxy_spectra.items():
        wave = spectrum["wavelength"]
        flux = apply_lyman_limit(wave, spectrum["flux"])
        ax.plot(wave, flux,color=colors[i], label=name)
        i += 1

    for name, spectrum in lbg_spectra.items():
        wave = spectrum["wavelength"]
        flux = apply_lyman_limit(wave, spectrum["flux"])
        ax.plot(wave, flux, color=colors[i], label=name)
        i += 1

    ax.axvline(LYMAN_LIMIT,color="black",linestyle=":", label="Lyman limit (912A)")
    ax.axvline(LYMAN_ALPHA,color="black",linestyle=":",label="Lyman-alpha (1216A)")
    ax.axvspan(NORM_MIN,NORM_MAX,color="grey",alpha=0.1,label="Normalized interval")

    ax.set_xlabel("Rest-frame wavelength (A)")
    ax.set_ylabel("Normalised flux density")
    ax.set_xlim(0, 10000)
    ax.set_ylim()
    ax.legend(loc="upper right",ncol=2, fontsize='small')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "high3z5._normalised_spectra.png")
    plt.show()


def plot_tracks(ax,tracks,x_colour,y_colour,colour_map):
    for name, track in tracks.items():
        x = track[x_colour]
        y = track[y_colour]
        ax.plot(x,y,color=colour_map[name],label=name.replace("_", " "))


def plot_colour_colour(galaxy_tracks,lbg_tracks):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.set_box_aspect(1)
    galaxy_palette = ["darkred", "red", "blue", "green", "darkgreen","darkblue","darkorange","purple","brown","teal"]
    galaxy_colours = {name: galaxy_palette[i % len(galaxy_palette)]
        for i, name in enumerate(galaxy_tracks)}
    
    lbg_palette = ["magenta","blue","cyan","gold"]
    lbg_colours = {name: lbg_palette[i % len(lbg_palette)]
        for i, name in enumerate(lbg_tracks)}

    panels = [{"x": "g-r", "y": "u-g", "number": 1}, {"x": "g-i", "y": "u-g", "number": 2}, {"x": "r-i", "y": "g-r", "number": 3}]

    for ax, panel in zip(axes, panels):
        plot_tracks(ax,galaxy_tracks,panel["x"],panel["y"],galaxy_colours)
        plot_tracks(ax,lbg_tracks,panel["x"],panel["y"],lbg_colours)        
        ax.set_xlabel(panel["x"])
        ax.set_ylabel(panel["y"])
        ax.legend(loc='best', fontsize='small', ncol=2)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "high3z5.5_colour_colour.png")
    plt.show()


def plot_colour_evolution(galaxy_tracks, lbg_tracks):
    colour_names = ["u-g", "g-r", "g-i", "r-i"]
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    for ax, colour_name in zip(axes, colour_names):
        ax.set_box_aspect(1)
        for name, track in galaxy_tracks.items():
            ax.plot(track["redshift"], track[colour_name], label=name)

        for name, track in lbg_tracks.items():
            ax.plot(track["redshift"], track[colour_name], linestyle="--",label=name)

        ax.set_xlabel("Redshift")
        ax.set_ylabel(colour_name)
        ax.legend(loc="best", fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "high3z5.5_colour_evolution.png", bbox_inches="tight")
    plt.show()


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
    mad = np.median(np.abs(deviations)) #median of absolute of deviations (https://en.wikipedia.org/wiki/Median_absolute_deviation)
    noise_sigma = 1.4826 * mad
    extended_wavelength = np.arange(wavelength[-1], extend_to, wavelength_step)
    random_noise = rng.normal(0.0, noise_sigma, extended_wavelength.size)
    extended_flux = baseline + random_noise
    full_wavelength = np.concatenate((wavelength, extended_wavelength))
    full_flux = np.concatenate((flux, extended_flux))
    return full_wavelength, full_flux


def extend_lbg_blueward(wavelength, flux, blue_limit=0.0):
    step = np.median(np.diff(wavelength))
    new_wavelength = np.arange(blue_limit,wavelength[0],step)
    new_flux = np.zeros_like(new_wavelength)
    wavelength_extended = np.concatenate((new_wavelength, wavelength))
    flux_extended = np.concatenate((new_flux, flux))
    return wavelength_extended, flux_extended


def extend_lbg_spectra(spectra):
    """Extend all LBG spectra in both directions."""
    rng = np.random.default_rng(RANDOM_SEED)
    for name, spectrum in spectra.items():
        wavelength = spectrum["wavelength"]
        flux = spectrum["flux"]
        wavelength, flux = extend_lbg_blueward(wavelength, flux)
        wavelength, flux = extend_spectrum(wavelength, flux, rng=rng)

        spectrum["wavelength"] = wavelength
        spectrum["flux"] = flux

    return spectra


def main():
    galaxy_spectra = load_spectra(GALAXY_DIR)
    lbg_spectra = load_spectra(LBG_DIR)
    filters = load_filters(FILTER_DIR)
    galaxy_spectra = normalise_all_spectra(galaxy_spectra)
    lbg_spectra = normalise_all_spectra(lbg_spectra)
    lbg_spectra = extend_lbg_spectra(lbg_spectra)
    plot_rest_frame_spectra(galaxy_spectra,lbg_spectra)
    galaxy_tracks = calculate_evolution_tracks(galaxy_spectra, filters, REDSHIFTS, is_lbg=False)
    lbg_tracks = calculate_evolution_tracks(lbg_spectra,filters,REDSHIFTS,is_lbg=True)
    plot_colour_colour(galaxy_tracks,lbg_tracks)
    plot_colour_evolution(galaxy_tracks,lbg_tracks)


main()

# def diagnose_track_coverage(tracks):
#     colour_names = ["u-g", "g-r", "g-i", "r-i"]
#     for name, track in tracks.items():
#         redshift = track["redshift"]
#         for colour in colour_names:
#             values = track[colour]
#             valid = np.isfinite(values)
#             z_valid = redshift[valid]
#             print(f"{colour}:z = {z_valid.min():.2f} to {z_valid.max():.2f}")

#    diagnose_track_coverage(lbg_tracks)
