import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import trapezoid


GALAXY_DIR = Path(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\templates")
LBG_DIR = Path(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\LBG_templates\LBG_templates")
FILTER_DIR = Path(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\CFHT_megacam_filters_Cooke2013")
OUTPUT_DIR = Path(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\take_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


Z_STEP = 0.1
Z_MARKER_INTERVAL = 0.5

Z3_MAX = 3.5
Z4_MAX = 4.5
Z5_MAX = 5.5

LBG_PLOT_Z_MIN = 2.5 # LBG templates are high-z Lyman-break galaxies. Don't do the extrapolation

NORM_MIN = 1300.0
NORM_MAX = 1500.0

LYMAN_LIMIT = 912.0
LYMAN_ALPHA = 1216.0

LBG_REFERENCE_REDSHIFT = 3.0
FOREST_Z_MIN = 1.0
FOREST_Z_MAX = 6.0

C = 2.99792458E18  # Speed of light in Angstrom/s (https://www.convertunits.com/from/speed+of+light+%5bvacuum%5d/to/angstrom/second)

FILTER_NAMES = ["u", "g", "r", "i", "z"]
MIN_FILTER_COVERAGE = 0.99

EXTEND_TO = 10000.0
WAVELENGTH_STEP = 1.0
NOISE_WINDOW = 100.0
RANDOM_SEED = 42

EXCLUDE_ELLIPTICAL_ABOVE_Z2 = False
COLOUR_MIN = -1.0
COLOUR_MAX = 4.0


SELECTIONS = {
    "z~3": {"z_max": Z3_MAX,
        "panels": [
            ("g-r", "u-g", "ugr"),
            ("g-i", "u-g", "ugi"),
            ("r-i", "u-r", "uri"),
            ("r-z", "u-r", "uiz")]},
    "z~4": {
        "z_max": Z4_MAX,
        "panels": [
            ("r-i", "g-r", "gri"),
            ("r-z", "g-r", "grz"),
            ("i-z", "g-i", "giz")]},
    "z~5": {
        "z_max": Z5_MAX,
        "panels": [("i-z", "r-i", "riz")]},
}

EVOLUTION_COLOUR_LIMIT = 15.0
EVOLUTION_COLOURS = ["u-g", "g-r", "g-i", "u-r","r-i", "r-z", "i-z"]



def load_dat_file(filename):
    data = np.loadtxt(filename)
    wavelength = np.asarray(data[:, 0], dtype=float)
    values = np.asarray(data[:, 1], dtype=float)
    return wavelength, values


def load_spectra(directory):
    spectra = {}
    files = sorted(list(directory.glob("*.dat")) + list(directory.glob("*.txt")))
    for filename in files:
        wavelength, flux = load_dat_file(filename)
        spectra[filename.stem] = {"wavelength": wavelength, "flux": flux}
    return spectra


def load_filters(directory):
    filters = {}
    files = sorted(list(directory.glob("*.dat")) + list(directory.glob("*.txt")))
    for filename in files:
        filter_name = filename.stem.replace("CFHTLS_", "").lower()
        wavelength, throughput = load_dat_file(filename)
        filters[filter_name] = {"wavelength": wavelength,"throughput": throughput,}
    return filters

def normalise_spectrum(wavelength, flux, spectrum_name="spectrum"):
    """Normalise by mean f_lambda between 1300 and 1500 Angstrom."""
    mask = ((wavelength >= NORM_MIN) & (wavelength <= NORM_MAX) & np.isfinite(flux))
    norm_wave = wavelength[mask]
    norm_flux = flux[mask]
    width = norm_wave[-1] - norm_wave[0]
    mean_flux = trapezoid(norm_flux, norm_wave) / width
    return flux / mean_flux


def normalise_all_spectra(spectra, family_name):
    for name, spectrum in spectra.items():
        spectrum["flux"] = normalise_spectrum(spectrum["wavelength"], spectrum["flux"], spectrum_name=name)
        wave = spectrum["wavelength"]
        flux = spectrum["flux"]
        finite = np.isfinite(flux)
    return spectra


def apply_lyman_limit(wavelength, flux):
    flux = flux.copy()
    flux[wavelength < LYMAN_LIMIT] = 0.0
    return flux


def lya_forest(redshift):
    if redshift <= FOREST_Z_MIN:
        return 1.0
    if redshift >= FOREST_Z_MAX:
        return 0.0
    x = (redshift - FOREST_Z_MIN) / (FOREST_Z_MAX - FOREST_Z_MIN)
    return 1.0 - x


def lbg_forest_correction(redshift):
    """
    z=1 -> 1.35 (35% increase)
    z=3 -> 1.00 (no change)
    z=6 -> 0.35 (65% decrease)
    """
    return float(np.interp(redshift, [1.0, 3.0, 6.0], [1.35, 1.0, 0.35]))


def apply_lyman_forest(wavelength, flux, redshift):
    flux = flux.copy()
    transmission = lya_forest(redshift)
    mask = ((wavelength >= LYMAN_LIMIT) & (wavelength < LYMAN_ALPHA))
    flux[mask] *= transmission
    return flux


def correct_lbg_forest(wavelength, flux, redshift):
    flux = flux.copy()
    original_correction = lbg_forest_correction(LBG_REFERENCE_REDSHIFT)
    new_correction = lbg_forest_correction(redshift)
    mask = ((wavelength >= LYMAN_LIMIT) & (wavelength < LYMAN_ALPHA))
    flux[mask] *= new_correction / original_correction
    return flux


def apply_high_redshift_absorption(wavelength, flux, redshift):
    flux = flux.copy()
    if redshift >= 6.0:
        flux[wavelength < LYMAN_ALPHA] = 0.0
    return flux


def prepare_spectrum(wavelength, flux, redshift, is_lbg=False):
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
    if not np.isfinite(mean_fnu):
        return np.nan
    if mean_fnu == 0:
        return 0.0
    if mean_fnu < 0:
        return np.nan

    return mean_fnu


def calculate_ab_magnitude(mean_fnu):
    if mean_fnu == 0:
        return np.inf
    if not np.isfinite(mean_fnu) or mean_fnu < 0:
        return np.nan
    return -2.5 * np.log10(mean_fnu)


def calculate_magnitudes(wavelength, flux, filters):
    magnitudes = {}
    for name in FILTER_NAMES:
        filt = filters[name]
        mean_fnu = calculate_filter_flux(wavelength,flux,filt["wavelength"],filt["throughput"])
        magnitudes[name] = calculate_ab_magnitude(mean_fnu)
    return magnitudes



def calculate_colours(magnitudes):
    colours = {
        "u-g": magnitudes["u"] - magnitudes["g"],
        "g-r": magnitudes["g"] - magnitudes["r"],
        "g-i": magnitudes["g"] - magnitudes["i"],
        "u-r": magnitudes["u"] - magnitudes["r"],
        "r-i": magnitudes["r"] - magnitudes["i"],
        "r-z": magnitudes["r"] - magnitudes["z"],
        "i-z": magnitudes["i"] - magnitudes["z"]}
    return colours



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
    rng = np.random.default_rng(RANDOM_SEED)
    for name, spectrum in spectra.items():
        wavelength = spectrum["wavelength"]
        flux = spectrum["flux"]
        wavelength, flux = extend_lbg_blueward(wavelength, flux)
        wavelength, flux = extend_spectrum(wavelength, flux, rng=rng)
        spectrum["wavelength"] = wavelength
        spectrum["flux"] = flux

    return spectra



def calculate_evolution_tracks(spectra, filters, redshifts, is_lbg=False):
    """Calculate colours for every spectrum at each redshift."""
    colour_names = ["u-g", "g-r", "g-i", "u-r", "r-i", "r-z", "i-z"]
    tracks = {}

    for name, spectrum in spectra.items():
        wavelength = spectrum["wavelength"]
        flux = spectrum["flux"]
        track = {"redshift": []}
        for colour_name in colour_names:
            track[colour_name] = []

        for redshift in redshifts:
            obs_wave, obs_flux = prepare_spectrum(wavelength, flux, redshift, is_lbg=is_lbg)
            magnitudes = calculate_magnitudes(obs_wave, obs_flux, filters)
            colours = calculate_colours(magnitudes)
            track["redshift"].append(redshift)

            for colour_name, colour_value in colours.items():
                track[colour_name].append(colour_value)

        for key in track:
            track[key] = np.asarray(track[key], dtype=float)
        tracks[name] = track
    return tracks


def plot_values_for_display(values):
    values = np.asarray(values, dtype=float).copy()
    values[np.isposinf(values)] = COLOUR_MAX
    values[np.isneginf(values)] = COLOUR_MIN
    return values


def marker_indices_for_redshift(redshifts):
    return np.where(
        np.isclose(redshifts / Z_MARKER_INTERVAL, np.round(redshifts / Z_MARKER_INTERVAL), atol=1e-8))[0]


def plot_tracks(ax, tracks, x_colour, y_colour, colour_map, linestyle="-"):
    for name, track in tracks.items():
        redshift = track["redshift"]
        x = plot_values_for_display(track[x_colour])
        y = plot_values_for_display(track[y_colour])
        marker_indices = marker_indices_for_redshift(redshift)
        ax.plot(x,y,color=colour_map[name],label=name.replace("_", " "), marker="o",
            markevery=marker_indices, markersize=3)


def make_template_colours(galaxy_tracks, lbg_tracks):
    galaxy_palette = ["darkred", "red", "blue", "green", "darkgreen", "darkblue", "darkorange", "purple", "brown", "teal"]
    lbg_palette = ["magenta", "cyan", "gold", "deepskyblue", "black"]
    galaxy_colours = {name: galaxy_palette[i % len(galaxy_palette)] for i, name in enumerate(galaxy_tracks)}
    lbg_colours = {name: lbg_palette[i % len(lbg_palette)] for i, name in enumerate(lbg_tracks)}
    return galaxy_colours, lbg_colours


def plot_selection_family(selection_name, selection, galaxy_tracks, lbg_tracks):
    panels = selection["panels"]
    z_max = selection["z_max"]
    n_panels = len(panels)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows,ncols,figsize=(6.2 * ncols, 5.2 * nrows), squeeze=False)
    axes = axes.flatten()

    galaxy_colours, lbg_colours = make_template_colours(galaxy_tracks, lbg_tracks,)
    galaxy_subset = {}

    for name, track in galaxy_tracks.items():
        mask = track["redshift"] <= z_max
        galaxy_subset[name] = {key: values[mask] for key, values in track.items()}

    lbg_subset = {}
    for name, track in lbg_tracks.items():
        mask = ((track["redshift"] >= LBG_PLOT_Z_MIN) & (track["redshift"] <= z_max))
        lbg_subset[name] = {key: values[mask] for key, values in track.items()}

    for ax, (x_colour, y_colour, panel_name) in zip(axes, panels):
        plot_tracks(ax,galaxy_subset,x_colour,y_colour,galaxy_colours,linestyle="-",)
        plot_tracks(ax,lbg_subset,x_colour,y_colour,lbg_colours,linestyle="--",)

        ax.set_xlabel(x_colour)
        ax.set_ylabel(y_colour)
        ax.set_xlim(COLOUR_MIN, COLOUR_MAX)
        ax.set_ylim(COLOUR_MIN, COLOUR_MAX)
        ax.set_title(f"{panel_name}: {y_colour} vs. {x_colour}")
        ax.set_aspect("equal", adjustable="box")

        ax.legend(loc="best", fontsize="x-small", ncol=2)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{selection_name.upper()} LBG colour-selection diagrams "
        f"(galaxies: z=0-{z_max:.1f}; LBG templates: "
        f"z={LBG_PLOT_Z_MIN:.1f}-{z_max:.1f})")
    fig.tight_layout()
    output_file = OUTPUT_DIR / f"{selection_name}_colour_colour.png"
    fig.savefig(output_file, bbox_inches="tight")
    plt.show()


def prepare_evolution_plot_values(values, limit=EVOLUTION_COLOUR_LIMIT):
    values = np.asarray(values, dtype=float).copy()
    values[np.isposinf(values)] = limit
    values[np.isneginf(values)] = -limit
    return values



def plot_colour_evolution(galaxy_tracks,lbg_tracks, colour_names=None, z_max=5.5,filename="colour_evolution.png)"):
    colour_names = EVOLUTION_COLOURS
    n_colours = len(colour_names)
    ncols = 2
    nrows = int(np.ceil(n_colours / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
    axes = axes.flatten()
    galaxy_colours, lbg_colours = make_template_colours(galaxy_tracks,lbg_tracks)
    for ax, colour_name in zip(axes, colour_names):
        for name, track in galaxy_tracks.items():
            redshift = track["redshift"]
            mask = redshift <= z_max
            z = redshift[mask]
            colour = prepare_evolution_plot_values(track[colour_name][mask])
            marker_indices = marker_indices_for_redshift(z)
            ax.plot(z,colour,color=galaxy_colours[name],marker="o",markevery=marker_indices,label=name.replace("_", " "))

        for name, track in lbg_tracks.items():
            redshift = track["redshift"]
            mask = redshift <= z_max
            z = redshift[mask]
            colour = prepare_evolution_plot_values(track[colour_name][mask])
            marker_indices = marker_indices_for_redshift(z)
            ax.plot(z, colour, color=lbg_colours[name], marker="o", markevery=marker_indices, label=name.replace("_", " "))

        ax.set_xlabel("Redshift")
        ax.set_ylabel(colour_name)
        ax.set_xlim(0, z_max)
        ax.set_ylim(-1,15)
        ax.set_xticks(np.arange(0, z_max + 0.01, 0.5))
        ax.legend(loc="best",fontsize="x-small", ncol=2)
        ax.set_title(f"{colour_name} evolution")

    for ax in axes[n_colours:]:
        ax.set_visible(False)

    fig.tight_layout()
    output_file = OUTPUT_DIR / filename
    fig.savefig(output_file,dpi=200,bbox_inches="tight")
    plt.show()


def plot_selection_colour_evolution(galaxy_tracks, lbg_tracks):
    selection_colours = {
        "z3": ["u-g", "g-r", "g-i", "u-r", "r-i", "r-z"],
        "z4": [ "g-r", "r-i", "r-z", "g-i", "i-z"],
        "z5": ["r-i", "i-z"]}

    selection_max_redshifts = {"z3": 3.5, "z4": 4.5, "z5": 5.5}
    for selection_name, colours in selection_colours.items():
        plot_colour_evolution(galaxy_tracks,lbg_tracks, colour_names=colours, z_max=selection_max_redshifts[selection_name], filename=f"{selection_name}_colour_evolution.png")


def main():
    galaxy_spectra = load_spectra(GALAXY_DIR)
    lbg_spectra = load_spectra(LBG_DIR)
    filters = load_filters(FILTER_DIR)
    lbg_spectra = extend_lbg_spectra(lbg_spectra)
    galaxy_redshifts = np.round(np.arange(0.0, Z5_MAX + Z_STEP / 2, Z_STEP),2,)
    lbg_redshifts = np.round(np.arange(LBG_PLOT_Z_MIN, Z5_MAX + Z_STEP / 2, Z_STEP),2)
    galaxy_tracks = calculate_evolution_tracks(galaxy_spectra,filters,galaxy_redshifts,is_lbg=False,)
    lbg_tracks = calculate_evolution_tracks(lbg_spectra,filters,lbg_redshifts,is_lbg=True,)
    for selection_name, selection in SELECTIONS.items():
        plot_selection_family(selection_name,selection,galaxy_tracks,lbg_tracks)
    plot_colour_evolution(galaxy_tracks,lbg_tracks,filename="all_colour_evolution.png")
    plot_selection_colour_evolution(galaxy_tracks,lbg_tracks)


main()
