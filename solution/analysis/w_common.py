import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hist import Hist
from hist.plot import histplot

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent

SAMPLE_DIRECTORY = PROJECT_DIRECTORY/"runs"
FIGURE_DIRECTORY = PROJECT_DIRECTORY/"figures"
FIGURE_DIRECTORY.mkdir(parents = True, exist_ok = True)

SAMPLE_FILE = SAMPLE_DIRECTORY/"ppTOw_14TeV.hepmc"

CENTER_OF_MASS_ENERGY = 14000
W_MASS_PDG = 80.4

TOP_PID = 6
ELECTRON_PID = 11
MUON_PID = 13
TAU_PID = 15
W_PID = 24

# The charged leptons a W can decay into
CHARGED_LEPTON_PIDS = {ELECTRON_PID, MUON_PID, TAU_PID}

CHILD_LABELS = {
    # Quarks
    1: r"$d/\bar{d}$",
    2: r"$u/\bar{u}$",
    3: r"$s/\bar{s}$",
    4: r"$c/\bar{c}$",
    5: r"$b/\bar{b}$",
    6: r"$t/\bar{t}$",
    # Charged leptons
    11: r"$e^\pm$",
    13: r"$\mu^\pm$",
    15: r"$\tau^\pm$",
    # Neutrinos
    12: r"$\nu_e/\bar{\nu}_e$",
    14: r"$\nu_\mu/\bar{\nu}_\mu$",
    16: r"$\nu_\tau/\bar{\nu}_\tau$",
    # Gauge bosons
    21: r"$g$",
    22: r"$\gamma$",
}

PARTICLE_NAMES = {
    # Quarks
    1: "d", -1: "dbar",
    2: "u", -2: "ubar",
    3: "s", -3: "sbar",
    4: "c", -4: "cbar",
    5: "b", -5: "bbar",
    6: "t", -6: "tbar",
    # Charged leptons
    11: "e-", -11: "e+",
    13: "mu-", -13: "mu+",
    15: "tau-", -15: "tau+",
    # Neutrinos
    12: "nu_e", -12: "nu_ebar",
    14: "nu_mu", -14: "nu_mubar",
    16: "nu_tau", -16: "nu_taubar",
    # Gauge bosons
    21: "g",
    22: "gamma",
    24: "W+", -24: "W-",
    # Incoming beams
    2212: "p+",
}


# Name a particle from its PDG ID
def particle_name(pid):
    return PARTICLE_NAMES.get(pid, str(pid))


# Name a particle and antiparticle species from its PDG ID
def particle_species_name(pid):
    particle = PARTICLE_NAMES.get(pid)
    antiparticle = PARTICLE_NAMES.get(-pid)

    if particle is None:
        return str(pid)

    if antiparticle is None:
        return particle

    return f"{particle}/{antiparticle}"


# Check whether a W boson is the last copy
def is_last_copy(particle):
    return not any(
        child.pid == particle.pid
        for child in particle.children
    )


# Find the last copies of the W bosons in an event
def find_last_ws(event):
    return [
        particle
        for particle in event.particles
        if abs(particle.pid) == W_PID and is_last_copy(particle)
    ]


# Select the values for a specific W-boson PID
def select_by_pid(pids, values, target_pid):
    return [
        value
        for pid, value in zip(pids, values)
        if pid == target_pid
    ]


# Select the values for a specific W-boson PID within a pT range
def select_by_pid_and_pt(pids, pt_values, values, target_pid, pt_minimum, pt_maximum):
    return [
        value
        for pid, pt_value, value in zip(pids, pt_values, values)
        if pid == target_pid and pt_minimum <= pt_value < pt_maximum
    ]


# Select the values for a specific W-boson PID within a pT range and rapidity window
def select_by_pid_pt_and_rapidity(
    pids,
    pt_values,
    rapidity_values,
    values,
    target_pid,
    pt_minimum,
    pt_maximum,
    rapidity_maximum
):
    return [
        value
        for pid, pt_value, rapidity_value, value in zip(pids, pt_values, rapidity_values, values)
        if pid == target_pid
        and pt_minimum <= pt_value < pt_maximum
        and (rapidity_maximum is None or abs(rapidity_value) < rapidity_maximum)
    ]


# Count the values outside the plotting range
def count_outside(values, minimum, maximum):
    return sum(
        not math.isfinite(value)
        or value < minimum
        or value >= maximum
        for value in values
    )


# Print one row of an entries-outside-range table
def print_outside_count(label, values, minimum, maximum):
    print(f"{label:<24}: {count_outside(values, minimum, maximum)}")


# Construct an unweighted histogram
def make_histogram(values, bins, minimum, maximum):
    histogram = Hist.new.Regular(
        bins,
        minimum,
        maximum
    ).Int64()

    finite_values = [
        value
        for value in values
        if math.isfinite(value)
    ]

    histogram.fill(finite_values)

    return histogram


# Find the most populated histogram bin
def find_peak_bin(values, bins, minimum, maximum):
    histogram = make_histogram(
        values,
        bins,
        minimum,
        maximum
    )

    counts = histogram.values()
    edges = histogram.axes[0].edges

    peak_bin = max(
        range(len(counts)),
        key = lambda index: counts[index]
    )

    return (
        edges[peak_bin],
        edges[peak_bin + 1],
        counts[peak_bin]
    )


# Print the summaries of a transverse-momentum distribution
def print_pt_summary(label, values, bins, minimum, maximum):
    peak_minimum, peak_maximum, peak_count = find_peak_bin(
        values,
        bins,
        minimum,
        maximum
    )

    peak_range = f"{peak_minimum:.1f}-{peak_maximum:.1f}"

    print(label)
    print(f"{'Peak bin':<24}: {peak_range} GeV")
    print(f"{'Entries':<24}: {peak_count}")
    print(f"{'Mean pT':<24}: {statistics.mean(values):.4f} GeV")
    print(f"{'Median pT':<24}: {statistics.median(values):.4f} GeV")


# Format a value and its error into one right-aligned column
def format_with_error(value, error, width = 18):
    text = f"{value:.4f} +- {error:.4f}"

    return f"{text:>{width}}"


# Draw one histogram
def draw_histogram(
    values,
    bins,
    minimum,
    maximum,
    xlabel,
    ylabel,
    filename,
    density = False,
    log_scale = False
):
    histogram = make_histogram(
        values,
        bins,
        minimum,
        maximum
    )

    fig, ax = plt.subplots(figsize = (8, 6))

    histplot(
        histogram,
        ax = ax,
        histtype = "barstep",
        yerr = True,
        density = density,
        linewidth = 1.5
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.tick_params(direction = "in", top = True, right = True)

    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY/filename, bbox_inches = "tight")
    plt.close(fig)


# Draw two overlaid histograms
def draw_histogram_pair(
    values_1,
    values_2,
    label_1,
    label_2,
    bins,
    minimum,
    maximum,
    xlabel,
    ylabel,
    filename,
    density = False,
    log_scale = False
):
    histogram_1 = make_histogram(
        values_1,
        bins,
        minimum,
        maximum
    )

    histogram_2 = make_histogram(
        values_2,
        bins,
        minimum,
        maximum
    )

    fig, ax = plt.subplots(figsize = (8, 6))

    histplot(
        histogram_1,
        ax = ax,
        histtype = "barstep",
        yerr = True,
        density = density,
        linewidth = 1.5,
        label = label_1
    )

    histplot(
        histogram_2,
        ax = ax,
        histtype = "barstep",
        yerr = True,
        density = density,
        linewidth = 1.5,
        linestyle = "--",
        label = label_2
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon = False)
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.tick_params(direction = "in", top = True, right = True)

    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY/filename, bbox_inches = "tight")
    plt.close(fig)


# Sum the four-momenta of a set of particles
def sum_four_momenta(particles):
    energy = sum(
        particle.momentum.e
        for particle in particles
    )
    px = sum(
        particle.momentum.px
        for particle in particles
    )
    py = sum(
        particle.momentum.py
        for particle in particles
    )
    pz = sum(
        particle.momentum.pz
        for particle in particles
    )

    return energy, px, py, pz


# Calculate the invariant mass from the four-momentum components
def calculate_invariant_mass(energy, px, py, pz):
    mass_squared = energy**2 - (px**2 + py**2 + pz**2)

    return math.sqrt(max(0, mass_squared))


# Find a direct charged-lepton and neutrino pair from a W decay
def find_lepton_neutrino_pair(w):
    for lepton in w.children:
        if abs(lepton.pid) not in CHARGED_LEPTON_PIDS:
            continue

        for neutrino in w.children:
            # The neutrino PID is the charged-lepton PID plus one
            if abs(neutrino.pid) != abs(lepton.pid) + 1:
                continue

            if lepton.pid*w.pid >= 0:
                continue

            if neutrino.pid*w.pid <= 0:
                continue

            return lepton, neutrino

    return None, None


# Boost the charged lepton's momentum into the W rest frame
def boost_to_w_frame_cos_theta_star(w, lepton):
    w_momentum = w.momentum
    lepton_momentum = lepton.momentum

    beta_x = w_momentum.px/w_momentum.e
    beta_y = w_momentum.py/w_momentum.e
    beta_z = w_momentum.pz/w_momentum.e
    beta_squared = beta_x**2 + beta_y**2 + beta_z**2

    if beta_squared <= 0:
        boosted_px = lepton_momentum.px
        boosted_py = lepton_momentum.py
        boosted_pz = lepton_momentum.pz
    else:
        beta_magnitude = math.sqrt(beta_squared)
        gamma = 1/math.sqrt(1 - beta_squared)

        beta_hat_x = beta_x/beta_magnitude
        beta_hat_y = beta_y/beta_magnitude
        beta_hat_z = beta_z/beta_magnitude

        parallel_component = (
            lepton_momentum.px*beta_hat_x
            + lepton_momentum.py*beta_hat_y
            + lepton_momentum.pz*beta_hat_z
        )

        perpendicular_px = lepton_momentum.px - parallel_component*beta_hat_x
        perpendicular_py = lepton_momentum.py - parallel_component*beta_hat_y
        perpendicular_pz = lepton_momentum.pz - parallel_component*beta_hat_z

        boosted_parallel = gamma*(parallel_component - beta_magnitude*lepton_momentum.e)

        boosted_px = boosted_parallel*beta_hat_x + perpendicular_px
        boosted_py = boosted_parallel*beta_hat_y + perpendicular_py
        boosted_pz = boosted_parallel*beta_hat_z + perpendicular_pz

    w_momentum_magnitude = math.sqrt(w_momentum.px**2 + w_momentum.py**2 + w_momentum.pz**2)
    boosted_momentum_magnitude = math.sqrt(boosted_px**2 + boosted_py**2 + boosted_pz**2)

    if w_momentum_magnitude == 0 or boosted_momentum_magnitude == 0:
        return None

    cos_theta_star = (
        boosted_px*w_momentum.px
        + boosted_py*w_momentum.py
        + boosted_pz*w_momentum.pz
    )/(boosted_momentum_magnitude*w_momentum_magnitude)

    return cos_theta_star
