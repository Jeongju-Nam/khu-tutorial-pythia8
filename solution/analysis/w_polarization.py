import itertools
import math
import statistics
from collections import namedtuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyhepmc

from w_common import (
    FIGURE_DIRECTORY,
    SAMPLE_FILE,
    boost_to_w_frame_cos_theta_star,
    draw_histogram_pair,
    find_last_ws,
    find_lepton_neutrino_pair,
    format_with_error,
    print_outside_count,
    select_by_pid,
    select_by_pid_and_pt,
    select_by_pid_pt_and_rapidity
)

NUMBER_OF_POLARIZATION_EVENTS = 10000

PT_BIN_EDGES = [0, 50, 100, 150, 200]

RAPIDITY_CUTS = [None, 3, 2]

PolarizationResult = namedtuple(
    "PolarizationResult",
    [
        "fl",
        "fr",
        "f0",
        "fl_error",
        "fr_error",
        "f0_error",
        "a4",
        "a4_error",
        "entries"
    ]
)


# Compute the polarization fractions of one cos(theta*) sample, with the
# statistical errors propagated from the sample moments
def polarization_fractions(cos_theta_star_sample, charge_sign):
    entries = len(cos_theta_star_sample)

    moment_1 = statistics.mean(cos_theta_star_sample)
    moment_2 = statistics.mean(
        value**2
        for value in cos_theta_star_sample
    )
    moment_3 = statistics.mean(
        value**3
        for value in cos_theta_star_sample
    )
    moment_4 = statistics.mean(
        value**4
        for value in cos_theta_star_sample
    )

    variance_1 = (moment_2 - moment_1**2)/entries
    variance_2 = (moment_4 - moment_2**2)/entries
    covariance = (moment_3 - moment_1*moment_2)/entries

    fl = -0.5 - charge_sign*moment_1 + 2.5*moment_2
    fr = -0.5 + charge_sign*moment_1 + 2.5*moment_2
    f0 = 2 - 5*moment_2

    fl_error = math.sqrt(
        max(0, variance_1 + 6.25*variance_2 - 5*charge_sign*covariance)
    )
    fr_error = math.sqrt(
        max(0, variance_1 + 6.25*variance_2 + 5*charge_sign*covariance)
    )
    f0_error = 5*math.sqrt(max(0, variance_2))

    a4 = 4*moment_1
    a4_error = 4*math.sqrt(max(0, variance_1))

    return PolarizationResult(
        fl,
        fr,
        f0,
        fl_error,
        fr_error,
        f0_error,
        a4,
        a4_error,
        entries
    )


def w_polarization():
    # 1. Polarization observables
    w_pid = []
    w_pt = []
    w_rapidity = []
    cos_theta_star_values = []
    lepton_pt = []
    neutrino_pt = []

    with pyhepmc.open(SAMPLE_FILE) as events:
        for event in itertools.islice(events, NUMBER_OF_POLARIZATION_EVENTS):
            for w in find_last_ws(event):
                # Leptonic decays only, since the charged lepton is the analyser
                lepton, neutrino = find_lepton_neutrino_pair(w)

                if lepton is None:
                    continue

                # Charged-lepton angle in the W rest frame
                cos_theta_star = boost_to_w_frame_cos_theta_star(w, lepton)

                if cos_theta_star is None:
                    continue

                # Record what the moment method needs
                momentum = w.momentum

                w_pid.append(w.pid)
                w_pt.append(momentum.pt())
                w_rapidity.append(momentum.rap())
                cos_theta_star_values.append(cos_theta_star)
                lepton_pt.append(lepton.momentum.pt())
                neutrino_pt.append(neutrino.momentum.pt())

    print()
    print("Polarization sample")
    print(f"{'Events read':<24}: {NUMBER_OF_POLARIZATION_EVENTS}")
    print(f"{'Leptonic W decays':<24}: {len(w_pid)}")
    print(f"{'W+ decays':<24}: {sum(pid == 24 for pid in w_pid)}")
    print(f"{'W- decays':<24}: {sum(pid == -24 for pid in w_pid)}")

    # 2. Charged-lepton and neutrino pT by W charge
    lepton_pt_w_plus = select_by_pid(w_pid, lepton_pt, 24)
    lepton_pt_w_minus = select_by_pid(w_pid, lepton_pt, -24)
    neutrino_pt_w_plus = select_by_pid(w_pid, neutrino_pt, 24)
    neutrino_pt_w_minus = select_by_pid(w_pid, neutrino_pt, -24)

    lepton_pt_minimum = 0
    lepton_pt_maximum = 150

    print()
    print("Charged-lepton and neutrino entries outside plot range")
    print_outside_count("Charged lepton pT", lepton_pt, lepton_pt_minimum, lepton_pt_maximum)
    print_outside_count("Neutrino pT", neutrino_pt, lepton_pt_minimum, lepton_pt_maximum)

    print()
    print("Charged-lepton and neutrino pT by W charge")
    print(
        f"{'Charge':>8}"
        f"{'Entries':>10}"
        f"{'Mean lepton pT [GeV]':>24}"
        f"{'Mean neutrino pT [GeV]':>26}"
    )

    for charge_label, charge_lepton_pt, charge_neutrino_pt in (
        ("W+", lepton_pt_w_plus, neutrino_pt_w_plus),
        ("W-", lepton_pt_w_minus, neutrino_pt_w_minus)
    ):
        print(
            f"{charge_label:>8}"
            f"{len(charge_lepton_pt):>10d}"
            f"{statistics.mean(charge_lepton_pt):>24.4f}"
            f"{statistics.mean(charge_neutrino_pt):>26.4f}"
        )

    draw_histogram_pair(
        lepton_pt_w_plus,
        neutrino_pt_w_plus,
        r"$\ell^+$",
        r"$\nu_\ell$",
        50,
        lepton_pt_minimum,
        lepton_pt_maximum,
        r"$p_T$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "w_lepton_neutrino_pt_w_plus.png",
        density = True
    )
    draw_histogram_pair(
        lepton_pt_w_minus,
        neutrino_pt_w_minus,
        r"$\ell^-$",
        r"$\bar{\nu}_\ell$",
        50,
        lepton_pt_minimum,
        lepton_pt_maximum,
        r"$p_T$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "w_lepton_neutrino_pt_w_minus.png",
        density = True
    )

    # 3. Integrated polarization fractions
    # 3-1. cos(theta*) by W charge
    cos_theta_star_w_plus = select_by_pid(
        w_pid,
        cos_theta_star_values,
        24
    )
    cos_theta_star_w_minus = select_by_pid(
        w_pid,
        cos_theta_star_values,
        -24
    )

    # 3-2. fL, fR, f0 from the first and second moments of cos(theta*)
    result_plus = polarization_fractions(cos_theta_star_w_plus, 1)
    result_minus = polarization_fractions(cos_theta_star_w_minus, -1)

    print()
    print("W polarization fractions (moment method, statistical errors only)")
    print(
        f"{'Charge':>8}"
        f"{'Entries':>10}"
        f"{'fL':>18}"
        f"{'fR':>18}"
        f"{'f0':>18}"
    )

    for charge_label, result in (("W+", result_plus), ("W-", result_minus)):
        print(
            f"{charge_label:>8}"
            f"{result.entries:>10d}"
            f"{format_with_error(result.fl, result.fl_error)}"
            f"{format_with_error(result.fr, result.fr_error)}"
            f"{format_with_error(result.f0, result.f0_error)}"
        )

    # 3-3. fL - fR against the A4 coefficient
    print()
    print("A4 cross-check")
    print(
        f"{'Charge':>8}"
        f"{'A4':>18}"
        f"{'fL - fR':>18}"
        f"{'-/+ A4/2':>18}"
    )

    for charge_label, result, charge_sign in (
        ("W+", result_plus, 1),
        ("W-", result_minus, -1)
    ):
        print(
            f"{charge_label:>8}"
            f"{format_with_error(result.a4, result.a4_error)}"
            f"{format_with_error(result.fl - result.fr, result.a4_error/2)}"
            f"{format_with_error(-charge_sign*result.a4/2, result.a4_error/2)}"
        )

    # 3-4. cos(theta*) distribution
    draw_histogram_pair(
        cos_theta_star_w_plus,
        cos_theta_star_w_minus,
        r"$W^+$",
        r"$W^-$",
        50,
        -1,
        1,
        r"$\cos\theta^*$",
        "Number of leptonic W decays",
        "w_cos_theta_star_raw_count.png"
    )
    draw_histogram_pair(
        cos_theta_star_w_plus,
        cos_theta_star_w_minus,
        r"$W^+$",
        r"$W^-$",
        50,
        -1,
        1,
        r"$\cos\theta^*$",
        "Probability density",
        "w_cos_theta_star_normalized.png",
        density = True
    )

    # 4. pT-binned W polarization trend (Bern et al., PRD 84, 034008, figures 13-16)
    # 4-1. fL, fR, f0 in each pT bin, for W+ and W- separately
    pt_bin_centers = []
    result_plus_by_bin = []
    result_minus_by_bin = []

    print()
    print("W polarization fractions by pT bin")
    print(
        f"{'pT bin [GeV]':>16}"
        f"{'Charge':>8}"
        f"{'Entries':>10}"
        f"{'fL':>18}"
        f"{'fR':>18}"
        f"{'f0':>18}"
    )

    for bin_index in range(len(PT_BIN_EDGES) - 1):
        bin_minimum = PT_BIN_EDGES[bin_index]
        bin_maximum = PT_BIN_EDGES[bin_index + 1]
        bin_center = (bin_minimum + bin_maximum)/2
        bin_label = f"{bin_minimum}-{bin_maximum}"

        cos_theta_star_bin_plus = select_by_pid_and_pt(
            w_pid,
            w_pt,
            cos_theta_star_values,
            24,
            bin_minimum,
            bin_maximum
        )
        cos_theta_star_bin_minus = select_by_pid_and_pt(
            w_pid,
            w_pt,
            cos_theta_star_values,
            -24,
            bin_minimum,
            bin_maximum
        )

        if len(cos_theta_star_bin_plus) == 0 or len(cos_theta_star_bin_minus) == 0:
            print(
                f"{bin_label:>16}"
                f"{'both':>8}"
                f"{'skipped':>10}"
            )
            continue

        result_bin_plus = polarization_fractions(cos_theta_star_bin_plus, 1)
        result_bin_minus = polarization_fractions(cos_theta_star_bin_minus, -1)

        pt_bin_centers.append(bin_center)
        result_plus_by_bin.append(result_bin_plus)
        result_minus_by_bin.append(result_bin_minus)

        for charge_label, result in (("W+", result_bin_plus), ("W-", result_bin_minus)):
            print(
                f"{bin_label:>16}"
                f"{charge_label:>8}"
                f"{result.entries:>10d}"
                f"{format_with_error(result.fl, result.fl_error)}"
                f"{format_with_error(result.fr, result.fr_error)}"
                f"{format_with_error(result.f0, result.f0_error)}"
            )

    # 4-2. fL trend versus W pT
    fig, ax = plt.subplots(figsize = (8, 6))

    ax.errorbar(
        pt_bin_centers,
        [result.fl for result in result_plus_by_bin],
        yerr = [result.fl_error for result in result_plus_by_bin],
        marker = "o",
        linewidth = 1.5,
        capsize = 3,
        label = r"$W^+$"
    )
    ax.errorbar(
        pt_bin_centers,
        [result.fl for result in result_minus_by_bin],
        yerr = [result.fl_error for result in result_minus_by_bin],
        marker = "o",
        linewidth = 1.5,
        linestyle = "--",
        capsize = 3,
        label = r"$W^-$"
    )
    ax.set_xlabel(r"$p_T^W$ [GeV]")
    ax.set_ylabel(r"$f_L$")
    ax.legend(frameon = False)
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.tick_params(direction = "in", top = True, right = True)

    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY/"w_fl_vs_pt.png", bbox_inches = "tight")
    plt.close(fig)

    # 4-3. fL vs pT for W+ under different rapidity cuts
    results_by_rapidity_cut = {}

    for rapidity_cut in RAPIDITY_CUTS:
        results_for_cut = []

        for bin_index in range(len(PT_BIN_EDGES) - 1):
            bin_minimum = PT_BIN_EDGES[bin_index]
            bin_maximum = PT_BIN_EDGES[bin_index + 1]

            cos_theta_star_bin_cut = select_by_pid_pt_and_rapidity(
                w_pid,
                w_pt,
                w_rapidity,
                cos_theta_star_values,
                24,
                bin_minimum,
                bin_maximum,
                rapidity_cut
            )

            if len(cos_theta_star_bin_cut) == 0:
                results_for_cut.append(None)
                continue

            results_for_cut.append(polarization_fractions(cos_theta_star_bin_cut, 1))

        results_by_rapidity_cut[rapidity_cut] = results_for_cut

    print()
    print("W+ fL by pT bin, under rapidity cuts")

    header_row = f"{'Rapidity cut':>16}"
    for bin_index in range(len(PT_BIN_EDGES) - 1):
        bin_label = f"{PT_BIN_EDGES[bin_index]}-{PT_BIN_EDGES[bin_index + 1]} GeV"
        header_row += f"{bin_label:>18}"
    print(header_row)

    for rapidity_cut in RAPIDITY_CUTS:
        results_for_cut = results_by_rapidity_cut[rapidity_cut]
        cut_label = "No cut" if rapidity_cut is None else f"|y| < {rapidity_cut}"

        data_row = f"{cut_label:>16}"
        for result in results_for_cut:
            if result is None:
                data_row += f"{'-':>18}"
            else:
                data_row += format_with_error(result.fl, result.fl_error)
        print(data_row)

    # 4-4. fL(pT) curves for each rapidity cut
    pt_bin_centers_all = [
        (PT_BIN_EDGES[index] + PT_BIN_EDGES[index + 1])/2
        for index in range(len(PT_BIN_EDGES) - 1)
    ]

    fig, ax = plt.subplots(figsize = (8, 6))

    for rapidity_cut in RAPIDITY_CUTS:
        results_for_cut = results_by_rapidity_cut[rapidity_cut]

        plot_centers = [
            center
            for center, result in zip(pt_bin_centers_all, results_for_cut)
            if result is not None
        ]
        plot_values = [
            result.fl
            for result in results_for_cut
            if result is not None
        ]
        plot_errors = [
            result.fl_error
            for result in results_for_cut
            if result is not None
        ]

        cut_label = r"No cut" if rapidity_cut is None else rf"$|y_W| < {rapidity_cut}$"
        ax.errorbar(
            plot_centers,
            plot_values,
            yerr = plot_errors,
            marker = "o",
            linewidth = 1.5,
            capsize = 3,
            label = cut_label
        )

    ax.set_xlabel(r"$p_T^W$ [GeV]")
    ax.set_ylabel(r"$f_L$ ($W^+$)")
    ax.legend(frameon = False)
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.tick_params(direction = "in", top = True, right = True)

    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY/"w_fl_vs_pt_by_rapidity_cut.png", bbox_inches = "tight")
    plt.close(fig)


if __name__ == "__main__":
    w_polarization()
