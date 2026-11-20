import itertools
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyhepmc

from w_common import (
    CENTER_OF_MASS_ENERGY,
    CHILD_LABELS,
    FIGURE_DIRECTORY,
    SAMPLE_FILE,
    TOP_PID,
    W_MASS_PDG,
    W_PID,
    calculate_invariant_mass,
    draw_histogram,
    draw_histogram_pair,
    find_last_ws,
    find_lepton_neutrino_pair,
    particle_name,
    particle_species_name,
    print_outside_count,
    print_pt_summary,
    select_by_pid,
    sum_four_momenta
)

NUMBER_OF_INSPECTION_EVENTS = 10
NUMBER_OF_ANALYSIS_EVENTS = 10000

MAXIMUM_PARTICLES_TO_PRINT = 20


def w_production():
    # 1. Event reading with pyhepmc
    # 1-1. Basic pyhepmc event loop
    print()
    print("Event summary")
    print(
        f"{'Event':>10}"
        f"{'Particles':>10}"
        f"{'Vertices':>10}"
        f"{'Momentum unit':>20}"
    )

    with pyhepmc.open(SAMPLE_FILE) as events:
        for event in itertools.islice(events, NUMBER_OF_INSPECTION_EVENTS):
            print(
                f"{event.event_number:10d}"
                f"{len(event.particles):10d}"
                f"{len(event.vertices):10d}"
                f"{str(event.momentum_unit):>20}"
            )

    # 1-2. First-event particle table
    print()
    print("Particle table for the first event")

    with pyhepmc.open(SAMPLE_FILE) as events:
        for event in itertools.islice(events, NUMBER_OF_INSPECTION_EVENTS):
            print(
                f"{'ID':>10}"
                f"{'PID':>10}"
                f"  {'Particle':<10}"
                f"{'Status':>10}"
                f"{'px':>10}"
                f"{'py':>10}"
                f"{'pz':>10}"
                f"{'E':>10}"
                f"{'Parents':>10}"
                f"{'Children':>10}"
            )
            for particle in event.particles[:MAXIMUM_PARTICLES_TO_PRINT]:
                momentum = particle.momentum
                print(
                    f"{particle.id:10d}"
                    f"{particle.pid:10d}"
                    f"  {particle_name(particle.pid):<10}"
                    f"{particle.status:10d}"
                    f"{momentum.px:10.2f}"
                    f"{momentum.py:10.2f}"
                    f"{momentum.pz:10.2f}"
                    f"{momentum.e:10.2f}"
                    f"{len(particle.parents):10d}"
                    f"{len(particle.children):10d}"
                )

            break

    # 2. Last-copy W-boson selection
    # 2-1. All W entries and last-copy W entries
    multiplicity_counts = Counter()

    print()
    print("W multiplicity")

    with pyhepmc.open(SAMPLE_FILE) as events:
        print(
            f"{'Event':>10}"
            f"{'All W entries':>20}"
            f"{'Last-copy W':>20}"
        )

        for event in itertools.islice(events, NUMBER_OF_INSPECTION_EVENTS):
            all_ws = [
                particle
                for particle in event.particles
                if abs(particle.pid) == W_PID
            ]

            last_ws = find_last_ws(event)
            multiplicity = len(last_ws)

            print(
                f"{event.event_number:10d}"
                f"{len(all_ws):20d}"
                f"{multiplicity:20d}"
            )

            multiplicity_counts[multiplicity] += 1

            if multiplicity != 1:
                print(
                    "Event:", event.event_number,
                    "Does not contain exactly one last-copy W"
                )

    # 2-2. Last-copy W multiplicity frequency
    print()
    print("Last-copy W multiplicity frequency")

    for multiplicity in sorted(multiplicity_counts):
        label = f"Multiplicity {multiplicity}"
        print(f"{label:<24}: {multiplicity_counts[multiplicity]}")

    # 2-3. Direct children of last-copy W bosons
    print()
    print("Children of last-copy W bosons")
    print(
        f"{'Event':>10}"
        f"{'W PID':>10}"
        f"  {'W':<10}"
        f"  {'Child PIDs':<20}"
        "Child particles"
    )

    with pyhepmc.open(SAMPLE_FILE) as events:
        for event in itertools.islice(events, NUMBER_OF_INSPECTION_EVENTS):
            last_ws = find_last_ws(event)

            for w in last_ws:
                child_pids = [child.pid for child in w.children]

                print(
                    f"{event.event_number:10d}"
                    f"{w.pid:10d}"
                    f"  {particle_name(w.pid):<10}"
                    f"  {str(child_pids):<20}"
                    f"{', '.join(particle_name(pid) for pid in child_pids)}"
                )

    # 3. W-boson kinematics and decay analysis
    w_pid = []
    w_pt = []
    w_rapidity = []
    w_pseudorapidity = []
    w_phi = []
    w_mass = []

    w_mass_from_children = []
    w_mass_reconstruction_difference = []

    child_pid_counts = Counter()
    child_combination_counts = Counter()
    child_pt = []
    child_theta = []

    non_two_body_decay_events = []

    lepton_pt = []
    neutrino_pt = []
    lepton_neutrino_delta_r = []
    transverse_mass = []

    momentum_unit = None
    pt_manual = None
    pt_pyhepmc = None

    unexpected_multiplicity_events = []
    top_quark_events = []

    with pyhepmc.open(SAMPLE_FILE) as events:
        for event in itertools.islice(events, NUMBER_OF_ANALYSIS_EVENTS):
            if momentum_unit is None:
                momentum_unit = event.momentum_unit

            if any(abs(particle.pid) == TOP_PID for particle in event.particles):
                top_quark_events.append(event.event_number)

            last_ws = find_last_ws(event)

            if len(last_ws) != 1:
                unexpected_multiplicity_events.append(
                    (event.event_number, len(last_ws))
                )

            for w in last_ws:
                momentum = w.momentum

                w_pid.append(w.pid)
                w_pt.append(momentum.pt())
                w_rapidity.append(momentum.rap())
                w_pseudorapidity.append(momentum.eta())
                w_phi.append(momentum.phi())
                w_mass.append(momentum.m())

                # Cross-check the first selected W
                if pt_manual is None:
                    pt_manual = math.sqrt(momentum.px**2 + momentum.py**2)
                    pt_pyhepmc = momentum.pt()

                # W mass reconstructed from direct children
                (
                    reconstructed_energy,
                    reconstructed_px,
                    reconstructed_py,
                    reconstructed_pz
                ) = sum_four_momenta(w.children)

                reconstructed_mass = calculate_invariant_mass(
                    reconstructed_energy,
                    reconstructed_px,
                    reconstructed_py,
                    reconstructed_pz
                )

                w_mass_from_children.append(reconstructed_mass)
                w_mass_reconstruction_difference.append(reconstructed_mass - momentum.m())

                # Check whether the W decayed into exactly two direct children
                if len(w.children) != 2:
                    non_two_body_decay_events.append(
                        (event.event_number, w.pid, len(w.children))
                    )

                # Record the combination of direct-child PIDs
                child_combination_counts[
                    tuple(sorted(child.pid for child in w.children))
                ] += 1

                # Direct W-decay children
                for child in w.children:
                    child_pid_counts[abs(child.pid)] += 1

                    child_momentum = child.momentum

                    child_pt.append(child_momentum.pt())
                    child_theta.append(child_momentum.theta())

                # Leptonic W decay
                lepton, neutrino = find_lepton_neutrino_pair(w)

                if lepton is not None:
                    lepton_momentum = lepton.momentum
                    neutrino_momentum = neutrino.momentum

                    lepton_pt_value = lepton_momentum.pt()
                    neutrino_pt_value = neutrino_momentum.pt()

                    delta_phi = pyhepmc.delta_phi(lepton_momentum, neutrino_momentum)
                    delta_r = pyhepmc.delta_r_eta(lepton_momentum, neutrino_momentum)

                    mt_squared = 2*lepton_pt_value*neutrino_pt_value*(1 - math.cos(delta_phi))

                    lepton_pt.append(lepton_pt_value)
                    neutrino_pt.append(neutrino_pt_value)

                    lepton_neutrino_delta_r.append(delta_r)

                    transverse_mass.append(math.sqrt(max(0, mt_squared)))

    # 3-1. Selected W-boson counts
    print()
    print("W kinematics")
    print(f"{'Momentum unit':<24}: {str(momentum_unit)}")
    print(f"{'Selected W bosons':<24}: {len(w_pid)}")
    print(f"{'W+ bosons':<24}: {sum(pid == W_PID for pid in w_pid)}")
    print(f"{'W- bosons':<24}: {sum(pid == -W_PID for pid in w_pid)}")

    # 3-2. pT cross-check against a manual calculation
    print()
    print("pT cross-check")
    print(f"{'Manual calculation':<24}: {pt_manual:.4f} GeV")
    print(f"{'pyhepmc calculation':<24}: {pt_pyhepmc:.4f} GeV")

    # 3-3. W-mass reconstruction from direct children
    maximum_mass_difference = max(
        abs(difference)
        for difference in w_mass_reconstruction_difference
    )

    print()
    print("W-mass reconstruction from direct children")
    print(f"{'Maximum mass difference':<24}: {maximum_mass_difference:.2e} GeV")

    # 3-4. Last-copy W multiplicity over the analysis sample
    print()
    print("Events with unexpected last-copy W multiplicity")
    print(f"{'Number of events':<24}: {len(unexpected_multiplicity_events)}")

    for event_number, multiplicity in unexpected_multiplicity_events:
        label = f"Event {event_number}"
        print(f"{label:<24}: {multiplicity}")

    # 3-5. Direct W-child particle frequencies
    print()
    print("Direct children of last-copy W bosons")
    print(
        f"{'Absolute PDG ID':>20}"
        f"{'Particles':>20}"
        f"{'Count':>10}"
    )

    for pid, count in child_pid_counts.most_common():
        print(
            f"{pid:20d}"
            f"{particle_species_name(pid):>20}"
            f"{count:10d}"
        )

    # 3-6. Events where a selected W did not decay into exactly two children
    print()
    print("Non-two-body last-copy W decays")
    print(f"{'Number of events':<24}: {len(non_two_body_decay_events)}")

    if non_two_body_decay_events:
        print(
            f"{'Event':>10}"
            f"{'W PID':>10}"
            f"{'Number of children':>20}"
        )

        for event_number, pid, child_count in non_two_body_decay_events:
            print(
                f"{event_number:10d}"
                f"{pid:10d}"
                f"{child_count:20d}"
            )

    # 3-7. Combinations of direct W-child PIDs
    print()
    print("Direct-child PID combinations of last-copy W bosons")
    print(
        f"{'Child PIDs':<20}"
        f"{'Child particles':<20}"
        f"{'Count':>10}"
    )

    for combination, count in child_combination_counts.most_common():
        print(
            f"{str(combination):<20}"
            f"{', '.join(particle_name(pid) for pid in combination):<20}"
            f"{count:10d}"
        )

    # 3-8. Top-quark presence in this sample
    print()
    print("Top-quark presence check")
    print(f"{'Events with a top quark':<24}: {len(top_quark_events)}")

    # 3-9. Leptonic W-decay summary
    print()
    print("Leptonic W-decay sample")
    print(f"{'Selected leptonic decays':<24}: {len(lepton_pt)}")

    # 4. W-boson distributions
    # Separate W+ and W- samples
    w_plus_pt = select_by_pid(
        w_pid,
        w_pt,
        W_PID
    )
    w_minus_pt = select_by_pid(
        w_pid,
        w_pt,
        -W_PID
    )

    w_plus_rapidity = select_by_pid(
        w_pid,
        w_rapidity,
        W_PID
    )
    w_minus_rapidity = select_by_pid(
        w_pid,
        w_rapidity,
        -W_PID
    )

    w_plus_pseudorapidity = select_by_pid(
        w_pid,
        w_pseudorapidity,
        W_PID
    )
    w_minus_pseudorapidity = select_by_pid(
        w_pid,
        w_pseudorapidity,
        -W_PID
    )

    w_plus_phi = select_by_pid(
        w_pid,
        w_phi,
        W_PID
    )
    w_minus_phi = select_by_pid(
        w_pid,
        w_phi,
        -W_PID
    )

    w_plus_mass = select_by_pid(
        w_pid,
        w_mass,
        W_PID
    )
    w_minus_mass = select_by_pid(
        w_pid,
        w_mass,
        -W_PID
    )

    w_plus_mass_from_children = select_by_pid(
        w_pid,
        w_mass_from_children,
        W_PID
    )
    w_minus_mass_from_children = select_by_pid(
        w_pid,
        w_mass_from_children,
        -W_PID
    )

    # Axis ranges
    pt_minimum = 0
    pt_maximum = 200

    rapidity_maximum = math.log(CENTER_OF_MASS_ENERGY/W_MASS_PDG)
    rapidity_minimum = -rapidity_maximum

    pseudorapidity_minimum = -5
    pseudorapidity_maximum = 5

    phi_minimum = -math.pi
    phi_maximum = math.pi

    mass_minimum = 50
    mass_maximum = 100

    print()
    print("W-boson entries outside plot range")
    print_outside_count("pT", w_pt, pt_minimum, pt_maximum)
    print_outside_count("Rapidity", w_rapidity, rapidity_minimum, rapidity_maximum)
    print_outside_count(
        "Pseudorapidity",
        w_pseudorapidity,
        pseudorapidity_minimum,
        pseudorapidity_maximum
    )
    print_outside_count(
        "W+ pseudorapidity",
        w_plus_pseudorapidity,
        pseudorapidity_minimum,
        pseudorapidity_maximum
    )
    print_outside_count(
        "W- pseudorapidity",
        w_minus_pseudorapidity,
        pseudorapidity_minimum,
        pseudorapidity_maximum
    )
    print_outside_count("Phi", w_phi, phi_minimum, phi_maximum)
    print_outside_count("Mass", w_mass, mass_minimum, mass_maximum)

    # 4-1. W-boson transverse momentum
    draw_histogram_pair(
        w_plus_pt,
        w_minus_pt,
        r"$W^+$",
        r"$W^-$",
        50,
        pt_minimum,
        pt_maximum,
        r"$p_T$ [GeV]",
        "Number of W bosons",
        "w_pt_raw_count_linear.png"
    )
    draw_histogram_pair(
        w_plus_pt,
        w_minus_pt,
        r"$W^+$",
        r"$W^-$",
        50,
        pt_minimum,
        pt_maximum,
        r"$p_T$ [GeV]",
        "Number of W bosons",
        "w_pt_raw_count_log.png",
        log_scale = True
    )
    draw_histogram_pair(
        w_plus_pt,
        w_minus_pt,
        r"$W^+$",
        r"$W^-$",
        50,
        pt_minimum,
        pt_maximum,
        r"$p_T$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "w_pt_normalized_linear.png",
        density = True
    )
    draw_histogram_pair(
        w_plus_pt,
        w_minus_pt,
        r"$W^+$",
        r"$W^-$",
        50,
        pt_minimum,
        pt_maximum,
        r"$p_T$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "w_pt_normalized_log.png",
        density = True,
        log_scale = True
    )

    # 4-2. W-boson rapidity
    draw_histogram_pair(
        w_plus_rapidity,
        w_minus_rapidity,
        r"$W^+$",
        r"$W^-$",
        50,
        rapidity_minimum,
        rapidity_maximum,
        r"Rapidity $y$",
        "Number of W bosons",
        "w_rapidity_raw_count.png"
    )
    draw_histogram_pair(
        w_plus_rapidity,
        w_minus_rapidity,
        r"$W^+$",
        r"$W^-$",
        50,
        rapidity_minimum,
        rapidity_maximum,
        r"Rapidity $y$",
        "Probability density",
        "w_rapidity_normalized.png",
        density = True
    )

    # 4-3. W-boson pseudorapidity
    draw_histogram_pair(
        w_plus_pseudorapidity,
        w_minus_pseudorapidity,
        r"$W^+$",
        r"$W^-$",
        50,
        pseudorapidity_minimum,
        pseudorapidity_maximum,
        r"Pseudorapidity $\eta$",
        "Number of W bosons",
        "w_pseudorapidity.png"
    )

    # 4-4. W-boson azimuthal angle
    draw_histogram_pair(
        w_plus_phi,
        w_minus_phi,
        r"$W^+$",
        r"$W^-$",
        50,
        phi_minimum,
        phi_maximum,
        r"$\phi$ [rad]",
        "Number of W bosons",
        "w_phi_raw_count.png"
    )
    draw_histogram_pair(
        w_plus_phi,
        w_minus_phi,
        r"$W^+$",
        r"$W^-$",
        50,
        phi_minimum,
        phi_maximum,
        r"$\phi$ [rad]",
        r"Probability density [rad$^{-1}$]",
        "w_phi_normalized.png",
        density = True
    )

    # 4-5. W-boson invariant mass
    draw_histogram_pair(
        w_plus_mass,
        w_minus_mass,
        r"$W^+$",
        r"$W^-$",
        50,
        mass_minimum,
        mass_maximum,
        r"Invariant mass $m$ [GeV]",
        "Number of W bosons",
        "w_mass_raw_count_linear.png"
    )
    draw_histogram_pair(
        w_plus_mass,
        w_minus_mass,
        r"$W^+$",
        r"$W^-$",
        50,
        mass_minimum,
        mass_maximum,
        r"Invariant mass $m$ [GeV]",
        "Number of W bosons",
        "w_mass_raw_count_log.png",
        log_scale = True
    )
    draw_histogram_pair(
        w_plus_mass,
        w_minus_mass,
        r"$W^+$",
        r"$W^-$",
        50,
        mass_minimum,
        mass_maximum,
        r"Invariant mass $m$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "w_mass_normalized_linear.png",
        density = True
    )
    draw_histogram_pair(
        w_plus_mass,
        w_minus_mass,
        r"$W^+$",
        r"$W^-$",
        50,
        mass_minimum,
        mass_maximum,
        r"Invariant mass $m$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "w_mass_normalized_log.png",
        density = True,
        log_scale = True
    )

    # 4-6. W-boson invariant mass reconstructed from direct children
    draw_histogram_pair(
        w_plus_mass_from_children,
        w_minus_mass_from_children,
        r"$W^+$",
        r"$W^-$",
        50,
        mass_minimum,
        mass_maximum,
        r"Reconstructed invariant mass $m$ [GeV]",
        "Number of W bosons",
        "w_mass_from_children_linear.png"
    )
    draw_histogram_pair(
        w_plus_mass_from_children,
        w_minus_mass_from_children,
        r"$W^+$",
        r"$W^-$",
        50,
        mass_minimum,
        mass_maximum,
        r"Reconstructed invariant mass $m$ [GeV]",
        "Number of W bosons",
        "w_mass_from_children_log.png",
        log_scale = True
    )

    # 5. Direct children of last-copy W bosons
    # 5-1. Direct W-boson children particle species
    child_species = child_pid_counts.most_common()

    child_labels = [
        CHILD_LABELS.get(pid, str(pid))
        for pid, count in child_species
    ]
    child_counts = [
        count
        for pid, count in child_species
    ]
    child_errors = [
        math.sqrt(count)
        for count in child_counts
    ]
    positions = list(range(len(child_counts)))

    fig, ax = plt.subplots(figsize = (8, 6))

    ax.bar(
        positions,
        child_counts,
        yerr = child_errors,
        capsize = 3
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(child_labels)
    ax.set_xlabel("Direct W-boson child particle")
    ax.set_ylabel("Number of children")
    ax.grid(True, axis = "y", linestyle = ":", alpha = 0.5)
    ax.tick_params(direction = "in", top = True, right = True)

    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY/"w_child_species.png", bbox_inches = "tight")
    plt.close(fig)

    # Axis ranges
    child_pt_minimum = 0
    child_pt_maximum = 200

    child_theta_minimum = 0
    child_theta_maximum = math.pi

    print()
    print("Direct-child entries outside plot range")
    print_outside_count("pT", child_pt, child_pt_minimum, child_pt_maximum)
    print_outside_count("Theta", child_theta, child_theta_minimum, child_theta_maximum)

    # 5-2. Direct W-boson children transverse momentum
    draw_histogram(
        child_pt,
        50,
        child_pt_minimum,
        child_pt_maximum,
        r"Child $p_T$ [GeV]",
        "Number of W-boson children",
        "w_child_pt_linear.png"
    )
    draw_histogram(
        child_pt,
        50,
        child_pt_minimum,
        child_pt_maximum,
        r"Child $p_T$ [GeV]",
        "Number of W-boson children",
        "w_child_pt_log.png",
        log_scale = True
    )

    # 5-3. Direct W-boson children polar angle
    draw_histogram(
        child_theta,
        50,
        child_theta_minimum,
        child_theta_maximum,
        r"Polar angle $\theta$ [rad]",
        "Number of W-boson children",
        "w_child_theta.png"
    )

    # 6. Leptonic W decays
    if lepton_pt:
        lepton_pt_minimum = 0
        lepton_pt_maximum = 150

        delta_r_minimum = 0
        delta_r_maximum = 6

        transverse_mass_minimum = 0
        transverse_mass_maximum = 150

        print()
        print("Leptonic W-decay entries outside plot range")
        print_outside_count("Charged lepton pT", lepton_pt, lepton_pt_minimum, lepton_pt_maximum)
        print_outside_count("Neutrino pT", neutrino_pt, lepton_pt_minimum, lepton_pt_maximum)
        print_outside_count(
            "Delta R",
            lepton_neutrino_delta_r,
            delta_r_minimum,
            delta_r_maximum
        )
        print_outside_count(
            "Transverse mass",
            transverse_mass,
            transverse_mass_minimum,
            transverse_mass_maximum
        )

        # 6-1. Charged-lepton and neutrino transverse momenta
        draw_histogram_pair(
            lepton_pt,
            neutrino_pt,
            "Charged lepton",
            "Neutrino",
            50,
            lepton_pt_minimum,
            lepton_pt_maximum,
            r"$p_T$ [GeV]",
            "Number of leptonic W decays",
            "w_lepton_neutrino_pt_linear.png"
        )
        draw_histogram_pair(
            lepton_pt,
            neutrino_pt,
            "Charged lepton",
            "Neutrino",
            50,
            lepton_pt_minimum,
            lepton_pt_maximum,
            r"$p_T$ [GeV]",
            "Number of leptonic W decays",
            "w_lepton_neutrino_pt_log.png",
            log_scale = True
        )

        # 6-2. Charged-lepton and neutrino angular separation
        draw_histogram(
            lepton_neutrino_delta_r,
            50,
            delta_r_minimum,
            delta_r_maximum,
            r"$\Delta R(\ell,\nu)$",
            "Number of leptonic W decays",
            "w_lepton_neutrino_delta_r.png"
        )

        # 6-3. W-boson transverse mass
        draw_histogram(
            transverse_mass,
            50,
            transverse_mass_minimum,
            transverse_mass_maximum,
            r"Transverse mass $m_T$ [GeV]",
            "Number of leptonic W decays",
            "w_transverse_mass_linear.png"
        )
        draw_histogram(
            transverse_mass,
            50,
            transverse_mass_minimum,
            transverse_mass_maximum,
            r"Transverse mass $m_T$ [GeV]",
            "Number of leptonic W decays",
            "w_transverse_mass_log.png",
            log_scale = True
        )

    # 7. Full-sample W-boson pT check
    # 7-1. Charge and pT only, from every event in the file
    full_sample_w_pid = []
    full_sample_w_pt = []

    with pyhepmc.open(SAMPLE_FILE) as events:
        for event in events:
            last_ws = find_last_ws(event)

            for w in last_ws:
                full_sample_w_pid.append(w.pid)
                full_sample_w_pt.append(w.momentum.pt())

    full_sample_w_plus_pt = select_by_pid(
        full_sample_w_pid,
        full_sample_w_pt,
        W_PID
    )
    full_sample_w_minus_pt = select_by_pid(
        full_sample_w_pid,
        full_sample_w_pt,
        -W_PID
    )

    # 7-2. W+ and W- pT peaks
    print()
    print_pt_summary(
        "W+ pT summary (full sample)",
        full_sample_w_plus_pt,
        50,
        pt_minimum,
        pt_maximum
    )

    print()
    print_pt_summary(
        "W- pT summary (full sample)",
        full_sample_w_minus_pt,
        50,
        pt_minimum,
        pt_maximum
    )

    # 7-3. W-boson transverse momentum
    draw_histogram_pair(
        full_sample_w_plus_pt,
        full_sample_w_minus_pt,
        r"$W^+$",
        r"$W^-$",
        50,
        pt_minimum,
        pt_maximum,
        r"$p_T$ [GeV]",
        "Number of W bosons",
        "w_pt_full_sample_raw_count.png"
    )
    draw_histogram_pair(
        full_sample_w_plus_pt,
        full_sample_w_minus_pt,
        r"$W^+$",
        r"$W^-$",
        50,
        pt_minimum,
        pt_maximum,
        r"$p_T$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "w_pt_full_sample_normalized.png",
        density = True
    )


if __name__ == "__main__":
    w_production()
