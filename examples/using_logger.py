from pathlib import Path
import logging
from tempfile import TemporaryFile
import re

import matplotlib.pyplot as plt
import numpy as np

from cgkmc import simulations, containers, utils


NUM_STEPS = 25_000
DUMP_EVERY = 100


def main():

    logging_path = Path("bcc.log")
    with logging_path.open("w"):
        pass

    logging.basicConfig(
        filename=logging_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s TIME=%(t)s ENERGY=%(total_energy)s OCCUPANCY=%(occupancy)s'
    )

    simulation = simulations.Simulation(
        lattice=containers.CubicLattice(
            dimensions=(10, 10, 10),
            lattice_parameters=(3.0, 3.0, 3.0),
            atomic_basis=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        ),
        interactions=simulations.KthNearest(
            cutoffs=(2.7,),
            interaction_energies=(-1.0,),
            use_cache=True
        ),
        solvent=containers.Solvent(
            beta=utils.temp_to_beta(temperature=300, units=utils.Units.metal),
            diffusivity=1.0e+11,
            solubility_limit=1.0e-4
        ),
        growth=containers.Growth(
            initial_radius=15.0,
            num_steps=NUM_STEPS,
            desired_size=1_000
        )
    )

    with TemporaryFile(mode="w") as file:
        simulation.perform(file, dump_every=DUMP_EVERY)

    times = np.zeros(NUM_STEPS // DUMP_EVERY)
    energies = np.zeros(NUM_STEPS // DUMP_EVERY)
    occupancies = np.zeros(NUM_STEPS // DUMP_EVERY)
    with logging_path.open("r") as file:
        for i, line in enumerate(file):
            match = re.search(r"TIME=([-\d.eE]+) ENERGY=([-\d.eE]+) OCCUPANCY=([-\d.eE]+)", line)
            t, energy, occupancy = match.groups()
            times[i] = float(t)
            energies[i] = float(energy)
            occupancies[i] = float(occupancy)

    num_sites = len(simulation.lattice_points)
    plt.plot(times, energies / (occupancies * num_sites))
    plt.xlabel("time")
    plt.ylabel("energy per molecule")
    plt.grid()
    plt.savefig("energy.png", bbox_inches="tight")


if __name__ == "__main__":

    main()
