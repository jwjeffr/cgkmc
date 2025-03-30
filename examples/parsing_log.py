from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


NUM_STEPS = 25_000
DUMP_EVERY = 100


def main():

    times = np.zeros(NUM_STEPS // DUMP_EVERY)
    energies = np.zeros_like(times)
    occupancies = np.zeros_like(times)
    with Path("bcc.log").open("r") as file:
        for i, line in enumerate(file):
            match = re.search(r"TIME=([-\d.eE]+) ENERGY=([-\d.eE]+) OCCUPANCY=([-\d.eE]+)", line)
            t, energy, occupancy = match.groups()
            times[i] = float(t)
            energies[i] = float(energy)
            occupancies[i] = float(occupancy)

    # 10 x 10 x 10 lattice with 2 molecules per unit cell
    num_sites = 10 * 10 * 10 * 2
    plt.plot(times, energies / (occupancies * num_sites))
    plt.xlabel("time")
    plt.ylabel("energy per molecule")
    plt.grid()
    plt.savefig("energy.png", bbox_inches="tight")


if __name__ == "__main__":

    main()
