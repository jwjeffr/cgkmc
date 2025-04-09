"""
This module provides the main `Simulation` class to perform the growth KMC.
"""


from dataclasses import dataclass, field
from typing import Optional, Tuple, IO
from functools import cached_property
from io import StringIO
import logging
import warnings

import numpy as np
import scipy # type: ignore

from .containers import Solvent, Growth, CubicLattice, KthNearest

logger = logging.getLogger(__name__)


@dataclass
class Simulation:
    """
    Simulation class.
    Main method is `Simulation.perform`, which performs a crystal growth simulation using Kinetic Monte Carlo

    Attributes:
        lattice (CubicLattice):
            The lattice on which to perform KMC
        interactions (KthNearest):
            The interactions between lattice sites
        solvent (Solvent):
            The solvent surrounding the initial spherical crystal
        growth (Growth):
            Growth parameters for the simulation
        seed (int):
            Optional seed to provide to the simulation for random number generation. Defaults to 0
        generator (np.random.Generator):
            Generator for the simulation. This attribute is set after initialization
        lattice_points (np.ndarray):
            Lattice points in the supercell. This attribute is set after initialization
        bounds (np.ndarray):
            Supercell bounds. This attribute is set after initialization
        surface_density (float):
            Surface density of the initial spherical seed in units of molecules per area. This attribute is set after
            initialization
    """

    lattice: CubicLattice
    """lattice inside supercell"""

    interactions: KthNearest
    """interactions between lattice sites"""

    solvent: Solvent
    """solvent surrounding initial seed"""

    growth: Growth
    """Growth parameters"""

    seed: Optional[int] = 0
    """Optional seed for RNG"""

    generator: np.random.Generator = field(init=False)
    """Random number generator"""

    lattice_points: np.typing.NDArray[np.floating] = field(init=False)
    """Lattice points inside supercell"""

    bounds: np.typing.NDArray[np.floating] = field(init=False)
    """Super cell bounds"""

    surface_density: float = field(init=False)
    """Initial spherical surface density"""

    def __post_init__(self):

        self.generator = np.random.default_rng(seed=self.seed)
        self.lattice_points, self.bounds = self.lattice.initialize_simulation()

    @cached_property
    def hamiltonian(self) -> scipy.sparse.csr_matrix:

        r"""
        Defines the Hamiltonian matrix $\mathbf{Q}$, which defines the energy function:
        $$E(x) = \frac{1}{2}\mathbf{x}^\intercal\mathbf{Q}\mathbf{x}$$

        Returns:
            scipy.sparse.csr_matrix: Interaction matrix $\mathbf{Q}$
        """

        return self.interactions.get_hamiltonian(self.lattice_points, self.bounds)

    @cached_property
    def adjacency_matrix(self) -> scipy.sparse.csr_matrix:

        r"""
        Defines the adjacency matrix $\mathbf{A}$, where $A_{ij} = 1$ if sites $i$ and $j$ have an interaction.

        Returns:
            scipy.sparse.csr_matrix: Adjacency matrix $\mathbf{A}$
        """

        return scipy.sparse.csr_matrix(
            (np.ones_like(self.hamiltonian.data), self.hamiltonian.indices, self.hamiltonian.indptr),
            shape=self.hamiltonian.shape
        )

    @cached_property
    def num_neighbors(self) -> int:

        r"""
        Coordination number of the lattice.

        Returns:
            int: Coordination number, counting all neighbors
        """

        num_neighbors_per_site = self.adjacency_matrix.sum(axis=0)
        if not np.isclose(num_neighbors_per_site.std(), 0.0):
            warnings.warn(
                "Non-periodic lattice. There is likely problems with periodic boundary conditions. This is probably "
                "fine if your crystal does not grow to the boundary"
            )
        return num_neighbors_per_site.mean()

    def get_lammps_dump_str(
        self,
        types: np.typing.NDArray[np.floating],
        step: int,
        t: float,
        fmt: Tuple[str, str, str, str, str] = ("%.0f", "%.0f", "%.4f", "%.4f", "%.4f")
    ) -> str:

        r"""
        Shortcut method for getting a LAMMPS-style dump str, which looks like:

        ```
        ITEM: TIMESTEP
        0 0.0
        ITEM: NUMBER OF ATOMS
        1874
        ITEM: BOX BOUNDS xy xz xx yy zz
        0.0 318.08 0.0
        0.0 318.08 0.0
        0.0 303.165 0.0
        ITEM: ATOMS id type x y z
        39105 1 113.6000 131.7760 151.5825
        39189 1 113.6000 140.8640 131.3715
        39191 1 113.6000 140.8640 138.1085
        39193 1 113.6000 140.8640 144.8455
        39195 1 113.6000 140.8640 151.5825
        39197 1 113.6000 140.8640 158.3195
        ...
        ```

        at each timestep

        Arguments:
            types (np.ndarray):
                State matrix $\mathbf{x}$, where $x_i = 1$ if site $i$ is occupied and $0$ else.
            step (int):
                Current step in the simulation.
            t (float):
                Current time in the simulation.
            fmt (Tuple[str, str, str, str, str]):
                Sequence of format specifiers for LAMMPS-style dumping.

        Returns:
            str: The LAMMPS-style dump string
        """

        mask = types == 1

        ids = np.arange(len(self.lattice_points))
        occupied_ids = ids[mask].reshape(-1, 1)
        occupied_sites = self.lattice_points[mask, :]

        header = "\n".join([
            "ITEM: TIMESTEP",
            f"{step:.0f} {t}",
            "ITEM: NUMBER OF ATOMS",
            f"{len(occupied_ids):.0f}",
            "ITEM: BOX BOUNDS xy xz xx yy zz",
            f"0.0 {self.bounds[0]} 0.0",
            f"0.0 {self.bounds[1]} 0.0",
            f"0.0 {self.bounds[2]} 0.0",
            "ITEM: ATOMS id type x y z"
        ])

        data = np.concatenate((occupied_ids, np.ones_like(occupied_ids), occupied_sites), axis=1)

        with StringIO() as string_io:
            np.savetxt(string_io, data, fmt=fmt, comments="", header=header) # type: ignore
            return string_io.getvalue()

    @property
    def kappa(self) -> float:

        r"""
        Shorthand when computing dynamic evaporation prefactor, i.e.:

        $$\nu_t = \kappa / \left\langle \exp\left(\beta\Delta E_\text{evap}\right)\right\rangle$$

        Returns:
            float: $\kappa$
        """

        # only defined if the surface density has been calculated or not
        if not hasattr(self, "surface_density"):
            raise ValueError("surface density not yet calculated")

        return (4.0 / 3.0 * np.pi * self.lattice.density) ** (1 / 3) * self.solvent.diffusivity * \
            self.solvent.solubility_limit / (self.surface_density * self.growth.desired_size ** (1 / 3))

    def get_interface(
        self,
        types: np.typing.NDArray[np.floating]
    ) -> Tuple[np.typing.NDArray[np.integer], np.typing.NDArray[np.integer]]:

        r"""
        Function for computing interfacial solvent and solid sites.
        $\mathbf{A}\mathbf{x}$ counts the number of currently occupied neighbors.

        Arguments:
            types (np.ndarray): State vector $\mathbf{x}$

        Returns:
            Tuple[np.ndarray, np.ndarray]: IDs of solid sites and IDs of solvent sites at interface
        """

        occupied_neighbor_count = self.adjacency_matrix @ types
        solid_sites, = np.where(types * (self.num_neighbors - occupied_neighbor_count) > 0)
        solvent_sites, = np.where((1.0 - types) * occupied_neighbor_count > 0)

        return solid_sites, solvent_sites

    def perform(self, dump_file: IO, dump_every: int):

        """
        Main method for performing a Kinetic Monte Carlo simulation. This is performable by calling something like:

        ```py
        simulation = Simulation(...)
        with open("kmc.dump", "w") as file:
            simulation.perform(dump_file=file, dump_every=100)
        ```

        or any other IO-like object, such as StringIO or BytesIO

        Arguments:

            dump_file (IO):
                `IO` instance where user will store dump information
            dump_every (int):
                How often to dump information
        """

        # initialize a spherical crystal with specified radius
        center = self.lattice_points.mean(axis=0)
        types = (np.linalg.norm(self.lattice_points - center, axis=1) <= self.growth.initial_radius).astype(float)

        solid_sites, _ = self.get_interface(types)
        self.surface_density = len(solid_sites) / self.growth.initial_surface_area

        t = 0.0
        total_energy = 0.5 * types.T @ self.hamiltonian @ types
        for step in range(self.growth.num_steps):

            # if types.mean() is 0, entire cell is unoccupied, so no simulation
            # similarly, if types.mean() is 1, entire cell is occupied
            occupancy = types.mean()
            if not 0.0 < occupancy < 1.0:
                logging.error("simulation killed, no interface detected")
                return

            # only want dumps at a small frequency
            if not step % dump_every:
                positions = self.get_lammps_dump_str(types, step, t)
                print(positions, file=dump_file, end="")
                logging.info("simulation info", extra={
                    "step": step, "t": t, "total_energy": total_energy, "occupancy": occupancy
                })
                dump_file.flush()

            solid_sites, solvent_sites = self.get_interface(types)

            # need to rescale evaporation rates after calculating
            # this is the dynamically updated prefactor
            evaporation_barriers = -self.hamiltonian[solid_sites, :] @ types
            evaporation_rates = np.exp(-self.solvent.beta * evaporation_barriers)
            rate_prefactor = self.kappa / evaporation_rates.mean()
            evaporation_rates = rate_prefactor * evaporation_rates

            radius = (0.75 * self.lattice.molecular_volume * types.sum() / np.pi) ** (1 / 3)
            adsorption_rates = np.ones_like(solvent_sites) * self.solvent.diffusivity * \
                self.solvent.solubility_limit / (self.surface_density * radius)

            # concatenate events, so we can pick event according to residence time algorithm
            events = np.concatenate((solid_sites, solvent_sites))
            rates = np.concatenate((evaporation_rates, adsorption_rates))
            total_rate = rates.sum()

            # pick event and advance time according to residence time algorithm
            event = self.generator.choice(events, p=rates / total_rate)

            # if there was an evaporation event, we've already computed \Delta E_evap
            if types[event]:
                change_in_energy_evap = evaporation_barriers[np.where(events == event)].item()
                total_energy += change_in_energy_evap
            # if it was an adsorption event, need to compute it
            elif not types[event]:
                change_in_energy_evap = -(self.hamiltonian[event, :] @ types).item()
                total_energy += -change_in_energy_evap
            else:
                raise ValueError

            types[event] = 1.0 - types[event]
            t += self.generator.exponential(scale=1.0 / total_rate)
