"""
This module has container classes to propagate data through the KMC simulation. These container classes define the
solvent, growth, and lattice of the simulation, as well as the interaction energies between neighbors.
"""


from dataclasses import dataclass
from typing import Tuple, Optional
import logging
from pathlib import Path

import numpy as np
import scipy # type: ignore

from .utils import array_to_hex

logger = logging.getLogger(__name__)


@dataclass
class Solvent:
    r"""
    Solvent container class.

    Temperature (or $\beta$), diffusivity, and solubility limit fully define the solvent. $\beta$ should match the
    units of the interaction energies you specify. See `utils.Units` class for available energy units, and use
    `utils.temp_to_beta` to get $\beta$ from temperature in Kelvin.

    Attributes:
        beta (float):
            Thermodynamic $\beta$ defined as $1/(k_BT)$, where $k_B$ is the
            [Boltzmann constant](https://en.wikipedia.org/wiki/Boltzmann_constant) and $T$ is temperature
        diffusivity (float):
            Diffusivity of solute in liquid phase $D$
        solubility_limit (float):
            Solubility limit, or equivalently concentration of solute in liquid phase $n_\infty$
    """

    beta: float
    r"""thermodynamic $\beta$"""

    diffusivity: float
    r"""diffusivity of solute $D$"""

    solubility_limit: float
    r"""solubility limit $n_\infty$"""


@dataclass
class Growth:
    r"""
    Growth container class.
    Mimics experimental controls, i.e. initial crystal size, amount of time we grow, and the final size we want.

    Attributes:
        initial_radius (float):
            Initial radius of spherical seed in physical units, or the same units as the specified lattice parameters
        num_steps (int):
            Number of KMC steps to perform
        desired_size (int):
            The desired number of molecules in the final crystal. The final crystal will not have this exact number of
            molecules, but increasing this parameter will increase the final size.
    """

    initial_radius: float
    r"""initial radius $R_0$"""

    num_steps: int
    r"""number of KMC steps $N_\text{steps}$"""

    desired_size: int
    r"""desired size of final crystal $N_*$"""

    def __post_init__(self):

        if self.initial_radius <= 0:
            raise ValueError(f"{self.__class__.__name__}.initial_radius should be positive")

        if self.num_steps <= 0 or not isinstance(self.num_steps, int):
            raise ValueError(f"{self.__class__.__name__}.num_steps should be a positive integer")

        if self.desired_size <= 0 or not isinstance(self.desired_size, int):
            raise ValueError(f"{self.__class__.__name__}.desired_size should be a positive integer")

    @property
    def initial_surface_area(self) -> float:
        r"""
        Shortcut property for computing surface area.
        Crystal is assumed to be spherical, so surface area = $4\pi\times (\text{radius})^2$
        """

        return 4.0 * np.pi * self.initial_radius ** 2


@dataclass
class CubicLattice:

    r"""
    Lattice container class, which creates the resulting cubic lattice.

    Attributes:
        dimensions (np.ndarray):
            Dimensions of the cubic lattice in terms of unit cells. For a $10\times 15\times 20$ lattice, for example,
            specify `dimensions=np.array([10, 15, 20])`.
        lattice_parameters (np.ndarray):
            Lattice parameters of the lattice $a$, $b$, and $c$.
        atomic_basis (np.ndarray):
            Atomic basis of the solid phase's unit cell.
            Should be an array with size $(\text{number of molecules per unit cell}, 3)$
    """

    dimensions: np.typing.NDArray[np.integer]
    r"""Dimensions of the cubic lattice in terms of unit cells"""

    lattice_parameters: np.typing.NDArray[np.floating]
    r"""Lattice parameters of the lattice $a$, $b$, and $c$."""

    atomic_basis: np.typing.NDArray[np.floating]
    r"""Atomic basis of the solid phase's unit cell."""

    def __post_init__(self):
        # turn objects into arrays if they're not already arrays
        if not isinstance(self.dimensions, np.ndarray):
            self.dimensions = np.array(self.dimensions, dtype=int)
        if not isinstance(self.lattice_parameters, np.ndarray):
            self.lattice_parameters = np.array(self.lattice_parameters, dtype=float)
        if not isinstance(self.atomic_basis, np.ndarray):
            self.atomic_basis = np.array(self.atomic_basis, dtype=float)

        if len(self.dimensions.shape) != 1:
            raise ValueError(
                f"Invalid lattice geometry. {self.__class__.__name__}.dimensions should have shape "
                f"(number of spatial dimensions,)"
            )

        if len(self.lattice_parameters.shape) != 1:
            raise ValueError(
                f"Invalid lattice geometry. {self.__class__.__name__}.lattice_parameters should have shape "
                f"(number of spatial dimensions,)"
            )

        if len(self.atomic_basis.shape) != 2:
            raise ValueError(
                f"Invalid lattice geometry. {self.__class__.__name__}.atomic_basis should have shape "
                f"(number of particles in unit cell, number of spatial dimensions)."
            )

        if len(self.dimensions) != len(self.lattice_parameters) != self.atomic_basis.shape[1]:

            raise ValueError(
                f"Invalid lattice geometry. The number of spatial dimensions in {self.__class__.__name__}.dimensions, "
                f"{self.__class__.__name__}.lattice_parameters, and {self.__class__.__name__}.atomic_basis should "
                f"match."
            )

        if np.prod(self.lattice_parameters) <= 0:
            raise ValueError(
                "Invalid lattice geometry. The unit cell's volume, i.e. the product of lattice parameters, should be "
                "positive."
            )

    @property
    def density(self) -> float:

        r"""
        Molecular density of the solid phase, in units of molecules per volume

        Returns:
            float: density $\rho$
        """

        return self.atomic_basis.shape[0] / np.prod(self.lattice_parameters)

    @property
    def molecular_volume(self) -> float:

        r"""
        Molecular volume of the solid phase, in units of volume per molecule

        Returns:
            float: molecular volume $\omega = 1/\rho$
        """

        return 1.0 / self.density

    def initialize_simulation(self) -> Tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:

        r"""
        Initialize the simulation by creating the lattice positions and bounds.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The lattice points and supercell bounds
        """

        x = np.arange(self.dimensions[0]) * self.lattice_parameters[0]
        y = np.arange(self.dimensions[1]) * self.lattice_parameters[1]
        z = np.arange(self.dimensions[2]) * self.lattice_parameters[2]

        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        unit_cell_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

        lattice_points = unit_cell_points[:, None, :] + self.atomic_basis[None, :, :] * self.lattice_parameters
        lattice_points = lattice_points.reshape(-1, 3)
        bounds = self.lattice_parameters * self.dimensions

        logger.debug("lattice sites initialized", extra={"num_sites": len(lattice_points), "bounds": bounds.tolist()})

        return lattice_points, bounds


@dataclass
class KthNearest:

    r"""
    Container class for $k$'th nearest interactions.

    This class computes the interaction matrix $\mathbf{Q}$, acting as an energy driver for the growth simulation.

    Attributes:

        cutoffs (np.ndarray):

            Interaction distance cutoffs $(\delta_1, \delta_2, \cdots, \delta_k)$. For example, two molecules with
            interaction distance between $0$ and $\delta_1$ will interact with energy $\varepsilon_1$, two molecules
            with interaction distance between $\delta_1$ and $\delta_2$ will interact with energy $\varepsilon_2$, etc.

        interaction_energies (np.ndarray):

            Interaction energies $(\varepsilon_1, \varepsilon_2, \cdots, \varepsilon_k)$.

        maxint (int):

            Optional max tree size for [KDTree](https://en.wikipedia.org/wiki/K-d_tree) distance computation. If the
            number of molecules is larger than this value, `scipy.spatial.KDTree` will switch to brute-forcing the
            distance computations.

        use_cache (bool):

            Whether to use a caching mechanism or not. If true, the simulation will try and load $\mathbf{Q}$ from
            `.kmc_cache/`. If $\mathbf{Q}$ does not exist in the local cache already, it will be stored.
    """

    cutoffs: np.typing.NDArray[np.floating]
    r"""distance cutoffs $(\delta_1, \delta_2, \cdots, \delta_k)$"""

    interaction_energies: np.typing.NDArray[np.floating]
    r"""Interaction energies $(\varepsilon_1, \varepsilon_2, \cdots, \varepsilon_k)$"""

    maxint: Optional[int] = 10_000_000
    """maximum leaf size for `scipy.spatial.KDTree`"""

    use_cache: Optional[bool] = False
    r"""Whether to use the local cache or not to fetch $\mathbf{Q}$"""

    def __post_init__(self):

        if not isinstance(self.cutoffs, np.ndarray):
            self.cutoffs = np.array(self.cutoffs)

        if not isinstance(self.interaction_energies, np.ndarray):
            self.interaction_energies = np.array(self.interaction_energies)

        if len(self.cutoffs.shape) != 1:
            raise ValueError(
                f"Invalid interaction cutoffs. {self.__class__.__name__}.cutoffs should have shape "
                f"(number of neighbors,)"
            )

        if len(self.interaction_energies.shape) != 1:
            raise ValueError(
                f"Invalid interaction energies. {self.__class__.__name__}.interaction_energies should have shape "
                f"(number of neighbors,)"
            )

        if self.cutoffs.shape != self.interaction_energies.shape:
            raise ValueError(
                f"Incompatible interaction parameters. {self.__class__.__name__}.cutoffs and "
                f"{self.__class__.__name__}.interaction_energies should have the same shape."
            )

    def compute_hamiltonian(
        self,
        lattice_points: np.typing.NDArray[np.floating],
        bounds: np.typing.NDArray[np.floating]
    ) -> scipy.sparse.csr_matrix:

        r"""
        Compute the interaction matrix $\mathbf{Q}$ from the lattice points and bounds.

        Args:
            lattice_points (np.ndarray): Lattice points in the solid phase
            bounds (np.ndarray): Supercell bounds

        Returns:
            scipy.sparse.csr_matrix: $\mathbf{Q}$ in sparse format
        """

        tree = scipy.spatial.KDTree(lattice_points, leafsize=self.maxint, boxsize=bounds)

        distance_matrix = tree.sparse_distance_matrix(tree, max_distance=self.cutoffs[-1]).tocsr()
        distance_matrix.eliminate_zeros()

        interaction_types = np.searchsorted(self.cutoffs, distance_matrix.data, side="left")
        interaction_energies = self.interaction_energies[interaction_types]

        return scipy.sparse.csr_matrix(
            (interaction_energies, distance_matrix.indices, distance_matrix.indptr),
            shape=distance_matrix.shape
        )

    def get_hamiltonian(
        self,
        lattice_points: np.typing.NDArray[np.floating],
        bounds: np.typing.NDArray[np.floating]
    ) -> scipy.sparse.csr_matrix:

        r"""
        Get the interaction matrix $\mathbf{Q}$ from the lattice points and bounds. This either calls
        `KthNearest.compute_hamiltonian`, or loads $\mathbf{Q}$ from the user's local cache.

        Args:
            lattice_points (np.ndarray): Lattice points in the solid phase
            bounds (np.ndarray): Supercell bounds

        Returns:
            scipy.sparse.csr_matrix: $\mathbf{Q}$ in sparse format
        """

        if not self.use_cache:
            hamiltonian = self.compute_hamiltonian(lattice_points, bounds)

            logger.debug("hamiltonian initialized", extra={
                "num_interactions": hamiltonian.nnz, "cohesive_energy": 0.5 * hamiltonian.sum(axis=0).mean()
            })

            return hamiltonian

        cache_folder = Path(".kmc_cache")
        cache_folder.mkdir(exist_ok=True)
        hexes = [
            array_to_hex(self.cutoffs),
            array_to_hex(self.interaction_energies),
            array_to_hex(lattice_points),
            array_to_hex(bounds)
        ]
        hamiltonian_path = cache_folder / Path(f"{'_'.join(hexes)}.npz")

        if not hamiltonian_path.exists():
            hamiltonian = self.compute_hamiltonian(lattice_points, bounds)
            scipy.sparse.save_npz(hamiltonian_path, hamiltonian)

            logger.debug("hamiltonian initialized and saved", extra={
                "num_interactions": hamiltonian.nnz, "cohesive_energy": 0.5 * hamiltonian.sum(axis=0).mean(),
                "cache_path": hamiltonian_path.name
            })

            return hamiltonian

        hamiltonian = scipy.sparse.load_npz(hamiltonian_path)

        logger.debug("hamiltonian loaded from cache", extra={
            "num_interactions": hamiltonian.nnz, "cohesive_energy": 0.5 * hamiltonian.sum(axis=0).mean(),
            "cache_path": hamiltonian_path.name
        })

        return hamiltonian
