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
    Temperature (or $\beta$), diffusivity, and solubility limit fully define the solvent.
    $\beta$ should match the units of the interaction energies you specify.
    See `utils.Units` class for available energy units, and use temp_to_beta() to get $\beta$ from temperature in K
    """

    beta: float
    diffusivity: float
    solubility_limit: float


@dataclass
class Growth:
    """
    Growth container class.
    Mimics experimental controls, i.e. initial crystal size, amount of time we grow, and the final size we want.
    """

    initial_radius: float
    num_steps: int
    desired_size: int

    @property
    def initial_surface_area(self) -> float:
        r"""
        Shortcut property for computing surface area.
        Crystal is assumed to be spherical, so surface area = $4\pi\times (\text{radius})^2$
        """

        return 4.0 * np.pi * self.initial_radius ** 2


@dataclass
class CubicLattice:
    """
    CubicLattice
    """

    dimensions: np.typing.NDArray[np.integer]
    lattice_parameters: np.typing.NDArray[np.floating]
    atomic_basis: np.typing.NDArray[np.floating]

    def __post_init__(self):
        # turn objects into tensors if they're not already tensors
        if not isinstance(self.dimensions, np.ndarray):
            self.dimensions = np.array(self.dimensions, dtype=int)
        if not isinstance(self.lattice_parameters, np.ndarray):
            self.lattice_parameters = np.array(self.lattice_parameters, dtype=float)
        if not isinstance(self.atomic_basis, np.ndarray):
            self.atomic_basis = np.array(self.atomic_basis, dtype=float)

    @property
    def density(self) -> float:

        return self.atomic_basis.shape[0] / np.prod(self.lattice_parameters)

    @property
    def molecular_volume(self) -> float:

        return 1.0 / self.density

    def initialize_simulation(self) -> Tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:

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

    cutoffs: np.typing.NDArray[np.floating]
    interaction_energies: np.typing.NDArray[np.floating]
    maxint: Optional[int] = None
    use_cache: Optional[bool] = False

    def __post_init__(self):

        if not self.maxint:
            self.maxint = 10_000_000

        if not isinstance(self.cutoffs, np.ndarray):
            self.cutoffs = np.array(self.cutoffs)

        if not isinstance(self.interaction_energies, np.ndarray):
            self.interaction_energies = np.array(self.interaction_energies)

    def compute_hamiltonian(
        self,
        lattice_points: np.typing.NDArray[np.floating],
        bounds: np.typing.NDArray[np.floating]
    ) -> scipy.sparse.csr_matrix:

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
