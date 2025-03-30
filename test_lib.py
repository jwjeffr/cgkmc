from enum import Enum, auto
from typing import List, Tuple

import numpy as np
import pytest

from cgkmc import containers, simulations, utils


class Structure(Enum):

    SC = auto()
    BCC = auto()
    FCC = auto()

    @property
    def atomic_basis(self) -> List[List[float]]:
        if self == Structure.SC:
            return [[0.0, 0.0, 0.0]]
        if self == Structure.BCC:
            return [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        if self == Structure.FCC:
            return [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
        raise ValueError

    @property
    def first_nn_cutoff(self) -> float:
        if self == Structure.SC:
            return 1.1
        if self == Structure.BCC:
            return 0.9
        if self == Structure.FCC:
            return 0.8
        raise ValueError

    @property
    def first_coord_num(self) -> int:
        if self == Structure.SC:
            return 6
        if self == Structure.BCC:
            return 8
        if self == Structure.FCC:
            return 12
        raise ValueError


@pytest.fixture
def solvent() -> containers.Solvent:

    beta = utils.temp_to_beta(temperature=300, units=utils.Units.metal)
    return containers.Solvent(beta=beta, diffusivity=1.0e+11, solubility_limit=1.0e-4)


@pytest.fixture
def growth() -> containers.Growth:

    return containers.Growth(initial_radius=20.0, num_steps=1_000, desired_size=4_000)


@pytest.fixture(params=[Structure.SC, Structure.BCC, Structure.FCC])
def first_nn_simulation(
    request,
    solvent: containers.Solvent,
    growth: containers.Growth
) -> Tuple[simulations.Simulation, int]:
    sim = simulations.Simulation(
        lattice=containers.CubicLattice(
            dimensions=(5, 5, 5),
            lattice_parameters=(1.0, 1.0, 1.0),
            atomic_basis=request.param.atomic_basis
        ),
        interactions=simulations.KthNearest(
            cutoffs=(request.param.first_nn_cutoff,),
            interaction_energies=(-1.0,),
            use_cache=True
        ),
        solvent=solvent,
        growth=growth
    )
    return sim, request.param.first_coord_num


def test_coordination_numbers(first_nn_simulation):
    sim, expected_coord_num = first_nn_simulation
    first_coord_nums = sim.adjacency_matrix.sum(axis=0)
    assert np.all(np.isclose(first_coord_nums.mean(), expected_coord_num))
