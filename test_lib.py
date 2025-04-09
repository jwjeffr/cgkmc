from enum import Enum, auto
from typing import List, Tuple
from tempfile import TemporaryFile

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


def test_simulation_run(first_nn_simulation):

    sim, _ = first_nn_simulation
    with TemporaryFile() as file:
        sim.perform(dump_file=file, dump_every=100)


def test_zero_radius_errors_out(first_nn_simulation):

    with pytest.raises(ValueError):
        _ = containers.Growth(initial_radius=0.0, num_steps=1_000, desired_size=4_000)


def test_zero_steps_errors_out(first_nn_simulation):

    with pytest.raises(ValueError):
        _ = containers.Growth(initial_radius=20.0, num_steps=0, desired_size=4_000)


def test_non_integer_steps_errors_out(first_nn_simulation):

    with pytest.raises(ValueError):
        _ = containers.Growth(initial_radius=20.0, num_steps=0.1, desired_size=4_000)


def test_zero_desired_size_errors_out(first_nn_simulation):

    with pytest.raises(ValueError):
        _ = containers.Growth(initial_radius=20.0, num_steps=1_000, desired_size=0)


def test_non_integer_desired_size_errors_out(first_nn_simulation):

    with pytest.raises(ValueError):
        _ = containers.Growth(initial_radius=20.0, num_steps=1_000, desired_size=0.1)


def test_bad_dimensions(first_nn_simulation):

    with pytest.raises(ValueError):
        _ = containers.CubicLattice(
            dimensions=[[5, 5, 5]],
            lattice_parameters=(1.0, 1.0, 1.0),
            atomic_basis=[[0.0, 0.0, 0.0]]
        )


def test_bad_lattice_parameters(first_nn_simulation):

    with pytest.raises(ValueError):
        _ = containers.CubicLattice(
            dimensions=(5, 5, 5),
            lattice_parameters=[[1.0, 1.0, 1.0]],
            atomic_basis=[[0.0, 0.0, 0.0]]
        )


def test_bad_atomic_basis(first_nn_simulation):

    with pytest.raises(ValueError):
        _ = containers.CubicLattice(
            dimensions=(5, 5, 5),
            lattice_parameters=(1.0, 1.0, 1.0),
            atomic_basis=[0.0, 0.0, 0.0]
        )


def test_zero_volume_errors_out():

    with pytest.raises(ValueError):
        _ = containers.CubicLattice(
            dimensions=(5, 5, 5),
            lattice_parameters=(1.0, 1.0, 0.0),
            atomic_basis=[[0.0, 0.0, 0.0]]
        )


def test_dimension_mismatch_errors_out():

    with pytest.raises(ValueError):
        _ = containers.CubicLattice(
            dimensions=(5, 5, 5),
            lattice_parameters=(1.0, 1.0),
            atomic_basis=[[0.0, 0.0, 0.0]]
        )


def test_bad_cutoffs_errors_out():

    with pytest.raises(ValueError):
        _ = containers.KthNearest(
            cutoffs=[[0.5, 1.0]],
            interaction_energies=(-1.0, -0.5)
        )


def test_bad_interactions_errors_out():

    with pytest.raises(ValueError):
        _ = containers.KthNearest(
            cutoffs=(0.5, 1.0),
            interaction_energies=[[-1.0, -0.5]]
        )


def test_cutoffs_interactions_mismatch_errors_out():

    with pytest.raises(ValueError):
        _ = containers.KthNearest(
            cutoffs=(0.5, 1.0, 1.5),
            interaction_energies=(-1.0, -0.5)
        )


def test_non_periodic_sends_warning(first_nn_simulation):

    sim, _ = first_nn_simulation
    sim.adjacency_matrix[0, 1] = 0.0
    sim.adjacency_matrix[1, 0] = 0.0
    sim.adjacency_matrix.eliminate_zeros()
    with pytest.warns():
        _ = sim.num_neighbors


def test_kappa_missing_surface_density_errors_out(first_nn_simulation):

    sim, _ = first_nn_simulation
    with pytest.raises(ValueError):
        _ = sim.kappa


def test_invalid_unit_system_errors_out():

    with pytest.raises(ValueError):

        _ = utils.Units.boltzmann_constant("fake-unit-system")


def test_negative_temperature_sends_warning():

    with pytest.warns():
        _ = utils.temp_to_beta(temperature=-1, units=utils.Units.metal)


def test_zero_temperature_sends_warning():

    with pytest.warns():
        beta = utils.temp_to_beta(temperature=0.0, units=utils.Units.metal)

    assert beta == np.inf
