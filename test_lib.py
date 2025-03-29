import numpy as np
import pytest

from cgkmc import containers, simulations, utils


@pytest.fixture
def sc_1nn_simulation() -> simulations.Simulation:

    return simulations.Simulation(
        lattice=containers.CubicLattice(
            dimensions=(5, 5, 5),
            lattice_parameters=(1.0, 1.0, 1.0),
            atomic_basis=[
                [0.0, 0.0, 0.0]
            ]
        ),
        interactions=simulations.KthNearest(
            cutoffs=(1.1,),
            interaction_energies=(-1.0,),
            use_cache=True
        ),
        solvent=containers.Solvent(
            beta=utils.temp_to_beta(temperature=300, units=utils.Units.metal),
            diffusivity=1.0e+11,
            solubility_limit=1.0e-4
        ),
        growth=containers.Growth(
            initial_radius=20.0,
            num_steps=1_000,
            desired_size=4_000
        )
    )


def test_coordination_numbers_sc(sc_1nn_simulation: simulations.Simulation):

    first_coord_nums = sc_1nn_simulation.adjacency_matrix.sum(axis=0)
    assert np.all(np.isclose(first_coord_nums.mean(), 6))
