from io import StringIO

from cgkmc import simulations, containers, utils


def main():

    simulation = simulations.Simulation(
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
            initial_radius=2.0,
            num_steps=1_000,
            desired_size=4_000
        )
    )

    with StringIO() as file:
        simulation.perform(file, dump_every=100)
        out = file.getvalue()

    print(out)


if __name__ == "__main__":

    main()
