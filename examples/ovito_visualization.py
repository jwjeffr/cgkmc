from pathlib import Path
import gzip

from cgkmc import simulations, containers, utils


def gzip_file(path: Path):
    with path.open("rb") as f_in, gzip.open(f"{path}.gz", "wb") as f_out:
        f_out.writelines(f_in)
    path.unlink()


def main():

    simulation = simulations.Simulation(
        lattice=containers.CubicLattice(
            dimensions=(20, 20, 20),
            lattice_parameters=(3.0, 3.0, 3.0),
            atomic_basis=[[0.0, 0.0, 0.0]]
        ),
        interactions=simulations.KthNearest(
            cutoffs=(3.3,),
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
            num_steps=25_000,
            desired_size=1_000
        )
    )

    dump_path = Path("cube.dump")
    with dump_path.open("w") as file:
        simulation.perform(file, dump_every=100)
    gzip_file(dump_path)


if __name__ == "__main__":

    main()
