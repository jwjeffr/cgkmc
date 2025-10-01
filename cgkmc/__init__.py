r"""
.. include:: ../README.md

# Examples

## 📦 Simple cubic growth

Below is an example of the growth of a small simple cubic lattice, and outputting the simulation info to stdout

```py
.. include:: ../examples/simple_cubic.py
```

This prints the simulation dump in LAMMPS-style dump format:

```
ITEM: TIMESTEP
0 0.0
ITEM: NUMBER OF ATOMS
33
ITEM: BOX BOUNDS xy xz xx yy zz
0.0 5.0 0.0
0.0 5.0 0.0
0.0 5.0 0.0
ITEM: ATOMS id type x y z
12 1 0.0000 2.0000 2.0000
31 1 1.0000 1.0000 1.0000
32 1 1.0000 1.0000 2.0000
...
ITEM: TIMESTEP
100 2.39212295038683e-07
ITEM: NUMBER OF ATOMS
95
ITEM: BOX BOUNDS xy xz xx yy zz
0.0 5.0 0.0
0.0 5.0 0.0
0.0 5.0 0.0
ITEM: ATOMS id type x y z
0 1 0.0000 0.0000 0.0000
1 1 0.0000 0.0000 1.0000
2 1 0.0000 0.0000 2.0000
...
```

and so on. This can then be redirected to a file, and visualized by many trajectory viewing softwares, including
[Open Visualization Tool](https://www.ovito.org/) (OVITO),
[Visual Molecular Dynamics](https://www.ks.uiuc.edu/Research/vmd/) (VMD), and
[more](https://en.wikipedia.org/wiki/List_of_molecular_graphics_systems).

## ⚛ Visualizing with OVITO

Below is a simple example of growing a simple cubic crystal starting from a spherical seed, dumping to a file
`cube.dump`, and compressing the dump file to `cube.dump.gz`.

```py
.. include:: ../examples/ovito_visualization.py
```

This dump file (and its compressed version) can be loaded into OVITO for visualization! Simply:

- Open OVITO
- Load the dump file (`File` ➡ `Load File` ➡ `cube.dump.gz` ➡ `Open`)

Your window should look like this:

<p align="center">
<img src="https://raw.githubusercontent.com/jwjeffr/cgkmc/refs/heads/main/examples/ovito_steps/step1.jpg" alt="Loading in dump file" style="width: 50%;">
</p>

The particles here are not particularly useful for visualization. Instead, we can create a surface mesh. Click:

- `Add modification` (top right, under the dump file name)
- `Construct surface mesh` (Scroll down to `Visualization` category)

Your window should now look like this:

<p align="center">
<img src="https://raw.githubusercontent.com/jwjeffr/cgkmc/refs/heads/main/examples/ovito_steps/step2.jpg" alt="Constructing surface mesh" style="width: 50%;">
</p>

The particles here are still a bit of an eyesore. You can get rid of them by clicking `Particles` under the
`Visual elements` tab, also under the dump file name. After doing this, your window should look like this:

<p align="center">
<img src="https://raw.githubusercontent.com/jwjeffr/cgkmc/refs/heads/main/examples/ovito_steps/step3.jpg" alt="Hiding particles" style="width: 50%;">
</p>

and now this looks much better! You can now press the play button (bottom center) to view the animation. Your window at
the final frame should look like this:

<p align="center">
<img src="https://raw.githubusercontent.com/jwjeffr/cgkmc/refs/heads/main/examples/ovito_steps/step4.jpg" alt="Final frame" style="width: 50%;">
</p>

This final frame makes sense! The $\\{100\\}$ surfaces have the lowest surface energy in simple cubic crystals with
first nearest neighbor interactions, so the crystal should grow into something roughly rectangular.

## 📝 Using a logger

Below is an example of using a logger to grab information from the simulation. `cgkmc` uses Python's native `logging`
library (with docs [here](https://docs.python.org/3/library/logging.html)), so one can use a `logging` config to grab
simulation information.

```py
.. include:: ../examples/writing_log.py
```

Here, `occupancy` is the proportion of occupied sites. This creates a log file that looks like:

```
2025-03-30 12:36:26,790 INFO:simulation info TIME=6.758222371127191e-06 ENERGY=-3518.0 OCCUPANCY=0.491
2025-03-30 12:36:26,817 INFO:simulation info TIME=6.787981905980638e-06 ENERGY=-3547.0 OCCUPANCY=0.494
2025-03-30 12:36:26,843 INFO:simulation info TIME=6.813314575607091e-06 ENERGY=-3559.0 OCCUPANCY=0.496
```

and so on. This can be parsed using Python's built-in
[regular expression](https://en.wikipedia.org/wiki/Regular_expression) parser, and the results can be visualized:

```py
.. include:: ../examples/parsing_log.py
```

which generates the plot below:

<p align="center">
<img src="https://raw.githubusercontent.com/jwjeffr/cgkmc/refs/heads/main/examples/energy.png" alt="Energy vs. time">
</p>

You can get very creative with this! My preferred is logging in [JSON Lines format](https://jsonlines.org/).
mCoding has a very great YouTube video on this, which is [here](https://www.youtube.com/watch?v=9L77QExPmI0).

## 🧪 Case study: PETN

### What is PETN?

[Pentaerythritol tetranitrate](https://en.wikipedia.org/wiki/Pentaerythritol_tetranitrate) (PETN) is a very well-studied
organic crystalline system. Here, we will run `cgkmc` on a PETN system to predict the morphology of a PETN nanocrystal.

### PETN Parameters

Following work by Singh. et al. [here](https://doi.org/10.1021/acs.cgd.3c01487), we can reduce PETN molecules to
individual effective atoms inside a body-tetragonal unit cell, with lattice parameters $a = b = 9.087\;\text{Å}$,
$c = 6.738\;\text{Å}$, and "atomic" basis $\\{(0, 0, 0), (1/2, 1/2, 1/2)\\}$. Atomic is in quotes here because
the particles are molecules, not atoms. Additionally, we can use the interaction energies `(-0.294, -0.184, -0.002)` and
corresponding cutoff distances `(7.0, 7.5, 9.5)`, respectively in $\text{eV}$ and $\text{Å}$.

### Script

A script for the simulation of such a system is below:

```py
from pathlib import Path

from cgkmc import simulations, containers, utils


def main():

    simulation = simulations.Simulation(
        lattice=containers.CubicLattice(
            dimensions=(62, 62, 140),
            lattice_parameters=(9.087, 9.087, 6.738),
            atomic_basis=[
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5]
            ]
        ),
        interactions=simulations.KthNearest(
            cutoffs=(7.0, 7.5, 9.5),
            interaction_energies=(-0.294, -0.184, -0.002),
            use_cache=True
        ),
        solvent=containers.Solvent(
            beta=utils.temp_to_beta(temperature=300, units=utils.Units.metal),
            diffusivity=1.0e+11,
            solubility_limit=1.0e-4
        ),
        growth=containers.Growth(
            initial_radius=75.0,
            num_steps=1_000_000,
            desired_size=40_000
        )
    )

    with Path("petn.dump").open("w") as file:
        simulation.perform(file, dump_every=1_000)


if __name__ == "__main__":

    main()
```

This generates a dump file that we can analyze in OVITO! The animation is below:

<div style="padding:75% 0 0 0;position:relative;">
<iframe
    src="https://player.vimeo.com/video/1078144061?h=446c9e7099&amp;badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media" style="position:absolute;top:0;left:0;width:100%;height:100%;"
    title="petn"
>
</iframe>
</div>
<script src="https://player.vimeo.com/api/player.js">
</script>

Some clear crystal morphology jumps out, namely $\\{110\\}$ surfaces. See a more full study
[here](https://arxiv.org/abs/2509.25490)!

## ⌨ Using the command line interface

We also provide an option to use a command-line interface (CLI)! For example, for the simple cubic example from earlier,
the input file looks like:

```json
.. include:: ../examples/simple_cubic.json
```

and the resulting CLI call looks like:

```bash
cgkmc --input_file input_file.json --dump_file dump_file.dump --dump_every 1000
```

which runs a simulation, and dumps molecular coordinates every 1000 steps. You can also specify an optional log file
with `--log_file`, which gives you access to dynamical variables like occupancy and energy.

This CLI automatically comes with `cgkmc`. No additional building is necessary! You can also run:

```bash
cgkmc --help
```

for more info!
"""

__version__ = "0.1.1"
__authors__ = ["Jacob Jeffries"]
__author_emails__ = ["jwjeffr@clemson.edu"]
__url__ = "https://github.com/jwjeffr/cgkmc"

from . import containers as containers
from . import simulations as simulations
from . import utils as utils

import argparse
from pathlib import Path
import json
import logging


def cli():

    """
    @private
    """

    parser = argparse.ArgumentParser(
        prog="cgkmc",
        description="Command line interface for Crystal Growth Kinetic Monte Carlo (cgkmc)",
        epilog="See https://jwjeffr.github.io/cgkmc/ for full documentation!"
    )
    parser.add_argument("-i", "--input_file", type=Path, help="path to input json file")
    parser.add_argument("-d", "--dump_file", type=Path, help="file to dump coordinates to"),
    parser.add_argument("-n", "--dump_every", type=int, help="how often to dump coordinates")
    parser.add_argument("-l", "--log_file", type=Path, help="optional log file", required=False, default=None)
    args = parser.parse_args()

    if args.log_file:
        with args.log_file.open("w"):
            pass
        logging.basicConfig(
            filename=args.log_file,
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s TIME=%(t)s ENERGY=%(total_energy)s OCCUPANCY=%(occupancy)s'
        )

    with args.input_file.open("r") as file:
        input_dict = json.load(file)

    simulation = simulations.Simulation(
        lattice=containers.CubicLattice(**input_dict["lattice"]),
        interactions=containers.KthNearest(**input_dict["interactions"]),
        solvent=containers.Solvent(**input_dict["solvent"]),
        growth=containers.Growth(**input_dict["growth"])
    )

    with args.dump_file.open("w") as file:
        simulation.perform(dump_file=file, dump_every=args.dump_every)
