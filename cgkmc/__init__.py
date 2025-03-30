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

WIP: Include OVITO visualization screenshots

```py
.. include:: ../examples/ovito_visualization.py
```

## 📝 Using a logger

Below is an example of using a logger to grab information from the simulation. `cgkmc` uses Python's native `logging`
library, so one can use a `logging` config to grab simulation information.

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

WIP: Describe PETN, include PETN code, and visualizations (surface energy figure, final morphology, etc)

Replace placeholder video

<p align="center">
    <video width="500px" height="400px" controls>
        <source src="https://raw.githubusercontent.com/jwjeffr/cgkmc/refs/heads/main/examples/petn.mp4" type="video/mp4">
    </video>
</p>

"""

__version__ = "0.0.6"
__authors__ = ["Jacob Jeffries"]
__author_emails__ = ["jwjeffr@clemson.edu"]
__url__ = "https://github.com/jwjeffr/cgkmc"

from . import containers as containers
from . import simulations as simulations
from . import utils as utils
