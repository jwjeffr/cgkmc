"""
.. include:: ../README.md

# Examples

## 📦 Simple cubic growth

Below is an example of the growth of a small simple cubic lattice, printing a LAMMPS-style dump string

```py
.. include:: ../examples/simple_cubic.py
```

## ⚛ Visualizing with OVITO

WIP: Include OVITO visualization screenshots

```py
.. include:: ../examples/ovito_visualization.py
```

## 📝 Using a logger

Below is an example of using a logger to grab information from the simulation. `cgkmc` uses Python's native `logging`
library, so one can use a `logging` config to grab simulation information.

```py
.. include:: ../examples/using_logger.py
```

"""

__version__ = "0.0.4"
__authors__ = ["Jacob Jeffries"]
__author_emails__ = ["jwjeffr@clemson.edu"]
__url__ = "https://github.com/jwjeffr/cgkmc"

from . import containers as containers
from . import simulations as simulations
from . import utils as utils
